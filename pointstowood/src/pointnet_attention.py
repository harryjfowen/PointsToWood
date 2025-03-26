from typing import Callable, Optional, Union
import math
import torch
from torch import Tensor
import torch_sparse

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor, SparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter_max, scatter_softmax

class AttentivePointNetConv(MessagePassing):
    def __init__(self, local_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 in_channels: int = 64,
                 attn_dim: int = 32,
                 attention_type: str = 'softmax',
                 add_self_loops: bool = True, **kwargs):
                
        kwargs.setdefault('aggr', 'max')
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops
        self.attention_type = attention_type

        # Ensure positive dimensions
        self.in_channels = max(1, in_channels)  # Prevent zero channels
        self.attn_dim = max(1, attn_dim)       # Prevent zero attention dim
        
        # Projections for attention
        self.query = torch.nn.Linear(self.in_channels, self.attn_dim)
        self.key = torch.nn.Linear(self.in_channels, self.attn_dim)
        self.value = torch.nn.Linear(self.in_channels, self.in_channels)
        
        # Layer normalization for stability
        self.layer_norm_q = torch.nn.LayerNorm(self.attn_dim)
        self.layer_norm_k = torch.nn.LayerNorm(self.attn_dim)
        self.layer_norm_v = torch.nn.LayerNorm(self.in_channels)
        
        # Output normalization
        self.layer_norm_out = torch.nn.LayerNorm(self.in_channels)
        
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        
        # Safely initialize attention layers
        def safe_init(layer):
            if layer is not None and layer.weight.size(0) > 0 and layer.weight.size(1) > 0:
                torch.nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

        # Initialize attention layers
        safe_init(self.query)
        safe_init(self.key)
        safe_init(self.value)

        # Reset other components if they exist
        if self.local_nn is not None:
            reset(self.local_nn)
        if self.global_nn is not None:
            reset(self.global_nn)

    def forward(
        self,
        x: Union[OptTensor, PairOptTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:
        if not isinstance(x, tuple):
            x = (x, None)
        if isinstance(pos, Tensor):
            pos = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        out = self.propagate(edge_index, x=x, pos=pos)
        
        if self.global_nn is not None:
            out = self.global_nn(out)
        
        return out

    def message(self, x_i: Optional[Tensor], x_j: Optional[Tensor], 
               pos_i: Tensor, pos_j: Tensor, edge_index_i: Tensor) -> Tensor:
        
        # Relative positions and distance weighting
        rel_pos = pos_j[:, :3] - pos_i[:, :3] 
        distances = torch.norm(rel_pos, dim=1, keepdim=True) + 1e-8
        max_distances = scatter_max(distances, edge_index_i, dim=0)[0][edge_index_i]
        rel_pos = rel_pos / max_distances

        if x_j is not None:
            # Compute attention with layer normalization
            q = self.layer_norm_q(self.query(x_i)) if x_i is not None else self.query(torch.zeros_like(x_j))
            k = self.layer_norm_k(self.key(x_j))
            v = self.layer_norm_v(self.value(x_j))

            # Attention scores
            scores = (q * k).sum(dim=-1) / math.sqrt(k.size(-1))
            
            # Apply either softmax or sigmoid
            if self.attention_type == 'softmax':
                attention = scatter_softmax(scores, edge_index_i)
            else:  # sigmoid
                attention = torch.sigmoid(scores)
            
            attention = attention.unsqueeze(-1)  # Add dimension for broadcasting

            # Weighted features with residual connection
            weighted_features = v * attention
            if x_i is not None:
                weighted_features = weighted_features + x_j  # residual connection
            
            # Final layer norm
            weighted_features = self.layer_norm_out(weighted_features)
            
            msg = torch.cat([rel_pos, pos_j[:, 3:4], weighted_features], dim=-1)
        else:
            msg = torch.cat([rel_pos, pos_j[:, 3:4]], dim=-1)
        
        return self.local_nn(msg) if self.local_nn is not None else msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(local_nn={self.local_nn}, '
                f'global_nn={self.global_nn})')

