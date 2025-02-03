from typing import Callable, Optional, Union

import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairOptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter_max

class PointNetConv(MessagePassing):
    
    def __init__(self, local_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 shape_kernel: Optional[Callable] = None,
                 attention_nn: Optional[Callable] = None,
                 add_self_loops: bool = True, **kwargs):
        
        self.radius = kwargs.pop('radius', None)
        
        kwargs.setdefault('aggr', 'max')
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.shape_kernel = shape_kernel
        self.attention_nn = attention_nn
        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.local_nn)
        reset(self.global_nn)
        reset(self.shape_kernel)

    def forward(
        self,
        x: Union[OptTensor, PairOptTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
        batch: Tensor,
    ) -> Tensor:

        if not isinstance(x, tuple):
            x = (x, None)

        if isinstance(pos, Tensor):
            pos = (pos, pos)

        if isinstance(batch, tuple):
            batch = batch[0]
        if batch.dim() > 1:
            batch = batch[0]
        

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)
        
        out = self.propagate(edge_index, x=x, pos=pos, batch=batch)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out


    def message(self, x_i: Optional[Tensor], x_j: Optional[Tensor], 
               pos_i: Tensor, pos_j: Tensor, edge_index_i: Tensor, batch: Tensor) -> Tensor:
        
        relative_pos = pos_j[:, :3] - pos_i[:, :3] 
        distances = torch.norm(relative_pos, dim=1, keepdim=True) + 1e-8
        max_distances = scatter_max(distances, edge_index_i, dim=0)[0][edge_index_i]
        relative_pos = relative_pos / max_distances
        msg = relative_pos
        shape_features = self.shape_kernel(relative_pos, edge_index_i)  
        msg = torch.cat([msg, shape_features], dim=-1)

        if x_j is not None:            
            msg = torch.cat([msg, (x_j - (x_i if x_i is not None else 0) + 1e-8)], dim=-1)

        msg = torch.cat([msg, pos_j[:, 3].unsqueeze(-1)], dim=-1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(local_nn={self.local_nn}, '
                f'global_nn={self.global_nn})')


