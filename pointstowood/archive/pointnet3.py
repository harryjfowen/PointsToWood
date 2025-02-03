from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn
from torch_scatter import scatter_mean, scatter_max, scatter_std, scatter_add, scatter_softmax

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

class AttentionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        query = self.query(x_i)
        key = self.key(x_j)
        value = self.value(x_j)
        
        attention_scores = torch.sum(query * key, dim=1, keepdim=True)
        attention_weights = self.softmax(attention_scores)
        
        attended_values = attention_weights * value
        return attended_values

class PointNetConv(MessagePassing):
    
    def __init__(self, local_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 add_self_loops: bool = True, **kwargs):
        
        self.radius = kwargs.pop('radius', None)

        kwargs.setdefault('aggr', 'sum')
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.local_nn)
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


    def message(self, x_i: Optional[Tensor], x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor, edge_index_i: Tensor) -> Tensor:
        msg = torch.zeros((pos_j.size(0), pos_j.size(1)), device=pos_j.device)
        
        relative_pos = pos_j[:, :3] - pos_i[:, :3]
        
        max_distances, _ = scatter_max(relative_pos.abs().max(dim=1, keepdim=True)[0], edge_index_i, dim=0)
        max_distances = max_distances[edge_index_i]
        max_distances = torch.clamp(max_distances, min=1e-8)
        
        msg[:, :3] = relative_pos / max_distances
        
        if not torch.all(pos_j[:, 3] == 0):
            msg[:, 3] = pos_j[:, 3]

        if x_j is not None and x_i is not None:
            attended_values = self.attention_layer(x_i, x_j)
            msg = torch.cat([attended_values, msg], dim=1)
            if torch.isnan(attended_values).any():
                raise ValueError("NaN values found in attended_values after attention")
        else:
            if x_j is not None:
                msg = torch.cat([x_j, msg], dim=1)

        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(local_nn={self.local_nn}, '
                f'global_nn={self.global_nn})')

class PointNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(PointNet, self).__init__()
        self.attention_layer = AttentionLayer(in_channels, out_channels)
        self.local_nn = nn.Sequential(
            nn.Linear(out_channels + 4, 128),  # Adjust the input features to match the concatenated msg
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

    def message(self, x_i: Optional[Tensor], x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor, edge_index_i: Tensor) -> Tensor:
        msg = torch.zeros((pos_j.size(0), pos_j.size(1)), device=pos_j.device)
        
        relative_pos = pos_j[:, :3] - pos_i[:, :3]
        
        max_distances, _ = scatter_max(relative_pos.abs().max(dim=1, keepdim=True)[0], edge_index_i, dim=0)
        max_distances = max_distances[edge_index_i]
        max_distances = torch.clamp(max_distances, min=1e-8)
        
        msg[:, :3] = relative_pos / max_distances
        
        if not torch.all(pos_j[:, 3] == 0):
            msg[:, 3] = pos_j[:, 3]
        
        if x_j is not None and x_i is not None:
            attended_values = self.attention_layer(x_i, x_j)
            msg = torch.cat([attended_values, msg], dim=1)
            if torch.isnan(attended_values).any():
                raise ValueError("NaN values found in attended_values after attention")
        else:
            if x_j is not None:
                msg = torch.cat([x_j, msg], dim=1)

        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def forward(self, x, pos, edge_index):
        return self.message(x, x[edge_index[1]], pos, pos[edge_index[1]], edge_index[0])


