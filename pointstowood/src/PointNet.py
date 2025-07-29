from typing import Callable, Optional, Union

import torch
from torch import Tensor

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
from torch_scatter import scatter_mean, scatter_max, scatter_std


class PointNetConv(MessagePassing):

    def __init__(self, local_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 add_self_loops: bool = True,
                 radius: Optional[float] = None, **kwargs):
        
        kwargs.setdefault('aggr', 'max')
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.radius = radius
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


    def message(self, x_j: Optional[Tensor], pos_i: Tensor,
                    pos_j: Tensor, edge_index_i: Tensor) -> Tensor:
            
            msg = torch.zeros((pos_j.size(0), pos_j.size(1)), device=pos_j.device)
            
            relative_pos = (pos_j[:, :3] - pos_i[:, :3]) 

            if self.radius is not None:
                relative_pos = relative_pos / self.radius
            else:
                max_distances, _ = scatter_max(torch.norm(relative_pos, dim=1, keepdim=True), edge_index_i, dim=0)
                msg[:, :3] = relative_pos / (max_distances[edge_index_i] + 1e-8)
            
            msg[:, 3] = pos_j[:, 3]
        
            if x_j is not None:
                msg = torch.cat([x_j, msg], dim=1)
            if self.local_nn is not None:
                msg = self.local_nn(msg)
            return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(local_nn={self.local_nn}, '
                f'global_nn={self.global_nn})')
    