from typing import Callable, Optional, Union
from sparsemax import Sparsemax

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

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
from torch_scatter import scatter_max, scatter_add, scatter_mean


def fibonacci_sphere(n: int, radius: float = 1.0, dim: int = 3) -> Tensor:
    if n == 1:
        return torch.zeros(1, dim, dtype=torch.float32)
    origin = torch.zeros(1, dim, dtype=torch.float32)
    indices = torch.arange(n - 1, dtype=torch.float32)
    phi = (indices + 0.5) * (torch.pi * (3 - torch.sqrt(torch.tensor(5.0))))
    y = 1 - (indices / float(n - 2)) * 2
    r = torch.sqrt(1 - y**2)
    x = r * torch.cos(phi)
    z = r * torch.sin(phi)
    sphere_points = torch.stack([x, y, z], dim=1) * radius
    if dim > 3:
        extra = torch.zeros(sphere_points.size(0), dim - 3, dtype=sphere_points.dtype)
        sphere_points = torch.cat([sphere_points, extra], dim=1)
    return torch.cat([origin, sphere_points], dim=0)

class AnisotropicConv(MessagePassing):
    def __init__(self,
                 local_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 num_kernel_points: int = 16,
                 radius: Optional[float] = None,
                 add_self_loops: bool = True,
                 learnable_kernels: bool = False,
                 use_sparsemax: bool = True,
                 learnable_rho: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops
        self.radius = radius
        self.use_sparsemax = use_sparsemax

        self._learnable_rho = learnable_rho
        
        if learnable_rho:
            self.raw_rho = nn.Parameter(torch.zeros(1))

        self.rho_min = 0.1
        self.rho_max = 1.0

        base_kernels = fibonacci_sphere(num_kernel_points, dim=3)
        base_kernels[0, :] = 0.0
        self.kernel_points = nn.Parameter(base_kernels, requires_grad=learnable_kernels)
        
        self.kernel_reflectance_importance = nn.Parameter(torch.ones(num_kernel_points))  # Per-kernel reflectance importance
        
        self.kernel_bias = nn.Parameter(torch.zeros(num_kernel_points))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)
        nn.init.zeros_(self.kernel_bias)

        if self._learnable_rho:
            nn.init.zeros_(self.raw_rho)

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
                num_nodes = min(pos[0].size(0), pos[1].size(0))
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        out = self.propagate(edge_index, x=x, pos=pos)

        return out

    def message(self, x_j: OptTensor, pos_i: Tensor, pos_j: Tensor, index: Tensor) -> Tensor:
        rel_pos = pos_j[:, :3] - pos_i[:, :3]
        dists = torch.norm(rel_pos, dim=1, keepdim=True)
        max_d, _ = scatter_max(dists, index, dim=0)
        
        if self._learnable_rho:
            rho = self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(self.raw_rho)
        else:
            rho = 1.0

        rel_pos_norm = rel_pos / (rho * max_d[index] + 1e-8)

        refl_j = pos_j[:, 3:4]
        rel_refl = refl_j - pos_i[:, 3:4]
            
        attention_features = torch.cat([
            rel_pos_norm,
            refl_j,
            rel_refl
        ], dim=-1)
            
        kernel_dirs = F.normalize(self.kernel_points, dim=1).unsqueeze(0)
        
        if pos_j.size(1) > 3:
            kernel_padding = torch.zeros(1, kernel_dirs.size(1), 2, device=kernel_dirs.device)
            kernel_dirs = torch.cat([kernel_dirs, kernel_padding], dim=-1)

        attn = torch.sum(attention_features.unsqueeze(1) * kernel_dirs, dim=-1)
        weights = Sparsemax(dim=1)(attn)

        self_indices = (dists.squeeze(-1) < 1e-8).nonzero(as_tuple=True)[0]
        if len(self_indices) > 0:
            weights = weights.clone()
            weights[self_indices] = 0.0
            weights[self_indices, 0] = 1.0

        feat_list = []
        if x_j is not None:
            feat_list.append(x_j)
        
        #feat_list.append(pos_j[:, 3].unsqueeze(-1))

        reflectance_weight = torch.sum(weights * self.kernel_reflectance_importance.unsqueeze(0), dim=1)  # Kernel-weighted reflectance importance
        weighted_reflectance = (pos_j[:, 3] * reflectance_weight).unsqueeze(-1)
        feat_list.append(weighted_reflectance)

        #

        feat_list.append(rel_pos)
        feat_list.append(dists)
        feat = torch.cat(feat_list, dim=-1).unsqueeze(-1)

        weighted = feat * weights.unsqueeze(1)

        if self.training:
            active_kernels = (weights > 0).float().mean(dim=0)
            entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1)
            self.kernel_entropy = entropy.mean()
            self.active_kernels = active_kernels.mean()

        return weighted

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None, dim_size: Optional[int] = None) -> Tensor:
        agg = scatter_add(inputs, index, dim=0, dim_size=dim_size)
        agg = agg.transpose(1, 2).contiguous().view(dim_size, -1)
        if self.local_nn is not None:
            agg = self.local_nn(agg)
        return agg

    def update(self, aggr_out: Tensor) -> Tensor:
        if self.global_nn is not None:
            aggr_out = self.global_nn(aggr_out)
        return aggr_out