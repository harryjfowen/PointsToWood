"""
AnisotropicConv: Wood/leaf point conv — geometry-first, reflectance as flashlight.

Geometry: Directional kernel routing (Fibonacci sphere) + density-normalised aggregation.
Flashlight: three complementary structure tensors. Each is a weighted outer-product average
  over the local neighbourhood, differing only in the per-edge weight:

  M_geom     = (1/k) Σ d̂⊗d̂                              — uniform (pure geometry)
  M_refl     = Σ c_j · d̂⊗d̂  /  Σ c_j                    — contrast-weighted
  M_cobright = Σ (c_max - c_j) · d̂⊗d̂  /  Σ(c_max - c_j) — complement-weighted

  where c_j = |r_j - r_i| (center-relative contrast) and c_max = max_j(c_j).

  Partition identity: W_j + W'_j = c_max for every edge, so M_refl and M_cobright are
  the conditional expectations of the outer-product tensor under the contrast measure and
  its complement. The unnormalised tensors satisfy:

      (Σ c_j) · M_refl  +  (Σ(c_max - c_j)) · M_cobright  =  k · c_max · M_geom

  i.e. M_refl and M_cobright reconstruct M_geom when mixed in proportion to their
  respective weight sums. The eigenvalue spectra of the three tensors are NOT related
  by this identity (eigenvalues are not closed under linear combination), so eigvals(M_geom)
  carries independent information and is kept in the flashlight.

  M_refl captures directions toward DISSIMILAR-brightness neighbours. For a bright twig
  center surrounded by dark leaves, leaf directions dominate M_refl (high c_j), while the
  co-bright wood chain (c_j ≈ 0) gets zero weight. M_refl is blind to the chain.

  M_cobright captures directions toward SIMILAR-brightness neighbours. For the same twig,
  wood chain neighbors (c_j ≈ 0) receive weight c_max while leaf neighbors (c_j ≈ c_max)
  receive weight ≈ 0. The twig chain axis becomes the dominant eigenvector of M_cobright
  even in a neighbourhood dominated by leaf returns — with no calibration constants.

  Calibration-free tensor construction: the weights use only relative contrast within
  each neighbourhood (c_max is the local dynamic range, not a global threshold or σ).
  The downstream flashlight_mlp carries the learned parameters.

Output per point: agg_feat (F*K) + flashlight (11 channels).
  flashlight = [eigvals(M_geom)(3), eigvals(M_refl)(3), eigvals(M_cobright)(3),
                mean_c(1), c_max(1)]
  mean_c and c_max together let the MLP distinguish a single specular outlier
  (large c_max, small mean_c) from a genuine wood/leaf interface (both large).
  All channels are unsigned and scale-invariant within each neighbourhood.
"""

import math
from typing import Callable, Optional, Union

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
from torch_scatter import scatter_max, scatter_add


def _eigvalsh_3x3_analytical(M: Tensor) -> Tensor:
    """Closed-form eigenvalues for batched 3x3 symmetric matrices.

    Uses Cardano's trigonometric solution — pure tensor ops, no LAPACK.
    ~3-5x faster than torch.linalg.eigvalsh for 3x3. Returns eigenvalues
    in ascending order [N, 3].

    Always computed in fp32: the intermediate sqrt/acos operations underflow
    silently in fp16/bf16 on near-isotropic tensors. AMP callers get fp32
    output cast back to their dtype.
    """
    dtype_in = M.dtype
    if dtype_in != torch.float32:
        M = M.float()

    a, b, c = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
    d, e    = M[:, 1, 1], M[:, 1, 2]
    f       = M[:, 2, 2]

    p1 = a + d + f
    q  = p1 / 3.0
    p2 = a * d - b * b + a * f - c * c + d * f - e * e
    p3 = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)

    pp = (3.0 * p2 - p1 * p1) / 9.0
    r  = (2.0 * p1 * p1 * p1 - 9.0 * p1 * p2 + 27.0 * p3) / 54.0

    phi_denom = (-pp).clamp(min=1e-8).sqrt().pow(3)
    cos_arg   = (r / phi_denom).clamp(-1.0, 1.0)
    phi       = torch.acos(cos_arg) / 3.0

    two_sqrt = 2.0 * (-pp).clamp(min=0.0).sqrt()
    e0 = q + two_sqrt * torch.cos(phi + (2.0 * math.pi / 3.0))
    e1 = q + two_sqrt * torch.cos(phi + (4.0 * math.pi / 3.0))
    e2 = q + two_sqrt * torch.cos(phi)

    eigvals = torch.stack([e0, e1, e2], dim=1)
    eigvals, _ = eigvals.sort(dim=1)
    return torch.nan_to_num(eigvals, nan=0.0, posinf=0.0, neginf=0.0).to(dtype_in)


def fibonacci_sphere(n: int, radius: float = 1.0) -> Tensor:
    """Kernel directions on unit sphere for routing."""
    if n == 1:
        return torch.zeros(1, 3, dtype=torch.float32)
    origin  = torch.zeros(1, 3, dtype=torch.float32)
    indices = torch.arange(n - 1, dtype=torch.float32)
    phi = (indices + 0.5) * (torch.pi * (3 - torch.sqrt(torch.tensor(5.0))))
    y   = 1 - (indices / float(n - 2)) * 2
    r   = torch.sqrt(1 - y ** 2)
    x   = r * torch.cos(phi)
    z   = r * torch.sin(phi)
    return torch.cat([origin, torch.stack([x, y, z], dim=1) * radius], dim=0)


class AnisotropicConv(MessagePassing):
    def __init__(self,
                 local_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 num_kernel_points: int = 16,
                 add_self_loops: bool = True,
                 learnable_kernels: bool = False,
                 use_softmax: bool = False,
                 softmax_temperature: float = 1.0,
                 flashlight_out_dim: int = 0,
                 memory_efficient: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops
        self.num_kernel_points = num_kernel_points

        base_kernels = fibonacci_sphere(num_kernel_points)
        base_kernels[0, :] = 0.0
        self.kernel_points = nn.Parameter(base_kernels, requires_grad=learnable_kernels)
        if not learnable_kernels:
            self.register_buffer('kernel_dirs', F.normalize(base_kernels.clone(), dim=1))
        else:
            self.register_buffer('kernel_dirs', None)

        self.use_softmax = use_softmax
        self.softmax_temperature = float(max(1e-3, softmax_temperature))
        self.memory_efficient = bool(memory_efficient)

        if not use_softmax:
            from sparsemax import Sparsemax
            self.attention_fn = Sparsemax(dim=1)
        else:
            self.attention_fn = None

        # Dedicated flashlight MLP: processes the 11 structure-tensor channels separately
        # from local_nn so that the flashlight signal is not diluted by F*K geometry channels.
        # Input LayerNorm intentionally omitted: eigenvalue magnitudes carry the signal
        # (flat neighbourhood → ~0, structured → larger); normalising the input erases it.
        self.flashlight_out_dim = int(flashlight_out_dim)
        if self.flashlight_out_dim > 0:
            h1 = max(64, self.flashlight_out_dim * 4)
            h2 = max(32, self.flashlight_out_dim * 2)
            self.flashlight_mlp = nn.Sequential(
                nn.Linear(11, h1),
                nn.LeakyReLU(inplace=True),
                nn.LayerNorm(h1),
                nn.Linear(h1, h2),
                nn.LeakyReLU(inplace=True),
                nn.LayerNorm(h2),
                nn.Linear(h2, self.flashlight_out_dim),
                nn.LeakyReLU(inplace=True),
                nn.LayerNorm(self.flashlight_out_dim),
            )
        else:
            self.flashlight_mlp = None

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(self, x: Union[OptTensor, PairOptTensor],
                pos: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """
        Args:
            x:   Node features [N, F] or pair for bipartite.
            pos: Node positions [N, >=4]; pos[:, :3] = xyz, pos[:, 3] = reflectance.
            edge_index: Graph connectivity.

        Returns:
            [N, K*F + 11] or [N, local_nn_out + flash_dim] when flashlight_out_dim > 0.
        """
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

        num_nodes = pos[1].size(0) if isinstance(pos, (list, tuple)) else pos.size(0)
        return self.propagate(edge_index, x=x, pos=pos, num_nodes=num_nodes)

    def message(self, x_j: OptTensor, pos_i: Tensor, pos_j: Tensor, index: Tensor,
                num_nodes: Optional[int] = None):
        rel_pos = (pos_j[:, :3] - pos_i[:, :3]).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        rel_dir = F.normalize(rel_pos, dim=1, eps=1e-6)
        dists   = torch.norm(rel_pos, dim=1, keepdim=True)
        max_d, _ = scatter_max(dists, index, dim=0,
                               dim_size=num_nodes if num_nodes is not None else None)

        kernel_dirs = (self.kernel_dirs if self.kernel_dirs is not None
                       else F.normalize(self.kernel_points, dim=1, eps=1e-6)).unsqueeze(0)
        attn = torch.sum(rel_dir.unsqueeze(1) * kernel_dirs, dim=-1)
        if self.use_softmax:
            weights = F.softmax((attn.float() / self.softmax_temperature), dim=1).to(attn.dtype)
        else:
            weights = self.attention_fn(attn.float()).to(attn.dtype)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        is_self_loop = dists.squeeze(-1) < 1e-8
        weights = weights.clone()
        if is_self_loop.any():
            weights[is_self_loop] = 0.0
            weights[is_self_loop, 0] = 1.0

        non_self = ~is_self_loop
        if non_self.any():
            w_non = weights[non_self]
            w_non[:, 0] = 0.0
            w_non = w_non / w_non.sum(dim=1, keepdim=True).clamp(min=1e-8)
            weights[non_self] = w_non

        norm_radius = max_d[index].clamp(min=1e-8)
        dist_norm   = (dists / norm_radius).clamp(0.0, 4.0)

        feat_list = [rel_dir, dist_norm]
        if x_j is not None:
            feat_list.insert(0, x_j)
        feat = torch.cat(feat_list, dim=-1)

        refl_j = (pos_j[:, 3].to(feat.dtype) if pos_j.size(1) >= 4
                  else torch.zeros(feat.size(0), device=feat.device, dtype=feat.dtype))
        refl_i = (pos_i[:, 3].to(feat.dtype) if pos_i.size(1) >= 4
                  else torch.zeros(feat.size(0), device=feat.device, dtype=feat.dtype))

        if self.training:
            ent     = (-(weights * (weights + 1e-8).log()).sum(dim=1)).mean()
            max_ent = math.log(max(weights.size(1), 2))
            self.kernel_entropy = (ent / max_ent).clamp(0.0, 1.0)

        return (feat, weights, refl_j, rel_dir, refl_i, is_self_loop)

    def aggregate(self, inputs, index: Tensor,
                  ptr: Optional[Tensor] = None, dim_size: Optional[int] = None) -> Tensor:
        feat, weights, refl_j, rel_dir, refl_i, is_self_loop = inputs

        kernel_mass    = scatter_add(weights, index, dim=0, dim_size=dim_size)  # [N, K]
        neighbor_count = kernel_mass.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [N, 1]
        n_pts = dim_size

        # Center-relative contrast; NaN-guarded for bad sensor returns.
        c = (refl_j - refl_i).abs().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # [E]

        # Cache outer products once for all three structure tensors.
        # Self-loop has rel_dir = 0 (after normalize with eps), so outer_flat_self = 0.
        # Self-loops therefore never contribute to any M_* numerator regardless of weight.
        outer_flat = (rel_dir.unsqueeze(-1) * rel_dir.unsqueeze(-2)).view(-1, 9)  # [E, 9]
        nc = neighbor_count.unsqueeze(-1)  # [N, 1, 1] for broadcasting over [N, 3, 3]

        # M_geom: uniform — pure geometric neighbourhood shape.
        # Note: trace(M_geom) = (non-self edges) / (total count incl. self) = k/(k+1).
        # This constant bias is absorbed by the flashlight_mlp.
        M_geom = (scatter_add(outer_flat, index, dim=0, dim_size=dim_size)
                  .view(n_pts, 3, 3) / nc)

        # M_refl: contrast-weighted — captures directions toward DISSIMILAR neighbours.
        # Self contributes c_self = 0 to contrast_sum; it cancels in mean_c.
        contrast_sum = scatter_add(c, index, dim=0, dim_size=dim_size)  # [N]
        M_refl = (scatter_add(c.unsqueeze(1) * outer_flat, index, dim=0, dim_size=dim_size)
                  .view(n_pts, 3, 3) / contrast_sum.clamp(min=1e-6).view(n_pts, 1, 1))

        # M_cobright: complement-weighted — captures directions toward SIMILAR neighbours.
        # c_max is the local contrast range, the natural scale for the decomposition.
        # Weight = c_max - c_j: maximum for co-bright pairs (c_j ≈ 0), zero for the most
        # contrasting neighbour. No calibration constants; scale is set by the neighbourhood.
        #
        # Self-loop exclusion: the self-loop has c_self = 0, so complement_self = c_max.
        # Since outer_flat_self = 0, it would not enter the M_cobright numerator, but it
        # WOULD inflate compl_sum by c_max, biasing eigenvalues down by ~1/(n_cobright+1).
        # We explicitly zero the self-loop complement before the scatter to avoid this.
        c_max      = scatter_max(c, index, dim=0, dim_size=dim_size)[0]  # [N]
        complement = (c_max[index] - c).clamp(min=0.0)                   # [E]
        complement = complement.masked_fill(is_self_loop, 0.0)           # exclude self-loop
        compl_sum  = scatter_add(complement, index, dim=0, dim_size=dim_size)  # [N]
        M_cobright = (scatter_add(complement.unsqueeze(1) * outer_flat, index, dim=0, dim_size=dim_size)
                      .view(n_pts, 3, 3) / compl_sum.clamp(min=1e-6).view(n_pts, 1, 1))

        del outer_flat

        def _sym_eigvals(M):
            S = (M + M.transpose(-1, -2)) * 0.5
            return _eigvalsh_3x3_analytical(S)

        eigvals_geom     = _sym_eigvals(M_geom);     del M_geom
        eigvals_refl     = _sym_eigvals(M_refl);     del M_refl
        eigvals_cobright = _sym_eigvals(M_cobright); del M_cobright

        # Non-self neighbour count for unbiased mean_c (excluding self-loop's c=0).
        # neighbor_count is [N, 1]; squeeze to [N] before dividing into contrast_sum [N].
        non_self_count = (neighbor_count.squeeze(1) - 1.0).clamp(min=1.0)  # [N]
        mean_c    = (contrast_sum / non_self_count).unsqueeze(1).clamp(max=2.0)  # [N, 1]
        c_max_out = c_max.unsqueeze(1).clamp(max=2.0)                            # [N, 1]

        # 11-channel flashlight: three eigenvalue spectra + mean contrast + local dynamic range.
        # mean_c and c_max together let the MLP distinguish a specular outlier
        # (c_max >> mean_c) from a genuine wood/leaf interface (both similar and large).
        flashlight_raw = torch.cat(
            [eigvals_geom, eigvals_refl, eigvals_cobright, mean_c, c_max_out], dim=-1
        )
        flashlight_raw = torch.nan_to_num(flashlight_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Geometry aggregation
        if self.memory_efficient:
            agg_chunks = []
            for k_idx in range(weights.size(1)):
                agg_k = scatter_add(
                    feat * weights[:, k_idx:k_idx + 1].to(feat.dtype),
                    index, dim=0, dim_size=dim_size)
                agg_chunks.append(agg_k)
            del feat, weights
            agg_feat = torch.stack(agg_chunks, dim=1) / neighbor_count.unsqueeze(1)
            del agg_chunks
            agg_feat = agg_feat.contiguous().view(dim_size, -1)
        else:
            weighted_feat = feat.unsqueeze(-1) * weights.unsqueeze(1).to(feat.dtype)
            agg_feat = scatter_add(weighted_feat, index, dim=0, dim_size=dim_size)
            del weighted_feat, feat, weights
            agg_feat = (agg_feat / neighbor_count.unsqueeze(1)
                        ).transpose(1, 2).contiguous().view(dim_size, -1)

        if self.flashlight_mlp is not None:
            agg_feat = torch.nan_to_num(agg_feat, nan=0.0, posinf=0.0, neginf=0.0)
            if self.local_nn is not None:
                agg_feat = self.local_nn(agg_feat)
            combined = torch.cat([agg_feat, self.flashlight_mlp(flashlight_raw)], dim=-1)
        else:
            combined = torch.cat([agg_feat, flashlight_raw], dim=-1)
            combined = torch.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
            if self.local_nn is not None:
                combined = self.local_nn(combined)

        combined = torch.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

        if self.training:
            with torch.no_grad():
                cobright_lift = F.relu(eigvals_cobright[:, 2] - eigvals_geom[:, 2])
                refl_lift     = F.relu(eigvals_refl[:, 2]     - eigvals_geom[:, 2])
                self.diagnostics = {
                    'contrast_gate_mean':     mean_c.mean().detach(),
                    'cobright_lift_mean':     cobright_lift.mean().detach(),
                    'cobright_lift_max_mean': eigvals_cobright[:, 2].mean().detach(),
                    'refl_lift_mean':         refl_lift.mean().detach(),
                    'c_max_mean':             c_max.mean().detach(),
                    'has_nan':                torch.isnan(flashlight_raw).any().detach(),
                }
            # Distillation proxy: local reflectance activity per point.
            self.last_refl_gate_per_point = mean_c.squeeze(1).clamp(0.0, 1.0).detach()

        return combined

    def update(self, aggr_out: Tensor) -> Tensor:
        if self.global_nn is not None:
            aggr_out = self.global_nn(aggr_out)
        return aggr_out
