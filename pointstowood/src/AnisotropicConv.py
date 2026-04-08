"""
AnisotropicConv: Wood/leaf point conv — geometry-first, reflectance as flashlight.

Geometry: Directional kernel routing (Fibonacci sphere) + density-normalized aggregation.
Flashlight: D = M_refl - M_geom, where M_refl weights direction outer-products by local
  reflectance contrast (absolute deviation from per-neighbourhood median). Sensor-agnostic:
  a uniformly bright or dark neighbourhood gives r_local≈0 so M_refl≈0 and the gate suppresses.
  Both bright and dark deviations count so dark twigs in bright canopy are detectable.
Reflectance reliability gate: per-eigenvalue learned gate — scales eigvals(D) independently.
Output: agg_feat (F*K) + eigvals_gated (3).
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


def fibonacci_sphere(n: int, radius: float = 1.0) -> Tensor:
    """Kernel directions on unit sphere for routing."""
    if n == 1:
        return torch.zeros(1, 3, dtype=torch.float32)
    origin = torch.zeros(1, 3, dtype=torch.float32)
    indices = torch.arange(n - 1, dtype=torch.float32)
    phi = (indices + 0.5) * (torch.pi * (3 - torch.sqrt(torch.tensor(5.0))))
    y = 1 - (indices / float(n - 2)) * 2
    r = torch.sqrt(1 - y**2)
    x = r * torch.cos(phi)
    z = r * torch.sin(phi)
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
                 refl_gate_bias_init: float = 0.0,
                 refl_gate_cap: float = 1.0,
                 **kwargs):
        # Pop any legacy parameters that model.py might pass
        kwargs.pop('use_dualnorm_lite', None)
        kwargs.pop('dualnorm_lite', None)
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
        self.refl_gate_bias_init = refl_gate_bias_init
        self.refl_gate_cap = float(min(1.0, max(0.0, refl_gate_cap)))

        # Trust-aware per-edge reflectance gain: modulate edge contributions before aggregation
        self.use_refl_edge_gain = True
        self.refl_edge_gain_logit = nn.Parameter(torch.tensor(-2.2, dtype=torch.float32))
        self.refl_edge_gain_cap = 0.35

        if not use_softmax:
            from sparsemax import Sparsemax
            self.attention_fn = Sparsemax(dim=1)
        else:
            self.attention_fn = None

        # Per-point reflectance reliability gate.
        # Input: geom kernel mass (K) + geom eigvals (3) + refl eigvals (3) + D eigvals (3) + contrast strength (1)
        # Output: 3 eigval gates + 1 trust scalar
        # Gate sees all three spectra to learn relationships:
        # - Eigval gates (3): independently scale each eigenvalue (directional control)
        # - Trust scalar (1): point-level reliability for edge-level reflectance gain
        gate_hidden = max(8, num_kernel_points)
        self.refl_reliability_gate = nn.Sequential(
            nn.Linear(num_kernel_points + 10, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 4),
            nn.Sigmoid(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)
        reset(self.refl_reliability_gate)
        # Gate bias controls initial reflectance reliance.
        # 0.0 = neutral, negative = cautious, positive = rely-more.
        if isinstance(self.refl_reliability_gate[2], nn.Linear):
            nn.init.constant_(self.refl_reliability_gate[2].bias, self.refl_gate_bias_init)

    def forward(self, x: Union[OptTensor, PairOptTensor],
                pos: Union[Tensor, PairTensor], edge_index: Adj,
                sf: Optional[Tensor] = None, voxel_size: Optional[Union[float, Tensor]] = None,
                batch_idx: Optional[Tensor] = None, **unused_kwargs) -> Tensor:
        """Anisotropic convolution: geometry-first routing with reflectance gating.

        Args:
            x: Node features, shape [N, F] or (N_src, F_src), (N_tgt, F_tgt) for bipartite.
            pos: Node positions. Expects pos[:, :3] = xyz coordinates, pos[:, 3] = reflectance.
                 Shape [N, >=4] or tuple of (pos_src, pos_tgt).
            edge_index: Graph connectivity.
            sf, voxel_size, batch_idx: Legacy parameters (unused, accepted for API compatibility).

        Returns:
            Output features [N, K*F + 3] where K*F are aggregated features across kernel
            directions and 3 are gated eigenvalues of the reflectance difference tensor.
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
        rel_pos = pos_j[:, :3] - pos_i[:, :3]
        rel_dir = F.normalize(rel_pos, dim=1, eps=1e-6)
        dists = torch.norm(rel_pos, dim=1, keepdim=True)
        max_d, _ = scatter_max(dists, index, dim=0, dim_size=num_nodes if num_nodes is not None else None)

        # Geometry-only kernel routing
        kernel_dirs = (self.kernel_dirs if self.kernel_dirs is not None else F.normalize(self.kernel_points, dim=1, eps=1e-6)).unsqueeze(0)
        attn = torch.sum(rel_dir.unsqueeze(1) * kernel_dirs, dim=-1)
        if self.use_softmax:
            weights = F.softmax((attn.float() / self.softmax_temperature), dim=1).to(attn.dtype)
        else:
            weights = self.attention_fn(attn.float()).to(attn.dtype)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        self_mask = dists.squeeze(-1) < 1e-8
        weights = weights.clone()
        if self_mask.any():
            weights[self_mask] = 0.0
            weights[self_mask, 0] = 1.0

        # Kernel-0 is reserved for self only.
        # For non-self edges, force kernel-0 mass to zero and renormalize.
        non_self = ~self_mask
        if non_self.any():
            w_non = weights[non_self]
            w_non[:, 0] = 0.0
            w_non = w_non / w_non.sum(dim=1, keepdim=True).clamp(min=1e-8)
            weights[non_self] = w_non

        # Normalize radial feature by max distance in neighborhood (density-invariant)
        norm_radius = max_d[index].clamp(min=1e-8)
        # Clamp to 4.0 to handle sparse neighborhoods and prevent extreme values
        dist_norm = (dists / norm_radius).clamp(0.0, 4.0)

        feat_list = [rel_dir, dist_norm]
        if x_j is not None:
            feat_list.insert(0, x_j)
        feat = torch.cat(feat_list, dim=-1)
        weighted_feat = feat.unsqueeze(-1) * weights.unsqueeze(1)

        # Raw neighbor reflectance for within-direction consistency in aggregate
        if pos_j.size(1) >= 4:
            refl_j = pos_j[:, 3].to(feat.dtype)
        else:
            refl_j = torch.zeros(feat.size(0), device=feat.device, dtype=feat.dtype)

        if self.training:
            # Normalized to [0,1] — max entropy = log(K). Do not compare to old eigenvalue-based support.
            ent = (-(weights * (weights + 1e-8).log()).sum(dim=1)).mean()
            max_ent = math.log(max(weights.size(1), 2))
            self.kernel_entropy = (ent / max_ent).clamp(0.0, 1.0)

        return (weighted_feat, weights, refl_j, rel_dir)

    def _compute_neighborhood_median(self, refl_j: Tensor, index: Tensor,
                                      kernel_mass_geom: Tensor, dim_size: int) -> Tensor:
        """Compute per-neighborhood median reflectance (robust to specular spikes).

        Uses two-level stable sort to avoid float32 precision loss from encoding.

        Args:
            refl_j: Reflectance values at edges [E]
            index: Node indices for each edge [E]
            kernel_mass_geom: Aggregate kernel weights per node [N, K]
            dim_size: Number of nodes

        Returns:
            Median reflectance per node [N]
        """
        # Sort edges by reflectance first, then by node index to group by node
        idx_by_refl = torch.argsort(refl_j, stable=True)
        idx_by_index = torch.argsort(index[idx_by_refl], stable=True)
        sorted_idx = idx_by_refl[idx_by_index]
        sorted_refl = refl_j[sorted_idx]

        # Compute edge counts per node and find median position
        counts_int = kernel_mass_geom.sum(dim=1).round().long().clamp(min=1)
        cum_counts = torch.zeros(dim_size + 1, dtype=torch.long, device=refl_j.device)
        cum_counts[1:] = counts_int.cumsum(0)
        median_pos = cum_counts[:-1] + counts_int // 2
        median_pos = median_pos.clamp(0, sorted_refl.size(0) - 1)

        return sorted_refl[median_pos]

    def aggregate(self, inputs, index: Tensor, ptr: Optional[Tensor] = None, dim_size: Optional[int] = None) -> Tensor:
        weighted_feat, weights, refl_j, rel_dir = inputs

        # Compute geometric kernel mass for the gate before scatter_add
        kernel_mass_geom = scatter_add(weights, index, dim=0, dim_size=dim_size)  # [N, K]
        neighbor_count = kernel_mass_geom.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # --- Tensor flashlight: reflectance illuminating geometry ---
        # Local contrast: absolute deviation from per-neighbourhood median.
        # Median is robust to specular spikes; a uniformly bright neighbourhood
        # yields r_local ≈ 0 for all points so M_refl ≈ 0 and the gate suppresses.
        # Both bright AND dark deviations count (abs) so a dark twig in a bright
        # leaf canopy is just as detectable as a bright twig in a dark one.

        n_pts = dim_size
        refl_median = self._compute_neighborhood_median(refl_j, index, kernel_mass_geom, n_pts)

        r_local = (refl_j - refl_median[index]).abs()  # [E] absolute contrast from median

        # Structure tensors: outer products of unit direction vectors
        # M_geom = (1/k) Σ d̂⊗d̂           — geometric neighbourhood shape
        # M_refl = (1/k) Σ r_local·d̂⊗d̂   — brightness-weighted shape
        # D = M_refl - M_geom: where does brightness deviate from pure geometry?
        outer_flat = (rel_dir.unsqueeze(-1) * rel_dir.unsqueeze(-2)).view(-1, 9)  # [E, 9]
        nc = neighbor_count.unsqueeze(-1)  # [N, 1, 1] after second unsqueeze below

        M_geom = scatter_add(outer_flat, index, dim=0, dim_size=dim_size).view(n_pts, 3, 3) / nc  # nc [N,1,1] broadcasts to [N,3,3]
        M_refl = scatter_add(r_local.unsqueeze(1) * outer_flat, index, dim=0, dim_size=dim_size).view(n_pts, 3, 3) / nc

        # Eigenvalues of geometry structure tensor (tells gate if geometry is coherent or scattered)
        M_geom_sym = (M_geom + M_geom.transpose(-1, -2)) * 0.5
        eigvals_geom = torch.linalg.eigvalsh(M_geom_sym)  # [N, 3] ascending
        eigvals_geom = torch.nan_to_num(eigvals_geom, nan=0.0, posinf=0.0, neginf=0.0)

        # Eigenvalues of reflectance structure tensor (tells gate if reflectance shows structure)
        M_refl_sym = (M_refl + M_refl.transpose(-1, -2)) * 0.5
        eigvals_refl = torch.linalg.eigvalsh(M_refl_sym)  # [N, 3] ascending
        eigvals_refl = torch.nan_to_num(eigvals_refl, nan=0.0, posinf=0.0, neginf=0.0)

        D_sym = (M_refl - M_geom)
        D_sym = (D_sym + D_sym.transpose(-1, -2)) * 0.5  # enforce symmetry for eigvalsh
        eigvals = torch.linalg.eigvalsh(D_sym)  # [N, 3] ascending: e0 ≤ e1 ≤ e2
        eigvals = torch.nan_to_num(eigvals, nan=0.0, posinf=0.0, neginf=0.0)

        # Contrast strength: mean absolute deviation from median — tells gate how
        # much local reflectance variation exists in this neighbourhood.
        contrast_strength = scatter_add(r_local, index, dim=0, dim_size=dim_size).unsqueeze(1) / neighbor_count  # [N, 1]

        # Reflectance reliability gate: learns when to trust reflectance vs geometry.
        # Input: kernel routing (K), geometry spectrum (3), reflectance spectrum (3),
        #        difference spectrum (3), and local contrast strength (1).
        kernel_mass_geom_norm = kernel_mass_geom / kernel_mass_geom.sum(dim=1, keepdim=True).clamp(min=1e-8)
        gate_input = torch.cat([kernel_mass_geom_norm, eigvals_geom, eigvals_refl, eigvals, contrast_strength], dim=1)  # [N, K+10]
        gate_output = self.refl_reliability_gate(gate_input)  # [N, 4]
        refl_gate = gate_output[:, :3]  # [N, 3] per-eigenvalue gates (directional control)
        trust = gate_output[:, 3]  # [N] scalar trust (point-level reliability)

        refl_gate = torch.nan_to_num(refl_gate, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if self.refl_gate_cap < 1.0:
            refl_gate = refl_gate * self.refl_gate_cap
        eigvals_gated = eigvals * refl_gate  # [N, 3]

        # Trust-aware per-edge reflectance gain: modulate edge contributions before aggregation.
        # Geometry still determines routing (kernel bins); reflectance only changes edge strength.
        # This allows reflectance to sharpen faint structures (twigs in noise) without inventing
        # structure from pure brightness.
        if self.use_refl_edge_gain:
            edge_trust = trust[index]  # [E] broadcast per-point trust to edges

            contrast_mean_edge = contrast_strength[index].squeeze(1)  # [E]
            # Normalize edge reflectance contrast by neighborhood scale
            refl_score = r_local / (contrast_mean_edge + 1e-6)
            # Clamp to 3.0 to prevent extreme amplification in high-contrast regions
            refl_score = torch.clamp(refl_score, 0.0, 3.0)

            # Beta init=-2.2 gives sigmoid≈0.09, cap=0.35 allows max gain ≈1.35x
            beta = torch.sigmoid(self.refl_edge_gain_logit).to(weighted_feat.dtype) * self.refl_edge_gain_cap
            refl_gain = 1.0 + beta * edge_trust * torch.tanh(refl_score)  # [E]

            weighted_feat = weighted_feat * refl_gain.view(-1, 1, 1)

        # Now aggregate features with optional reflectance modulation
        agg_feat = scatter_add(weighted_feat, index, dim=0, dim_size=dim_size)  # [N, F, K]
        agg_feat = agg_feat / neighbor_count.unsqueeze(1)

        agg_feat = agg_feat.transpose(1, 2).contiguous().view(dim_size, -1)
        combined = torch.cat([agg_feat, eigvals_gated], dim=-1)
        combined = torch.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

        if self.training:
            with torch.no_grad():
                has_nan = torch.isnan(eigvals).any()
                self.diagnostics = {
                    'D_mean': eigvals.mean().item(),
                    'D_max_eig_mean': eigvals[:, 2].mean().item(),
                    'D_max_eig_std': eigvals[:, 2].std().item(),
                    'contrast_strength_mean': contrast_strength.mean().item(),
                    'refl_gate_mean': refl_gate.mean().item(),
                    'refl_gate_min': refl_gate.min().item(),
                    'refl_gate_max': refl_gate.max().item(),
                    'eigvals_gated_mean': eigvals_gated.mean().item(),
                    'trust_mean': trust.mean().item(),
                    'has_nan': has_nan,
                }
                if has_nan:
                    self.diagnostics['warning'] = 'NaN detected in eigvals'

        # Trainer logging — maximum eigenvalue of D (dominant orientation of reflectance-geometry difference)
        self.last_D_max_eig = float(eigvals[:, 2].mean().detach().cpu().item())
        self.last_D_max_eig_per_point = eigvals[:, 2].detach()
        self.last_refl_gate = float(refl_gate.mean().detach().cpu().item())
        self.last_refl_gate_per_point = refl_gate.mean(dim=1).detach()
        self.last_trust = float(trust.mean().detach().cpu().item())
        self.last_contrast_strength_per_point = contrast_strength.squeeze(1).detach()

        if self.local_nn is not None:
            combined = self.local_nn(combined)
        return combined

    def update(self, aggr_out: Tensor) -> Tensor:
        if self.global_nn is not None:
            aggr_out = self.global_nn(aggr_out)
        return aggr_out
