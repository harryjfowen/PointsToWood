import torch
import torch.nn.functional as F
from torch_geometric.nn import voxel_grid, knn, knn_graph
from torch.nn import Sequential as Seq, Linear as Lin
from src.AnisotropicConv import AnisotropicConv
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter_add, scatter_max
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    for m in model.modules():
        if getattr(m, '_zero_init_after_global_init', False):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


def _effective_num_groups(channel: int, target: int) -> int:
    """Largest divisor of channel that is <= target and >= 1 (for GroupNorm)."""
    if target <= 0 or channel <= 0:
        return 1
    g = min(target, channel)
    while g > 0 and channel % g != 0:
        g -= 1
    return max(1, g)


def normalize_stage_kernel_points(num_kernel_points):
    """Accept a scalar or 3-stage kernel spec and return (sa1, sa2, sa3)."""
    if isinstance(num_kernel_points, torch.Tensor):
        values = [int(v) for v in num_kernel_points.detach().cpu().view(-1).tolist()]
    elif isinstance(num_kernel_points, (list, tuple)):
        values = [int(v) for v in num_kernel_points]
    else:
        values = [int(num_kernel_points)]

    if len(values) == 1:
        values = values * 3
    elif len(values) != 3:
        raise ValueError(
            f"num_kernel_points must be a single int or three ints for SA1/SA2/SA3, got {values}"
        )

    if any(v < 1 for v in values):
        raise ValueError(f"Kernel counts must all be >= 1, got {values}")

    return tuple(values)


class GroupNorm1d(nn.Module):
    """GroupNorm for (N, C) input: point-wise norm over channel groups (batch-composition invariant)."""
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(_effective_num_groups(num_channels, num_groups), num_channels, eps=eps)

    def forward(self, x):
        # x: (N, C) -> (N, C, 1) for GroupNorm
        return self.gn(x.unsqueeze(-1)).squeeze(-1)


class GroupNorm1dChannels(nn.Module):
    """GroupNorm for (B, C, L) input (Conv1d style): per-position norm over channel groups."""
    def __init__(self, num_channels: int, num_groups: int = 4, eps: float = 1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(_effective_num_groups(num_channels, num_groups), num_channels, eps=eps)

    def forward(self, x):
        return self.gn(x)


class BatchNorm1dForPoints(nn.Module):
    """BatchNorm1d for (N, C) input: treats N as batch, normalizes over points per channel."""
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_channels, eps=eps)

    def forward(self, x):
        # x: (N, C) -> (N, C, 1) for BatchNorm1d (batch=N, channels=C, length=1)
        return self.bn(x.unsqueeze(-1)).squeeze(-1)


class LayerNormChannels(nn.Module):
    """LayerNorm on channel dim for (B, C, L) input. Use after global fusion (e.g. seg_head)."""
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)


class FullWidthSegHead(nn.Module):
    """Full-width pointwise MLP head over the final FP features.

    The FP decoder already contains the fused local/detail/context representation.
    For the high-quality binary wood/leaf model, keep that C3 width through the
    decision block instead of immediately squeezing rare cues through a tiny
    bottleneck.
    """
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.10):
        super().__init__()
        hidden = in_channels
        bottleneck = max(in_channels // 2, num_classes)

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1),
            LayerNormChannels(hidden),
            nn.LeakyReLU(inplace=False),
        )
        self.refine = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1),
            LayerNormChannels(hidden),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Conv1d(hidden, hidden, 1),
            LayerNormChannels(hidden),
        )
        self.output_proj = nn.Sequential(
            nn.LeakyReLU(inplace=False),
            nn.Conv1d(hidden, bottleneck, 1),
            LayerNormChannels(bottleneck),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Conv1d(bottleneck, num_classes, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = F.leaky_relu(x + self.refine(x), inplace=False)
        x = self.output_proj(x)
        return self.classifier(x)


class CompressedSegHead(nn.Module):
    """Compact pointwise head: compress hard, refine at compact width, classify.

    This remains useful for student/fast ablation runs, but it is no longer the
    default for the complex high-quality model.
    """
    def __init__(self, in_channels: int, num_classes: int, compress: int = 64):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv1d(in_channels, compress, 1),
            LayerNormChannels(compress),
            nn.LeakyReLU(inplace=False),
        )
        # Light residual at compressed dim — non-linear boundary fitting without full-width cost.
        self.refine = nn.Sequential(
            nn.Conv1d(compress, compress, 1),
            LayerNormChannels(compress),
            nn.LeakyReLU(inplace=False),
            nn.Conv1d(compress, compress, 1),
            LayerNormChannels(compress),
        )
        self.classifier = nn.Conv1d(compress, num_classes, 1)

    def forward(self, x):
        x = self.compress(x)
        x = F.leaky_relu(x + self.refine(x), inplace=False)
        return self.classifier(x)


class ContrastiveHead(nn.Module):
    """Projection head for supervised contrastive learning.

    Two-layer MLP (in_channels → in_channels → proj_dim) with L2-normalisation.
    Sits parallel to seg_head on the FP1 output. Discarded at inference.
    """
    def __init__(self, in_channels: int, proj_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.LeakyReLU(inplace=False),
            nn.Linear(in_channels, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1)


# Backwards-compatible name for existing code/docs that refer to the full head.
ResidualSegHead = FullWidthSegHead


class DropPathPack(nn.Module):
    """Stochastic depth (DropPath) for point cloud batches — KPConvX-style.

    Drops the residual branch for entire samples (not individual points) to maintain
    spatial consistency. Use scale_by_keep=True so E[output]=input at inference.

    KPConvX uses a depth-linear schedule: deeper blocks get higher drop_prob.
    """
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x, lengths, return_mask: bool = False):
        if self.drop_prob <= 0 or not self.training:
            if return_mask:
                mask = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
                return x, mask
            return x

        keep_prob = 1.0 - self.drop_prob

        # Efficient GPU-native implementation (no Python loops or CPU transfers)
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths, device=x.device, dtype=torch.long)

        # One Bernoulli per sample, then expand to all points
        bernoulli = torch.empty(lengths.shape[0], device=x.device, dtype=x.dtype).bernoulli_(keep_prob)
        if self.scale_by_keep and keep_prob > 0:
            bernoulli.div_(keep_prob)

        # Expand sample-level mask to point-level (stays on GPU)
        mask = bernoulli.repeat_interleave(lengths).view(-1, 1)

        if return_mask:
            return x * mask, mask
        return x * mask


class InvertedBottleneck(nn.Module):
    """Inverted bottleneck block with pre-norm and GroupNorm (batch-size invariant)."""
    def __init__(self, in_channels: int, out_channels: int, expansion_factor: int = 4, drop_prob: float = 0.0, num_groups: int = 4, layer_scale_init: float = 0.1, spatial_mixing: bool = False, spatial_mix_init: float = 0.10):
        super().__init__()

        expanded = in_channels * expansion_factor

        self.norm1 = GroupNorm1dChannels(in_channels, num_groups=num_groups)
        self.expand_conv = nn.Conv1d(in_channels, expanded, 1, bias=False)
        self.norm2 = GroupNorm1dChannels(expanded, num_groups=num_groups)
        self.project_conv = nn.Conv1d(expanded, out_channels, 1, bias=False)

        self.use_skip = in_channels == out_channels
        if not self.use_skip:
            self.norm_shortcut = GroupNorm1dChannels(in_channels, num_groups=num_groups)
            self.shortcut_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.spatial_mixing = bool(spatial_mixing)
        if self.spatial_mixing:
            self.spatial_mix_norm = GroupNorm1d(num_groups=num_groups, num_channels=in_channels)
            self.spatial_mix_scale = nn.Parameter(torch.ones(in_channels) * float(spatial_mix_init))

        self.drop_path = DropPathPack(drop_prob=drop_prob)
        # LayerScale: stabilize optimization by softly gating residual branch at init.
        self.layer_scale = nn.Parameter(torch.ones(out_channels) * float(layer_scale_init)) if layer_scale_init > 0 else None
        self.activation = nn.LeakyReLU(inplace=True)

    def _spatial_mix(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return x
        src, dst = edge_index
        if src.numel() == 0:
            return x
        nbr_sum = scatter_add(x[src], dst, dim=0, dim_size=x.size(0))
        deg = scatter_add(torch.ones((src.size(0), 1), dtype=x.dtype, device=x.device), dst, dim=0, dim_size=x.size(0)).clamp(min=1.0)
        nbr_mean = nbr_sum / deg
        delta = self.spatial_mix_norm(nbr_mean - x)
        delta = delta * self.spatial_mix_scale.unsqueeze(0)
        return x + delta

    def forward(self, x, batch: torch.Tensor = None, edge_index: torch.Tensor = None):
        if self.spatial_mixing and edge_index is not None:
            # Low-cost local extractor (reuse precomputed graph, no extra kernel attention).
            x = self._spatial_mix(x, edge_index)
        x_conv = x.unsqueeze(0).transpose(1, 2)  # (1, C, N)

        if self.use_skip:
            residual = x
        else:
            residual = self.shortcut_conv(self.norm_shortcut(x_conv)).transpose(1, 2).squeeze(0)

        # Pre-norm: norm -> conv -> relu -> norm -> conv
        h = self.norm1(x_conv)
        h = self.expand_conv(h)
        h = F.leaky_relu(h, inplace=True)
        h = self.norm2(h)
        out = self.project_conv(h).transpose(1, 2).squeeze(0)

        # DropPath drops the transform branch (expand+project) for entire samples.
        # Apply regardless of use_skip: in the channel-changing case the shortcut_conv
        # still provides a valid residual path, so dropping the heavy branch is valid
        # stochastic depth and keeps all 18 SA blocks on the KPConvX-style schedule.
        if batch is not None:
            lengths = torch.bincount(batch)
            out = self.drop_path(out, lengths)

        if self.layer_scale is not None:
            out = out * self.layer_scale.unsqueeze(0)

        out = self.activation(out + residual)
        return out


class SAModule(torch.nn.Module):

    def __init__(self, resolution, k, NN, num_blocks=1, num_kernel_points=16, learnable_kernels=False,
                 drop_path_rate=0.0, block_offset=0, total_blocks=12, num_groups=16,
                 spatial_mix_lite: bool = False,
                 flash_dim: int = 32,
                 memory_efficient_conv: bool = False,
                 sparse_max: bool = True,
                 use_reflectance: bool = True,
                 refl_pool: str = 'max'):
        super(SAModule, self).__init__()
        self.resolution = resolution
        self.k = k
        self.spatial_mix_lite = bool(spatial_mix_lite)
        self.flash_dim = int(flash_dim)
        self.memory_efficient_conv = bool(memory_efficient_conv)
        self.sparse_max = bool(sparse_max)
        self.use_reflectance = bool(use_reflectance)
        self.refl_pool = refl_pool  # 'max' or 'mean'

        # NN[0] = F*K (agg_feat channels only). Split NN[-1] between geometry (local_nn) and
        # flashlight (dedicated MLP inside AnisotropicConv). Total conv output stays at NN[-1],
        # so the downstream residual blocks and SA cascade remain shape-compatible.
        if self.flash_dim > 0:
            c_geom = max(16, NN[-1] - self.flash_dim)
            actual_flash = NN[-1] - c_geom
            NN_local = [NN[0], NN[1], c_geom]
        else:
            actual_flash = 0
            # Legacy path: local_nn sees agg_feat + raw 11 flashlight channels concatenated.
            NN_local = [NN[0] + 11, NN[1], NN[-1]]

        self.conv = AnisotropicConv(
            local_nn=MLP(NN_local, num_groups=num_groups, conv_style=True),
            global_nn=None,
            add_self_loops=False,
            num_kernel_points=num_kernel_points,
            learnable_kernels=learnable_kernels,
            use_softmax=not self.sparse_max,
            flashlight_out_dim=actual_flash,
            memory_efficient=self.memory_efficient_conv,
        )
        # KPConvX-style global depth-linear schedule: block at global idx l gets (l+1)/total * rate
        drop_probs = [(block_offset + i + 1) / total_blocks * drop_path_rate for i in range(num_blocks)] if drop_path_rate > 0 else [0.0] * num_blocks
        self.residual_blocks = nn.ModuleList([
            InvertedBottleneck(NN[-1], NN[-1], drop_prob=p, num_groups=num_groups, spatial_mixing=self.spatial_mix_lite)
            for p in drop_probs
        ])

    def voxelsample(self, pos, batch, resolution, reflectance=None):
        """Voxel grid downsampling: scatter-mean XYZ, scatter-max reflectance per voxel.

        XYZ: position of the max-reflectance point in the voxel (when refl_pool='max'),
        otherwise scatter-mean centroid. Anchoring on the max-refl point places the coarse
        node at the twig surface, so k-NN edges reach co-bright chain neighbours rather than
        the surrounding leaf cloud. M_cobright then has chain candidates in its neighbourhood.
        Reflectance: scatter-max preserves the brightest (most direct bark) return in each
        voxel. Mean would dilute it to near zero for thin branches surrounded by needles.

        Returns:
            cluster: mapping from each input point to its voxel/coarse point index
                     Used for grid unpooling in decoder (no knn needed)
            pos_coarse: scatter-mean XYZ per voxel
            refl_coarse: scatter-max reflectance per voxel
            batch_coarse: batch vector for coarse nodes
        """
        res_scalar = resolution.item() if isinstance(resolution, torch.Tensor) else resolution
        voxel_indices = voxel_grid(pos, res_scalar, batch)
        cluster, perm = consecutive_cluster(voxel_indices)
        dim_size = int(perm.numel())
        batch_coarse = batch[perm]

        counts = torch.bincount(cluster, minlength=dim_size).to(pos.dtype).clamp(min=1.0)
        if reflectance is not None:
            if self.refl_pool == 'max':
                refl_coarse, argmax_idx = scatter_max(reflectance, cluster, dim=0, dim_size=dim_size)
                # Anchor coarse node at the max-reflectance point's XYZ, not the voxel centroid.
                # The centroid sits inside the leaf cloud; the max-refl point is the twig surface.
                # k-NN from the twig surface finds chain neighbours → M_cobright gets its signal.
                pos_coarse = pos[argmax_idx, :3]
            else:
                refl_coarse = scatter_add(reflectance, cluster, dim=0, dim_size=dim_size) / counts
                pos_coarse = scatter_add(pos[:, :3], cluster, dim=0, dim_size=dim_size) / counts.unsqueeze(1)
        else:
            refl_coarse = torch.zeros(dim_size, device=pos.device, dtype=pos.dtype)
            pos_coarse = scatter_add(pos[:, :3], cluster, dim=0, dim_size=dim_size) / counts.unsqueeze(1)
        return cluster, pos_coarse, refl_coarse, batch_coarse

    def forward(self, x, pos, batch, reflectance, sf, voxel_size=None):
        """
        Scale-invariance + density-agnostic strategy:
        1. Graph construction in METRES: voxel sampling at grid-tied resolution,
           k-NN edges (always k neighbours regardless of point spacing → sensor-agnostic).
        2. Normalize to unit scale (pos/sf) for conv → conv sees ~[-1,1] voxel regardless of size.
        3. Distances normalized by per-neighbourhood max_d in AnisotropicConv → density-invariant.
        4. Restore metres (pos*sf) so next layer sees normal coordinates again.

        Returns cluster indices for grid unpooling in decoder (KPConvX-style).
        """
        if not self.use_reflectance:
            reflectance = torch.zeros_like(reflectance)

        pos = torch.cat([pos[:, :3], reflectance.unsqueeze(-1)], dim=-1)

        # 1. Graph construction in METRES (k-NN, density-agnostic)
        # Always exactly k neighbours regardless of point spacing — sensor-agnostic.
        # Also get cluster mapping for grid unpooling in decoder.
        cluster, pos_coarse, refl_coarse, batch_coarse = self.voxelsample(
            pos[:, :3], batch, self.resolution, reflectance=reflectance
        )

        row, col = knn(
            pos[:, :3],
            pos_coarse,
            k=self.k,
            batch_x=batch,
            batch_y=batch_coarse,
        )
        edge_index = torch.stack([col, row], dim=0)

        # 2. Normalize xyz to unit scale for conv (pos/sf → ~[-1,1])
        # Non-in-place: critical for gradient checkpointing (forward reruns during backward)
        sf_scale = sf[batch].unsqueeze(-1)
        pos_conv = torch.cat([pos[:, :3] / sf_scale, pos[:, 3:]], dim=-1)
        sf_scale_coarse = sf[batch_coarse].unsqueeze(-1)
        pos_conv_coarse = torch.cat([pos_coarse / sf_scale_coarse, refl_coarse.unsqueeze(-1)], dim=-1)

        # 3. Conv: radial feature normalized by max_d (density-invariant), not fixed radius.
        x = self.conv(
            x,
            (pos_conv, pos_conv_coarse),
            edge_index,
        )

        residual_edge_index = None
        if self.spatial_mix_lite:
            # k-NN self-graph on coarse points for SpatialMixLite residual blocks.
            k_mix = min(8, self.k)
            residual_edge_index = knn_graph(
                pos_coarse,
                k=k_mix,
                batch=batch_coarse,
                loop=False,
            )

        for block in self.residual_blocks:
            x = block(x, batch_coarse, edge_index=residual_edge_index)

        pos_out = pos_coarse
        reflectance = refl_coarse
        batch = batch_coarse
        # Return cluster for grid unpooling: cluster[i] = index of coarse point for fine point i
        return x, pos_out, batch, reflectance, sf, cluster

class FPModule(torch.nn.Module):
    """Feature Propagation with grid unpooling (KPConvX-style).

    Instead of knn_interpolate which averages features from k neighbors
    (causing boundary blurring), we use the cluster indices from encoding
    to directly assign coarse features to fine points.

    Each fine point receives features from exactly the coarse point
    representing its voxel cell - no interpolation, no cross-boundary mixing.
    """
    def __init__(self, NN, num_groups=8, point_norm: str = "gn", guided_fusion: bool = True):
        super(FPModule, self).__init__()
        self.out_channels = NN[-1]
        self.skip_channels = max(0, NN[0] - self.out_channels)
        self.use_guided_fusion = bool(guided_fusion and self.skip_channels > 0)
        if self.use_guided_fusion:
            gate_hidden = max(32, min(self.out_channels, NN[0] // 2))
            self.guidance_gate = nn.Sequential(
                Lin(NN[0], gate_hidden),
                nn.LayerNorm(gate_hidden),
                nn.LeakyReLU(inplace=False),
                Lin(gate_hidden, self.skip_channels),
            )
            self.guidance_gate[-1]._zero_init_after_global_init = True

        self.NN = MLP(NN, num_groups=num_groups, point_norm=point_norm)

    def forward(self, x, batch, x_skip, batch_skip, cluster):
        """
        Args:
            x: coarse features [N_coarse, C]
            batch: coarse batch indices
            x_skip: fine skip features [N_fine, C_skip]
            batch_skip: fine batch indices
            cluster: mapping from fine points to coarse points [N_fine]
        """
        x_interp = x[cluster]

        if x_skip is not None:
            if self.use_guided_fusion:
                guidance_input = torch.cat([x_interp, x_skip], dim=1)
                detail_scale = 0.5 + torch.sigmoid(self.guidance_gate(guidance_input))
                x_skip = x_skip * detail_scale
                if self.training:
                    self.last_guidance_scale_mean = float(detail_scale.detach().mean().item())
            x_out = torch.cat([x_interp, x_skip], dim=1)
        else:
            x_out = x_interp

        return self.NN(x_out), batch_skip


def _point_norm_layer(channels: int, num_groups: int, point_norm: str):
    if point_norm == "gn":
        return GroupNorm1d(num_groups=num_groups, num_channels=channels)
    if point_norm == "ln":
        return nn.LayerNorm(channels)
    return BatchNorm1dForPoints(channels)


def MLP(channels, num_groups: int = 4, conv_style: bool = False, point_norm: str = "bn"):
    """MLP: LayerNorm for conv_style; configurable BN/GN/LN for point-wise decoder MLPs."""
    layers = []
    for i in range(1, len(channels)):
        layers.append(Lin(channels[i - 1], channels[i]))
        layers.append(torch.nn.LeakyReLU())
        layers.append(nn.LayerNorm(channels[i]) if conv_style else _point_norm_layer(channels[i], num_groups, point_norm))
    return Seq(*layers)


class NetFull(torch.nn.Module):
    """3-layer U-Net with AnisotropicConv for wood/leaf segmentation.

    Architecture: SA1 → SA2 → SA3 (encoder) → FP3 → FP2 → FP1 (decoder)
    - SA1–SA3: flashlight-enabled (reflectance structure tensor) at all stages.

    SA1/SA2 refine twig/leaf boundaries at fine scale (3.2cm, 5.2cm).
    SA3 captures branch-structure context (8.5cm).
    Mean XYZ centroid + max-refl representative at every voxelsampling stage.
    """
    def __init__(self, num_classes, C=256, num_kernel_points=(16, 16, 16), learnable_kernels=False, drop_path_rate=0.0, spatial_mix_lite: bool = False, use_gradient_checkpointing: bool = True, flash_dim: int = 32, memory_efficient_conv: bool = False, sparse_max: bool = True, compressed_head: bool = False, compressed_head_dim: int = 64, k_neighbors: int = 16):
        super(NetFull, self).__init__()

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.model_family = "full"
        self.c_base = int(C)
        self.k_neighbors = int(k_neighbors)
        self.stage_kernel_points = normalize_stage_kernel_points(num_kernel_points)
        k1, k2, k3 = self.stage_kernel_points
        self.num_kernel_points = int(k1)
        self.learnable_kernels = bool(learnable_kernels)
        self.spatial_mix_lite = bool(spatial_mix_lite)
        self.flash_dim = int(flash_dim)
        self.memory_efficient_conv = bool(memory_efficient_conv)
        self.sparse_max = bool(sparse_max)
        self.compressed_head = bool(compressed_head)
        self.compressed_head_dim = int(compressed_head_dim)
        # 6-12-2: SA1 learns geometric primitives, SA2 composes branch structures,
        # SA3 coarse context.
        self.sa1_blocks = 6
        self.sa2_blocks = 12
        self.sa3_blocks = 2

        # Voxelsample resolution RATIOS (relative to voxel_size, for scale invariance)
        # Base: 2m voxel → 2cm input → 3.2cm, 5.2cm, 8.5cm
        self.vx_ratio_1 = (0.02 * 1.618) / 2.0
        self.vx_ratio_2 = self.vx_ratio_1 * 1.618
        self.vx_ratio_3 = self.vx_ratio_2 * 1.618

        # Channel progression: C → 1.5C → 2C → 3C
        C1, C2, C3 = int(C * 1.5), C * 2, C * 3
        sa_in = lambda c, k: (c + 4) * k

        total_sa_blocks = self.sa1_blocks + self.sa2_blocks + self.sa3_blocks  # 20
        self.sa1_module = SAModule(0.032, self.k_neighbors, [sa_in(0, k1), C1 * 4, C1],
                                   num_blocks=self.sa1_blocks, num_kernel_points=k1, learnable_kernels=learnable_kernels,
                                   drop_path_rate=drop_path_rate, block_offset=0, total_blocks=total_sa_blocks,
                                   num_groups=16, spatial_mix_lite=spatial_mix_lite,
                                   flash_dim=self.flash_dim, memory_efficient_conv=self.memory_efficient_conv,
                                   sparse_max=self.sparse_max, refl_pool='max')
        self.sa2_module = SAModule(0.052, self.k_neighbors, [sa_in(C1, k2), C2 * 4, C2],
                                   num_blocks=self.sa2_blocks, num_kernel_points=k2, learnable_kernels=learnable_kernels,
                                   drop_path_rate=drop_path_rate, block_offset=self.sa1_blocks, total_blocks=total_sa_blocks,
                                   num_groups=16, spatial_mix_lite=spatial_mix_lite,
                                   flash_dim=self.flash_dim, memory_efficient_conv=self.memory_efficient_conv,
                                   sparse_max=self.sparse_max, refl_pool='max')
        k3_actual = k3
        self.sa3_module = SAModule(0.085, self.k_neighbors, [sa_in(C2, k3_actual), C3 * 4, C3],
                                   num_blocks=self.sa3_blocks, num_kernel_points=k3_actual, learnable_kernels=learnable_kernels,
                                   drop_path_rate=drop_path_rate, block_offset=self.sa1_blocks + self.sa2_blocks,
                                   total_blocks=total_sa_blocks, num_groups=16, spatial_mix_lite=spatial_mix_lite,
                                   flash_dim=self.flash_dim, memory_efficient_conv=self.memory_efficient_conv,
                                   sparse_max=self.sparse_max, refl_pool='max')

        # Grid unpooling (KPConvX-style) - no interpolation, sharp boundaries
        self.fp3_module = FPModule([C3 + C2, C3, C3], num_groups=16, point_norm="gn")
        self.fp2_module = FPModule([C3 + C1, C3, C3], num_groups=16, point_norm="gn")
        self.fp1_module = FPModule([C3, C3, C3], num_groups=16, point_norm="gn", guided_fusion=False)

        self.seg_head = (
            CompressedSegHead(C3, num_classes, compress=self.compressed_head_dim)
            if compressed_head
            else FullWidthSegHead(C3, num_classes)
        )

        self.proj_head = ContrastiveHead(C3, proj_dim=128)

        initialize_weights(self)

        with torch.no_grad():
            self.seg_head.classifier.bias.fill_(-0.5)

    def forward(self, data):
        voxel_size = getattr(data, 'voxel_size', None)
        if voxel_size is not None and isinstance(voxel_size, torch.Tensor) and voxel_size.numel() == 1:
            voxel_size = voxel_size.item()
        # Scale SA resolutions by voxel_size so cascade adapts to 1m / 2m / 4m etc.
        vx_scale = voxel_size if voxel_size is not None else 2.0
        self.sa1_module.resolution = self.vx_ratio_1 * vx_scale
        self.sa2_module.resolution = self.vx_ratio_2 * vx_scale
        self.sa3_module.resolution = self.vx_ratio_3 * vx_scale

        sa0_out = (None, data.pos, data.batch, data.reflectance, data.sf)

        # Encoder: each SA module returns (x, pos, batch, refl, sf, cluster)
        if self.use_gradient_checkpointing and self.training:
            sa1_out = checkpoint(self.sa1_module, *sa0_out, voxel_size=voxel_size, use_reentrant=False)
        else:
            sa1_out = self.sa1_module(*sa0_out, voxel_size=voxel_size)
        x1, pos1, batch1, refl1, sf, cluster1 = sa1_out

        if self.use_gradient_checkpointing and self.training:
            sa2_out = checkpoint(self.sa2_module, x1, pos1, batch1, refl1, sf, voxel_size=voxel_size, use_reentrant=False)
        else:
            sa2_out = self.sa2_module(x1, pos1, batch1, refl1, sf, voxel_size=voxel_size)
        x2, pos2, batch2, refl2, sf, cluster2 = sa2_out

        if self.use_gradient_checkpointing and self.training:
            sa3_out = checkpoint(self.sa3_module, x2, pos2, batch2, refl2, sf, voxel_size=voxel_size, use_reentrant=False)
        else:
            sa3_out = self.sa3_module(x2, pos2, batch2, refl2, sf, voxel_size=voxel_size)
        x3, pos3, batch3, refl3, sf, cluster3 = sa3_out

        # Store encoder features for CBL / feature KD. Eval-time capture is opt-in
        # for teacher distillation so normal inference does not pay the memory cost.
        if self.training or getattr(self, 'capture_encoder_features', False):
            self.encoder_stages = [
                {'features': x1, 'pos': pos1, 'batch': batch1, 'cluster': cluster1},
                {'features': x2, 'pos': pos2, 'batch': batch2, 'cluster': cluster2},
                {'features': x3, 'pos': pos3, 'batch': batch3, 'cluster': cluster3},
            ]

        # Decoder: grid unpooling using cluster indices (no knn interpolation)
        x, batch = self.fp3_module(x3, batch3, x2, batch2, cluster3)
        x, batch = self.fp2_module(x, batch, x1, batch1, cluster2)
        x, batch = self.fp1_module(x, batch, None, sa0_out[2], cluster1)

        if self.training:
            self.last_features = x
            self.last_pos = data.pos
            self.last_batch = batch
            self.last_proj = self.proj_head(x)  # (N, 128) L2-normalised

        x = x.unsqueeze(dim=0).permute(0, 2, 1)
        logits = torch.squeeze(self.seg_head(x)).to(torch.float)

        return logits


class NetLight(torch.nn.Module):
    """3-layer lightweight U-Net variant for distillation and fast inference.

    Same architecture family as NetFull but configurable: fewer channels,
    optional kernel tapering, and a lighter block pattern. Distillation should usually
    preserve the teacher's neighbourhood reach (k_neighbors) and kernel geometry,
    then compress width, depth, and the prediction head.

    Args:
        C: Base channel width. C=32 is a compact but expressive student for a
           C=128 teacher; C=16 is an aggressive tiny baseline.
        num_kernel_points: Scalar or 3-stage kernel counts. A scalar uses the same K in all SA stages.
        learnable_kernels: Fixed (False) by default — matches teacher, preserves flashlight semantics.
        sa1_blocks, sa2_blocks, sa3_blocks: Residual blocks per SA stage.
        k_neighbors: kNN neighbourhood size used by each SA stage.
    """
    def __init__(self, num_classes, C=32, num_kernel_points=(16, 16, 16), learnable_kernels=False,
                 drop_path_rate=0.0, spatial_mix_lite: bool = False,
                 sa1_blocks: int = 1, sa2_blocks: int = 2, sa3_blocks: int = 1,
                 k_neighbors: int = 16,
                 flash_dim: int = 32,
                 memory_efficient_conv: bool = False,
                 sparse_max: bool = True,
                 compressed_head: bool = False,
                 compressed_head_dim: int = 64):
        super(NetLight, self).__init__()
        self.model_family = "light"
        self.c_base = int(C)
        self.k_neighbors = int(k_neighbors)
        self.stage_kernel_points = normalize_stage_kernel_points(num_kernel_points)
        k1, k2, k3 = self.stage_kernel_points
        self.num_kernel_points = int(k1)
        self.learnable_kernels = bool(learnable_kernels)
        self.spatial_mix_lite = bool(spatial_mix_lite)
        self.flash_dim = int(flash_dim)
        self.memory_efficient_conv = bool(memory_efficient_conv)
        self.sparse_max = bool(sparse_max)
        self.compressed_head = bool(compressed_head)
        self.compressed_head_dim = int(compressed_head_dim)
        self.sa1_blocks = int(sa1_blocks)
        self.sa2_blocks = int(sa2_blocks)
        self.sa3_blocks = int(sa3_blocks)

        # Voxelsample resolution ratios (same as NetFull)
        self.vx_ratio_1 = (0.02 * 1.618) / 2.0
        self.vx_ratio_2 = self.vx_ratio_1 * 1.618
        self.vx_ratio_3 = self.vx_ratio_2 * 1.618

        # Channel progression: C → 1.5C → 2C → 3C (same ratios as NetFull)
        C1, C2, C3 = int(C * 1.5), C * 2, C * 3
        sa_in = lambda c, k: (c + 4) * k

        total_sa_blocks = sa1_blocks + sa2_blocks + sa3_blocks
        self.sa1_module = SAModule(0.032, self.k_neighbors, [sa_in(0, k1), C1 * 4, C1],
                                   num_blocks=sa1_blocks, num_kernel_points=k1, learnable_kernels=learnable_kernels,
                                   drop_path_rate=drop_path_rate, block_offset=0, total_blocks=total_sa_blocks,
                                   num_groups=8,
                                   spatial_mix_lite=spatial_mix_lite,
                                   flash_dim=self.flash_dim,
                                   memory_efficient_conv=self.memory_efficient_conv,
                                   sparse_max=self.sparse_max,
                                   refl_pool='max')
        self.sa2_module = SAModule(0.052, self.k_neighbors, [sa_in(C1, k2), C2 * 4, C2],
                                   num_blocks=sa2_blocks, num_kernel_points=k2, learnable_kernels=learnable_kernels,
                                   drop_path_rate=drop_path_rate, block_offset=sa1_blocks, total_blocks=total_sa_blocks,
                                   num_groups=8,
                                   spatial_mix_lite=spatial_mix_lite,
                                   flash_dim=self.flash_dim,
                                   memory_efficient_conv=self.memory_efficient_conv,
                                   sparse_max=self.sparse_max,
                                   refl_pool='max')
        self.sa3_module = SAModule(0.085, self.k_neighbors, [sa_in(C2, k3), C3 * 4, C3],
                                   num_blocks=sa3_blocks, num_kernel_points=k3, learnable_kernels=learnable_kernels,
                                   drop_path_rate=drop_path_rate, block_offset=sa1_blocks + sa2_blocks,
                                   total_blocks=total_sa_blocks,
                                   num_groups=8,
                                   spatial_mix_lite=False,
                                   flash_dim=self.flash_dim,
                                   memory_efficient_conv=self.memory_efficient_conv,
                                   sparse_max=self.sparse_max,
                                   refl_pool='max')

        # Grid unpooling (KPConvX-style) - no interpolation, sharp boundaries
        self.fp3_module = FPModule([C3 + C2, C3, C3], num_groups=8, point_norm="gn")
        self.fp2_module = FPModule([C3 + C1, C3, C3], num_groups=8, point_norm="gn")
        self.fp1_module = FPModule([C3, C3, C3], num_groups=8, point_norm="gn", guided_fusion=False)

        self.seg_head = (
            CompressedSegHead(C3, num_classes, compress=self.compressed_head_dim)
            if compressed_head
            else FullWidthSegHead(C3, num_classes)
        )
        self.proj_head = ContrastiveHead(C3, proj_dim=128)

        initialize_weights(self)

        with torch.no_grad():
            self.seg_head.classifier.bias.fill_(-0.5)

    def forward(self, data):
        voxel_size = getattr(data, 'voxel_size', None)
        if voxel_size is not None and isinstance(voxel_size, torch.Tensor) and voxel_size.numel() == 1:
            voxel_size = voxel_size.item()

        # Scale SA resolutions by voxel_size
        vx_scale = voxel_size if voxel_size is not None else 2.0
        self.sa1_module.resolution = self.vx_ratio_1 * vx_scale
        self.sa2_module.resolution = self.vx_ratio_2 * vx_scale
        self.sa3_module.resolution = self.vx_ratio_3 * vx_scale

        sa0_out = (None, data.pos, data.batch, data.reflectance, data.sf)

        # Encoder: each SA module returns (x, pos, batch, refl, sf, cluster)
        sa1_out = self.sa1_module(*sa0_out, voxel_size=voxel_size)
        x1, pos1, batch1, refl1, sf, cluster1 = sa1_out

        sa2_out = self.sa2_module(x1, pos1, batch1, refl1, sf, voxel_size=voxel_size)
        x2, pos2, batch2, refl2, sf, cluster2 = sa2_out

        sa3_out = self.sa3_module(x2, pos2, batch2, refl2, sf, voxel_size=voxel_size)
        x3, pos3, batch3, refl3, sf, cluster3 = sa3_out

        # Store encoder features for CBL / feature KD. Eval-time capture is opt-in
        # for teacher distillation so normal inference does not pay the memory cost.
        if self.training or getattr(self, 'capture_encoder_features', False):
            self.encoder_stages = [
                {'features': x1, 'pos': pos1, 'batch': batch1, 'cluster': cluster1},
                {'features': x2, 'pos': pos2, 'batch': batch2, 'cluster': cluster2},
                {'features': x3, 'pos': pos3, 'batch': batch3, 'cluster': cluster3},
            ]

        # Decoder: grid unpooling using cluster indices
        x, batch = self.fp3_module(x3, batch3, x2, batch2, cluster3)
        x, batch = self.fp2_module(x, batch, x1, batch1, cluster2)
        x, batch = self.fp1_module(x, batch, None, sa0_out[2], cluster1)

        if self.training:
            self.last_features = x
            self.last_pos = data.pos
            self.last_batch = batch
            self.last_proj = self.proj_head(x)  # (N, 128) L2-normalised

        x = x.unsqueeze(dim=0).permute(0, 2, 1)
        logits = torch.squeeze(self.seg_head(x)).to(torch.float)

        return logits
