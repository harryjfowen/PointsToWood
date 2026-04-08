import torch
import torch.nn.functional as F
from torch_geometric.nn import voxel_grid, knn, knn_graph
from torch.nn import Sequential as Seq, Linear as Lin
from src.AnisotropicConv import AnisotropicConv
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter_max, scatter_add
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


def _effective_num_groups(channel: int, target: int) -> int:
    """Largest divisor of channel that is <= target and >= 1 (for GroupNorm)."""
    if target <= 0 or channel <= 0:
        return 1
    g = min(target, channel)
    while g > 0 and channel % g != 0:
        g -= 1
    return max(1, g)


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

        if self.use_skip and batch is not None:
            lengths = torch.bincount(batch)
            out = self.drop_path(out, lengths)

        if self.layer_scale is not None:
            out = out * self.layer_scale.unsqueeze(0)

        out = self.activation(out + residual)
        return out


class SAModule(torch.nn.Module):

    def __init__(self, resolution, k, NN, num_blocks=1, num_kernel_points=16, learnable_kernels=False,
                 drop_path_rate=0.0, block_offset=0, total_blocks=12, num_groups=16,
                 refl_gate_bias_init: float = 0.0, refl_gate_cap: float = 1.0,
                 dualnorm_lite: bool = False, spatial_mix_lite: bool = False):
        super(SAModule, self).__init__()
        self.resolution = resolution
        self.k = k
        self.spatial_mix_lite = bool(spatial_mix_lite)

        self.conv = AnisotropicConv(
            local_nn=MLP(NN, num_groups=num_groups, conv_style=True),
            global_nn=None,
            add_self_loops=False,
            num_kernel_points=num_kernel_points,
            learnable_kernels=learnable_kernels,
            use_softmax=True,
            refl_gate_bias_init=refl_gate_bias_init,
            refl_gate_cap=refl_gate_cap,
            use_dualnorm_lite=dualnorm_lite,
        )
        # KPConvX-style global depth-linear schedule: block at global idx l gets (l+1)/total * rate
        drop_probs = [(block_offset + i + 1) / total_blocks * drop_path_rate for i in range(num_blocks)] if drop_path_rate > 0 else [0.0] * num_blocks
        self.residual_blocks = nn.ModuleList([
            InvertedBottleneck(NN[-1], NN[-1], drop_prob=p, num_groups=num_groups, spatial_mixing=self.spatial_mix_lite)
            for p in drop_probs
        ])

    def voxelsample(self, pos, batch, resolution):
        """Voxel grid downsampling (picks one representative per voxel cell).

        Returns:
            idx: indices of selected points (one per voxel)
            cluster: mapping from each input point to its voxel/coarse point index
                     Used for grid unpooling in decoder (no knn needed)
        """
        res_scalar = resolution.item() if isinstance(resolution, torch.Tensor) else resolution
        voxel_indices = voxel_grid(pos, res_scalar, batch)
        cluster, perm = consecutive_cluster(voxel_indices)
        return perm, cluster

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
        pos = torch.cat([pos[:, :3], reflectance.unsqueeze(-1)], dim=-1)

        # 1. Graph construction in METRES (k-NN, density-agnostic)
        # Always exactly k neighbours regardless of point spacing — sensor-agnostic.
        # Also get cluster mapping for grid unpooling in decoder.
        idx, cluster = self.voxelsample(pos[:, :3], batch, self.resolution)
        batch_coarse = batch[idx]
        row, col = knn(
            pos[:, :3],       # source: all fine points
            pos[idx, :3],     # query: coarse (voxel-sampled) points
            k=self.k,
            batch_x=batch,
            batch_y=batch_coarse,
        )
        edge_index = torch.stack([col, row], dim=0)

        # 2. Normalize xyz to unit scale for conv (pos/sf → ~[-1,1])
        # Non-in-place: critical for gradient checkpointing (forward reruns during backward)
        sf_scale = sf[batch].unsqueeze(-1)
        pos_conv = torch.cat([pos[:, :3] / sf_scale, pos[:, 3:]], dim=-1)

        # 3. Conv: radial feature normalized by max_d (density-invariant), not fixed radius.
        x = self.conv(
            x,
            (pos_conv, pos_conv[idx]),
            edge_index,
        )

        residual_edge_index = None
        if self.spatial_mix_lite:
            # k-NN self-graph on coarse points for SpatialMixLite residual blocks.
            k_mix = min(8, self.k)
            residual_edge_index = knn_graph(
                pos[idx, :3],
                k=k_mix,
                batch=batch_coarse,
                loop=False,
            )

        for block in self.residual_blocks:
            x = block(x, batch_coarse, edge_index=residual_edge_index)

        # pos was never mutated (non-in-place scaling), so idx into original pos for metres
        pos_out = pos[idx, :3]
        reflectance = pos[idx, 3]
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
    def __init__(self, NN, drop_prob=0.0, num_groups=8, point_norm: str = "gn"):
        super(FPModule, self).__init__()
        self.NN = MLP(NN, num_groups=num_groups, point_norm=point_norm)
        self.residual_block = InvertedBottleneck(NN[-1], NN[-1], drop_prob=drop_prob, num_groups=num_groups)

    def forward(self, x, batch, x_skip, batch_skip, cluster):
        """
        Args:
            x: coarse features [N_coarse, C]
            batch: coarse batch indices
            x_skip: fine skip features [N_fine, C_skip]
            batch_skip: fine batch indices
            cluster: mapping from fine points to coarse points [N_fine]
                     cluster[i] = index of coarse point for fine point i
        """
        # Grid unpooling: directly gather coarse features using cluster indices
        # No knn search, no distance weighting - sharp boundaries preserved
        x_interp = x[cluster]

        if x_skip is not None:
            x_out = torch.cat([x_interp, x_skip], dim=1)
        else:
            x_out = x_interp

        x_out = self.NN(x_out)
        x_out = self.residual_block(x_out, batch_skip)
        return x_out, batch_skip


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
    - SA1, SA2, SA3: Flashlight-enabled (reflectance structure tensor)
    - Block pattern 4-6-2: weighted toward fine/mid scale where leaf-twig errors occur
    """
    def __init__(self, num_classes, C=128, num_kernel_points=16, learnable_kernels=False, drop_path_rate=0.0, dualnorm_lite: bool = False, spatial_mix_lite: bool = False, use_gradient_checkpointing: bool = True):
        super(NetFull, self).__init__()

        # Gradient checkpointing: trade computation for memory (20-30% memory savings, ~15-20% slowdown)
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Voxelsample resolution RATIOS (relative to voxel_size, for scale invariance)
        # At runtime, multiply by voxel_size so resolutions scale with input
        # Base: 2m voxel (grid_size=2) → 2cm input → 3.2cm, 5.2cm, 8.5cm
        self.vx_ratio_1 = (0.02 * 1.618) / 2.0  # 0.016 → 1.6× input spacing (~3.2cm)
        self.vx_ratio_2 = self.vx_ratio_1 * 1.618  # 0.026 → 2.6× input spacing (~5.2cm)
        self.vx_ratio_3 = self.vx_ratio_2 * 1.618  # 0.042 → 4.2× input spacing (~8.5cm)

        # Channel progression: C → 1.5C → 2C → 3C
        C1, C2, C3 = int(C * 1.5), C * 2, C * 3
        # agg_feat (F*K) + eigvals_gated (3)
        # F = incoming channels + 3 direction dims + 1 normalized radial distance.
        # Total: (c + 4)*K + 3
        sa_in = lambda c: (c + 4) * num_kernel_points + 3

        # SAModules: resolution set dynamically in forward() based on voxel_size
        # Block pattern 4-6-2: weighted toward fine/mid scale where leaf-twig errors occur.
        # SA3 needs only 2 blocks — global context is simple at 8.5cm.
        # DropPath: KPConvX-style global depth-linear schedule across all 12 SA blocks
        total_sa_blocks = 4 + 6 + 2
        self.sa1_module = SAModule(0.032, 16, [sa_in(0), C1 * 4, C1],
                                   num_blocks=4, num_kernel_points=num_kernel_points, learnable_kernels=learnable_kernels,
                                   drop_path_rate=drop_path_rate, block_offset=0, total_blocks=total_sa_blocks,
                                   num_groups=16, refl_gate_bias_init=-0.8, refl_gate_cap=0.85,
                                   dualnorm_lite=dualnorm_lite, spatial_mix_lite=False)
        self.sa2_module = SAModule(0.052, 16, [sa_in(C1), C2 * 4, C2],
                                   num_blocks=6, num_kernel_points=num_kernel_points, learnable_kernels=learnable_kernels,
                                   drop_path_rate=drop_path_rate, block_offset=4, total_blocks=total_sa_blocks,
                                   num_groups=16, refl_gate_bias_init=-0.3, refl_gate_cap=0.95,
                                   dualnorm_lite=dualnorm_lite, spatial_mix_lite=spatial_mix_lite)
        self.sa3_module = SAModule(0.085, 16, [sa_in(C2), C3 * 4, C3],
                                   num_blocks=2, num_kernel_points=num_kernel_points, learnable_kernels=learnable_kernels,
                                   drop_path_rate=drop_path_rate, block_offset=10, total_blocks=total_sa_blocks,
                                   num_groups=16, refl_gate_bias_init=-0.3, refl_gate_cap=0.95,
                                   dualnorm_lite=dualnorm_lite, spatial_mix_lite=False)

        # Grid unpooling (KPConvX-style) - no interpolation, sharp boundaries
        self.fp3_module = FPModule([C3 + C2, C3, C3], num_groups=16, point_norm="gn")
        self.fp2_module = FPModule([C3 + C1, C3, C3], num_groups=16, point_norm="gn")
        self.fp1_module = FPModule([C3, C3, C3], num_groups=16, point_norm="gn")

        # LayerNorm over channels (per-point) in head; GN in residuals
        self.seg_head = nn.Sequential(
            nn.Conv1d(C3, C3 // 2, 1),
            LayerNormChannels(C3 // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(C3 // 2, num_classes, 1)
        )

        initialize_weights(self)

        # Initialize output bias negative: model starts predicting leaf (0),
        # must learn positive evidence for wood (1). bias=-0.5 → sigmoid≈0.38
        with torch.no_grad():
            self.seg_head[-1].bias.fill_(-0.5)

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
        # SA1, SA2: flashlight-enabled (use reflectance)
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

        # SA3: flashlight at coarse scale (~8cm) – bright sampling + pattern_diff for coarse bright vs dark regions
        if self.use_gradient_checkpointing and self.training:
            sa3_out = checkpoint(self.sa3_module, x2, pos2, batch2, refl2, sf, voxel_size=voxel_size, use_reentrant=False)
        else:
            sa3_out = self.sa3_module(x2, pos2, batch2, refl2, sf, voxel_size=voxel_size)
        x3, pos3, batch3, refl3, sf, cluster3 = sa3_out

        # Store encoder features for CBL / feature KD
        self.encoder_stages = [
            {'features': x1, 'pos': pos1, 'batch': batch1, 'cluster': cluster1},
            {'features': x2, 'pos': pos2, 'batch': batch2, 'cluster': cluster2},
            {'features': x3, 'pos': pos3, 'batch': batch3, 'cluster': cluster3},
        ]

        # Decoder: grid unpooling using cluster indices (no knn interpolation)
        x, batch = self.fp3_module(x3, batch3, x2, batch2, cluster3)
        x, batch = self.fp2_module(x, batch, x1, batch1, cluster2)
        x, batch = self.fp1_module(x, batch, None, sa0_out[2], cluster1)

        # Store pre-head features for contrastive boundary learning
        self.last_features = x
        self.last_pos = data.pos
        self.last_batch = batch

        x = x.unsqueeze(dim=0).permute(0, 2, 1)
        logits = torch.squeeze(self.seg_head(x)).to(torch.float)

        return logits


class NetLight(torch.nn.Module):
    """3-layer lightweight U-Net variant for distillation and fast inference.

    Same architecture as NetFull but configurable: fewer channels, fewer kernel points,
    and a lighter block pattern. Defaults match a good distillation student (C=32, K=8,
    fixed kernels, blocks=1-2-1).

    Args:
        C: Base channel width. C=32 → ~6x smaller than NetFull (C=128).
        num_kernel_points: Kernel points per conv (8 vs 16 in NetFull).
        learnable_kernels: Fixed (False) by default — matches teacher, preserves flashlight semantics.
        sa1_blocks, sa2_blocks, sa3_blocks: Residual blocks per SA stage. Pattern 1-2-1 is lean;
            2-3-1 gives more capacity at fine/mid scale if needed.
    """
    def __init__(self, num_classes, C=32, num_kernel_points=8, learnable_kernels=False,
                 drop_path_rate=0.0, dualnorm_lite: bool = False, spatial_mix_lite: bool = False,
                 sa1_blocks: int = 1, sa2_blocks: int = 2, sa3_blocks: int = 1):
        super(NetLight, self).__init__()

        # Voxelsample resolution ratios (same as NetFull)
        self.vx_ratio_1 = (0.02 * 1.618) / 2.0
        self.vx_ratio_2 = self.vx_ratio_1 * 1.618
        self.vx_ratio_3 = self.vx_ratio_2 * 1.618

        # Channel progression: C → 1.5C → 2C → 3C (same ratios as NetFull)
        C1, C2, C3 = int(C * 1.5), C * 2, C * 3
        sa_in = lambda c: (c + 4) * num_kernel_points + 3

        # KPConvX global depth-linear DropPath schedule across all SA blocks
        total_sa_blocks = sa1_blocks + sa2_blocks + sa3_blocks
        self.sa1_module = SAModule(0.032, 16, [sa_in(0), C1 * 4, C1],
                                   num_blocks=sa1_blocks, num_kernel_points=num_kernel_points, learnable_kernels=learnable_kernels,
                                   drop_path_rate=drop_path_rate, block_offset=0, total_blocks=total_sa_blocks,
                                   num_groups=8, refl_gate_bias_init=-0.8, refl_gate_cap=0.85,
                                   dualnorm_lite=dualnorm_lite, spatial_mix_lite=False)
        self.sa2_module = SAModule(0.052, 16, [sa_in(C1), C2 * 4, C2],
                                   num_blocks=sa2_blocks, num_kernel_points=num_kernel_points, learnable_kernels=learnable_kernels,
                                   drop_path_rate=drop_path_rate, block_offset=sa1_blocks, total_blocks=total_sa_blocks,
                                   num_groups=8, refl_gate_bias_init=-0.3, refl_gate_cap=0.95,
                                   dualnorm_lite=dualnorm_lite, spatial_mix_lite=spatial_mix_lite)
        self.sa3_module = SAModule(0.085, 16, [sa_in(C2), C3 * 4, C3],
                                   num_blocks=sa3_blocks, num_kernel_points=num_kernel_points, learnable_kernels=learnable_kernels,
                                   drop_path_rate=drop_path_rate, block_offset=sa1_blocks + sa2_blocks, total_blocks=total_sa_blocks,
                                   num_groups=8, refl_gate_bias_init=-0.3, refl_gate_cap=0.95,
                                   dualnorm_lite=dualnorm_lite, spatial_mix_lite=False)

        # Grid unpooling (KPConvX-style) - no interpolation, sharp boundaries
        self.fp3_module = FPModule([C3 + C2, C3, C3], num_groups=8, point_norm="gn")
        self.fp2_module = FPModule([C3 + C1, C3, C3], num_groups=8, point_norm="gn")
        self.fp1_module = FPModule([C3, C3, C3], num_groups=8, point_norm="gn")

        # LayerNorm over channels (per-point) in head; GN in residuals
        self.seg_head = nn.Sequential(
            nn.Conv1d(C3, C3 // 2, 1),
            LayerNormChannels(C3 // 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(C3 // 2, num_classes, 1)
        )

        initialize_weights(self)

        # Initialize output bias negative: model starts predicting leaf (0),
        # must learn positive evidence for wood (1). bias=-0.5 → sigmoid≈0.38
        with torch.no_grad():
            self.seg_head[-1].bias.fill_(-0.5)

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
        # SA1, SA2: flashlight-enabled (use reflectance)
        sa1_out = self.sa1_module(*sa0_out, voxel_size=voxel_size)
        x1, pos1, batch1, refl1, sf, cluster1 = sa1_out

        sa2_out = self.sa2_module(x1, pos1, batch1, refl1, sf, voxel_size=voxel_size)
        x2, pos2, batch2, refl2, sf, cluster2 = sa2_out

        # SA3: flashlight at coarse scale (~8cm) – bright sampling + pattern_diff for coarse bright vs dark regions
        sa3_out = self.sa3_module(x2, pos2, batch2, refl2, sf, voxel_size=voxel_size)
        x3, pos3, batch3, refl3, sf, cluster3 = sa3_out

        # Store encoder features for CBL / feature KD
        self.encoder_stages = [
            {'features': x1, 'pos': pos1, 'batch': batch1, 'cluster': cluster1},
            {'features': x2, 'pos': pos2, 'batch': batch2, 'cluster': cluster2},
            {'features': x3, 'pos': pos3, 'batch': batch3, 'cluster': cluster3},
        ]

        # Decoder: grid unpooling using cluster indices
        x, batch = self.fp3_module(x3, batch3, x2, batch2, cluster3)
        x, batch = self.fp2_module(x, batch, x1, batch1, cluster2)
        x, batch = self.fp1_module(x, batch, None, sa0_out[2], cluster1)

        # Store pre-head features for contrastive boundary learning
        self.last_features = x
        self.last_pos = data.pos
        self.last_batch = batch

        x = x.unsqueeze(dim=0).permute(0, 2, 1)
        logits = torch.squeeze(self.seg_head(x)).to(torch.float)

        return logits
