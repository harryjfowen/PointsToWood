import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.nn import voxel_grid, knn
import math


class FocalLoss(nn.Module):
    """
    Cyclical Focal Loss with optional label smoothing for binary segmentation.

    Gamma cycles from 0 to gamma_max using cosine schedule, addressing the
    conflict between focal loss and early training dynamics.

    Reference: Smith & Kindermans, "Cyclical Focal Loss"

    Args:
        gamma_max: Maximum focal exponent (reached mid-training)
        alpha: Class weight for positive class (wood). None = no weighting
        label_smoothing: Smooth labels toward 0.5. 0 = no smoothing, 0.1 = recommended
        cyclical: If True, cycle gamma. If False, use fixed gamma=gamma_max
        reduction: 'mean' or 'sum' or 'none'

    Cyclical schedule (front-loaded to match OneCycleLR):
        - Epoch 0: gamma = 0 (pure BCE, strong gradients)
        - pct_peak (default 25%): gamma = gamma_max (focus on hard examples)
        - End training: gamma = 0 (long tail for stable convergence)
    """
    def __init__(self, gamma_max: float = 2.0, alpha: float = None,
                 label_smoothing: float = 0.1, cyclical: bool = True,
                 pct_peak: float = 0.33, reduction: str = 'mean'):
        super().__init__()
        self.gamma_max = gamma_max
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.cyclical = cyclical
        self.pct_peak = pct_peak
        self.reduction = reduction
        self.current_gamma = 0.0 if cyclical else gamma_max

    def set_epoch(self, epoch: int, total_epochs: int):
        """Update gamma: 0 → gamma_max at pct_peak → 0 (front-loaded cycle)."""
        if self.cyclical and total_epochs > 0:
            progress = epoch / total_epochs

            if progress < self.pct_peak:
                # Ramp up: 0 → gamma_max
                self.current_gamma = self.gamma_max * (1 - math.cos(math.pi * progress / self.pct_peak)) / 2
            else:
                # Ramp down: gamma_max → 0 (long tail)
                self.current_gamma = self.gamma_max * (1 + math.cos(math.pi * (progress - self.pct_peak) / (1 - self.pct_peak))) / 2

    def forward(self, logits: Tensor, labels: Tensor, **kwargs) -> Tensor:
        # Label smoothing: push labels away from 0/1
        if self.label_smoothing > 0:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # BCE loss per element
        bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')

        # Focal weighting with current (possibly cycled) gamma
        if self.current_gamma > 0:
            probs = torch.sigmoid(logits)
            p_t = probs * labels + (1 - probs) * (1 - labels)
            focal_weight = (1 - p_t) ** self.current_gamma
            bce = focal_weight * bce

        # Class weighting (alpha for positive class)
        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            bce = alpha_t * bce

        if self.reduction == 'mean':
            return bce.mean()
        elif self.reduction == 'sum':
            return bce.sum()
        return bce


class ReflectanceFPPenalty(nn.Module):
    """
    Penalty on leaf points predicted as wood (false positives).

    Pushes logits for leaf points below -margin. When flat=True (default): applies to all leaf
    points equally. Helps XYZ-only eval: model learns to avoid wood on leaf regardless of reflectance.
    When flat=False: refl-weighted (higher penalty on high-refl leaf points).
    """
    def __init__(self, margin: float = 1.0, strength: float = 2.0, weight: float = 1.0, ramp: bool = True, flat: bool = True):
        super().__init__()
        self.margin = margin
        self.strength = strength
        self.weight = weight
        self.ramp = ramp
        self.flat = flat
        self.ramp_factor = 1.0  # 0 → 1 over training when ramp=True

    def set_epoch(self, epoch: int, total_epochs: int):
        """Linear ramp: 0 at start → 1 at end."""
        if self.ramp and total_epochs > 0:
            self.ramp_factor = min(1.0, epoch / total_epochs)
        else:
            self.ramp_factor = 1.0

    def forward(self, logits: Tensor, labels: Tensor, reflectance: Tensor = None) -> Tensor:
        leaf = labels < 0.5
        if not leaf.any():
            return torch.tensor(0.0, device=logits.device)
        violation = F.relu(logits[leaf] + self.margin)
        if not self.flat and reflectance is not None and reflectance.numel() == logits.numel():
            refl_leaf = reflectance[leaf].float()
            refl_scale = (refl_leaf.clamp(-1.0, 1.0) + 1.0) / 2.0
            w = 1.0 + self.strength * refl_scale
        else:
            w = 1.0
        loss = (w * violation).mean()
        return self.weight * self.ramp_factor * loss


class ContrastiveBoundaryLoss(nn.Module):
    """
    Multi-scale Contrastive Boundary Learning (CBL) - faithful to CVPR 2022 paper.

    "Contrastive Boundary Learning for Point Cloud Segmentation"

    Key features (from paper):
    1. CBL applied at ALL encoder stages, not just final output
    2. Labels propagated through sub-sampling via average pooling (Eq. 6)
    3. Boundary points determined by soft label distribution at each scale

    Args:
        k: Number of neighbors for contrastive learning
        temperature: Softmax temperature (lower = sharper contrast)
        weight: Loss weight λ for CBL (paper uses 0.5-1.0)
        boundary_threshold: Soft label threshold for boundary detection (0.1-0.9 = boundary)

    Reference: "Contrastive Boundary Learning for Point Cloud Segmentation" (CVPR 2022)
    """
    def __init__(self, k: int = 16, temperature: float = 1.0, weight: float = 0.5,
                 boundary_threshold: float = 0.1, ramp: bool = True, ramp_pct: float = 0.33):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.weight = weight
        self.current_weight = weight
        self.boundary_threshold = boundary_threshold
        self.ramp = ramp
        self.ramp_pct = ramp_pct
        self.ramp_factor = 1.0 if not ramp else 0.0
        self.last_sample_std = 0.0

    def set_epoch(self, epoch: int, total_epochs: int):
        """Gamma-style warmup: CBL weight ramps 0→base over early training, then stays at base."""
        if not self.ramp or total_epochs <= 0:
            self.ramp_factor = 1.0
            self.current_weight = self.weight
            return

        progress = epoch / max(1, total_epochs)
        peak = max(1e-6, min(1.0, self.ramp_pct))
        if progress < peak:
            t = progress / peak
            self.ramp_factor = 0.5 * (1.0 - math.cos(math.pi * t))
        else:
            self.ramp_factor = 1.0
        self.current_weight = self.weight * self.ramp_factor

    def propagate_labels(self, labels: Tensor, clusters: list) -> list:
        """
        Propagate labels through encoder stages via average pooling (Eq. 6 in paper).

        Args:
            labels: Original labels [N] (0=leaf, 1=wood)
            clusters: List of cluster tensors from each SA stage

        Returns:
            List of soft labels at each stage [N_stage] in range [0, 1]
        """
        soft_labels = [labels.float()]  # Stage 0: original labels

        current_labels = labels.float()
        for cluster in clusters:
            # Average pooling: soft label = mean of labels in each voxel
            n_clusters = cluster.max().item() + 1
            label_sum = scatter_add(current_labels, cluster, dim=0, dim_size=n_clusters)
            counts = scatter_add(torch.ones_like(current_labels), cluster, dim=0, dim_size=n_clusters)
            current_labels = label_sum / counts.clamp(min=1)
            soft_labels.append(current_labels)

        return soft_labels[1:]  # Return soft labels for each encoder stage (not original)

    def compute_stage_loss(self, features: Tensor, pos: Tensor, soft_labels: Tensor,
                           batch: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute CBL loss at a single encoder stage.

        Returns per-sample normalised losses so each sample contributes equally
        regardless of boundary point density.

        Args:
            features: Point features [N, C]
            pos: Point positions [N, 3]
            soft_labels: Soft labels [N] in range [0, 1]
            batch: Batch indices [N]

        Returns:
            per_sample_loss: [B] mean CBL loss per sample (0 where no valid anchors)
            per_sample_count: [B] number of valid anchors per sample
        """
        device = features.device
        n_samples = int(batch.max().item()) + 1
        zero_loss = torch.zeros(n_samples, device=device)
        zero_count = torch.zeros(n_samples, device=device)

        is_boundary = (soft_labels > self.boundary_threshold) & (soft_labels < (1 - self.boundary_threshold))

        if not is_boundary.any():
            return zero_loss, zero_count

        boundary_idx = torch.where(is_boundary)[0]
        n_boundary = boundary_idx.size(0)

        if n_boundary < 2:
            return zero_loss, zero_count

        boundary_pos = pos[boundary_idx, :3] if pos.size(1) >= 3 else pos[boundary_idx]
        boundary_feat = features[boundary_idx]
        boundary_soft = soft_labels[boundary_idx]
        boundary_batch = batch[boundary_idx]

        boundary_labels = (boundary_soft > 0.5).long()
        boundary_feat = F.normalize(boundary_feat, dim=-1)

        k_query = min(self.k + 1, n_boundary)
        row, col = knn(boundary_pos, boundary_pos, k_query, boundary_batch, boundary_batch)

        not_self = row != col
        row, col = row[not_self], col[not_self]

        if row.size(0) == 0:
            return zero_loss, zero_count

        feat_dist = torch.norm(boundary_feat[row] - boundary_feat[col], dim=1)
        same_class = boundary_labels[row] == boundary_labels[col]

        sim = -feat_dist / self.temperature
        n_anchors = n_boundary
        sim_max = torch.zeros(n_anchors, device=device)
        sim_max.scatter_reduce_(0, row, sim, reduce='amax', include_self=False)
        sim = sim - sim_max[row]

        exp_sim = torch.exp(sim)
        sum_all = scatter_add(exp_sim, row, dim=0, dim_size=n_anchors)
        sum_pos = scatter_add(exp_sim * same_class.float(), row, dim=0, dim_size=n_anchors)

        has_pos = scatter_add(same_class.float(), row, dim=0, dim_size=n_anchors) > 0
        has_neg = scatter_add((~same_class).float(), row, dim=0, dim_size=n_anchors) > 0
        valid_anchors = has_pos & has_neg

        if not valid_anchors.any():
            return zero_loss, zero_count

        eps = 1e-8
        loss_per_anchor = -torch.log(sum_pos / (sum_all + eps) + eps)

        # Per-sample normalisation: mean loss per sample regardless of boundary density
        valid_loss = loss_per_anchor[valid_anchors]
        valid_batch = boundary_batch[valid_anchors]
        per_sample_sum = scatter_add(valid_loss, valid_batch, dim=0, dim_size=n_samples)
        per_sample_count = scatter_add(torch.ones_like(valid_loss), valid_batch, dim=0, dim_size=n_samples)

        has_anchors = per_sample_count > 0
        per_sample_loss = zero_loss.clone()
        per_sample_loss[has_anchors] = per_sample_sum[has_anchors] / per_sample_count[has_anchors]

        return per_sample_loss, per_sample_count

    def forward(self, encoder_stages: list, labels: Tensor) -> Tensor:
        """
        Multi-scale CBL loss with per-sample normalised aggregation.

        Each sample contributes equally to the loss regardless of how many boundary
        points it contains. Per-sample losses are weighted across stages by anchor
        count, then averaged over samples.

        Args:
            encoder_stages: List of dicts with 'features', 'pos', 'batch', 'cluster' per stage
            labels: Original ground truth labels [N]

        Returns:
            Per-sample normalised CBL loss scalar
        """
        if not encoder_stages:
            return torch.tensor(0.0, device=labels.device)

        clusters = [stage['cluster'] for stage in encoder_stages]
        soft_labels_per_stage = self.propagate_labels(labels, clusters)

        n_samples = int(encoder_stages[0]['batch'].max().item()) + 1
        device = labels.device
        per_sample_weighted = torch.zeros(n_samples, device=device)
        per_sample_total_count = torch.zeros(n_samples, device=device)

        for stage, soft_labels in zip(encoder_stages, soft_labels_per_stage):
            stage_loss, stage_count = self.compute_stage_loss(
                features=stage['features'],
                pos=stage['pos'],
                soft_labels=soft_labels,
                batch=stage['batch']
            )
            per_sample_weighted += stage_loss * stage_count
            per_sample_total_count += stage_count

        # Normalise per sample across stages
        has_anchors = per_sample_total_count > 0
        if not has_anchors.any():
            return torch.tensor(0.0, device=device)

        per_sample_loss = torch.zeros(n_samples, device=device)
        per_sample_loss[has_anchors] = per_sample_weighted[has_anchors] / per_sample_total_count[has_anchors]

        # Mean over samples — equal contribution per sample
        valid_losses = per_sample_loss[has_anchors]
        total_loss = valid_losses.mean()
        self.last_sample_std = float(valid_losses.std().item()) if valid_losses.numel() > 1 else 0.0

        return self.current_weight * total_loss
