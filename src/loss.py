import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


def _epoch_progress(epoch: int, total_epochs: int) -> float:
    """Discrete epoch progress in [0, 1]. Single-epoch runs use the end state."""
    if total_epochs <= 1:
        return 1.0
    progress = (int(epoch) - 1) / float(total_epochs - 1)
    return min(1.0, max(0.0, progress))


class FocalLoss(nn.Module):
    """
    Cyclical Focal Loss with optional label smoothing for binary segmentation.

    Gamma cycles from 0 to gamma_max using cosine schedule, addressing the
    conflict between focal loss and early training dynamics.

    Boundary curriculum: optional per-point upweighting of mixed-label boundary
    regions (edge_scores), ramping from 1x to (1 + boundary_max)x over training.
    Focal loss handles generic easy-vs-hard; boundary weighting targets the
    specific hard regions (wood/leaf transitions) that matter most.

    Reference: Smith & Kindermans, "Cyclical Focal Loss"

    Args:
        gamma_max: Maximum focal exponent (reached mid-training)
        alpha: Class weight for positive class (wood). None = no weighting
        label_smoothing: Smooth labels toward 0.5. 0 = no smoothing, 0.1 = recommended
        cyclical: If True, cycle gamma. If False, use fixed gamma=gamma_max
        boundary_max: Maximum extra weight for boundary points (e.g. 2.0 = 3x at peak)
        boundary_ramp_start: Fraction of training before boundary ramp begins (default 0.1)
        reduction: 'mean' or 'sum' or 'none'
    """
    def __init__(self, gamma_max: float = 2.0, alpha: float = None,
                 label_smoothing: float = 0.1, cyclical: bool = True,
                 pct_peak: float = 0.33, boundary_max: float = 2.0,
                 boundary_ramp_start: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.gamma_max = gamma_max
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.cyclical = cyclical
        self.pct_peak = pct_peak
        self.reduction = reduction
        self.current_gamma = 0.0 if cyclical else gamma_max
        self.boundary_max = boundary_max
        self.boundary_ramp_start = boundary_ramp_start
        self.boundary_weight = 0.0

    def set_epoch(self, epoch: int, total_epochs: int):
        """Update gamma and boundary weight for this epoch."""
        progress = _epoch_progress(epoch, total_epochs)

        if self.cyclical and total_epochs > 0:
            if progress < self.pct_peak:
                # Ramp up: 0 → gamma_max
                self.current_gamma = self.gamma_max * (1 - math.cos(math.pi * progress / self.pct_peak)) / 2
            else:
                # Ramp down: gamma_max → 0 (long tail)
                self.current_gamma = self.gamma_max * (1 + math.cos(math.pi * (progress - self.pct_peak) / (1 - self.pct_peak))) / 2

        # Boundary curriculum: linear ramp from 0 to boundary_max
        if total_epochs > 0 and self.boundary_max > 0:
            if progress < self.boundary_ramp_start:
                self.boundary_weight = 0.0
            else:
                t = (progress - self.boundary_ramp_start) / (1.0 - self.boundary_ramp_start)
                self.boundary_weight = self.boundary_max * min(t, 1.0)

    def forward(self, logits: Tensor, labels: Tensor, edge_scores: Tensor = None) -> Tensor:
        # Label smoothing: push labels away from 0/1
        if self.label_smoothing > 0:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # BCE loss per element
        bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')

        # Symmetric focal weighting: suppresses easy examples (confident correct predictions)
        # and focuses gradient on hard cases. Easy trunk points (pwood≈1) get (1-p_t)^γ ≈ 0;
        # uncertain twig/leaf boundaries get near-full gradient.
        gneg = self.current_gamma
        if gneg > 0:
            probs = torch.sigmoid(logits)
            p_t = probs * labels + (1 - probs) * (1 - labels)
            bce = ((1 - p_t) ** gneg) * bce

        # Class weighting (alpha for positive class)
        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            bce = alpha_t * bce

        # Boundary curriculum: upweight mixed-label regions (normalized to preserve loss scale)
        if edge_scores is not None and self.boundary_weight > 0:
            sample_weights = 1.0 + self.boundary_weight * edge_scores
            bce = bce * sample_weights
            if self.reduction == 'mean':
                return bce.sum() / sample_weights.sum().clamp(min=1.0)
            elif self.reduction == 'sum':
                return bce.sum()
            return bce

        if self.reduction == 'mean':
            return bce.mean()
        elif self.reduction == 'sum':
            return bce.sum()
        return bce


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss with boundary-focused anchor sampling.

    Samples n_anchors points proportional to edge_scores (boundary-heavy) and
    computes SupCon over the full batch as positives/negatives. Temperature anneals
    from start_temp (soft, early training) to end_temp (hard, late training). Loss
    weight ramps from 0 to `weight` over the first ramp_frac fraction of training so
    the model builds basic features before contrastive geometry is imposed.

    Reference: Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)
    """
    def __init__(self, n_anchors: int = 512, start_temp: float = 0.2,
                 end_temp: float = 0.07, weight: float = 0.1,
                 ramp_frac: float = 0.15):
        super().__init__()
        self.n_anchors = int(n_anchors)
        self.start_temp = float(start_temp)
        self.end_temp = float(end_temp)
        self.weight = float(weight)
        self.ramp_frac = float(ramp_frac)
        self.temperature = float(start_temp)
        self.ramp_factor = 0.0

    def set_epoch(self, epoch: int, total_epochs: int):
        progress = _epoch_progress(epoch, total_epochs)
        self.temperature = self.start_temp + (self.end_temp - self.start_temp) * progress
        self.ramp_factor = min(1.0, progress / max(self.ramp_frac, 1e-6))

    def forward(self, features: Tensor, labels: Tensor, edge_scores: Tensor = None) -> Tensor:
        N = features.shape[0]
        if N < 4 or self.ramp_factor <= 0.0:
            return torch.tensor(0.0, device=features.device)

        if edge_scores is not None and edge_scores.numel() == N:
            weights = (0.5 + 0.5 * edge_scores.float().clamp(0.0, 1.0)).to(features.device)
        else:
            weights = torch.ones(N, device=features.device)

        n_a = min(self.n_anchors, N)
        anchor_idx = torch.multinomial(weights, n_a, replacement=False)

        z_a = features[anchor_idx]           # (n_a, D)
        labels_a = labels[anchor_idx].float()  # (n_a,)
        labels_all = labels.float()            # (N,)

        sim = torch.mm(z_a, features.T) / self.temperature  # (n_a, N)

        # Mask self out of denominator
        self_mask = torch.zeros(n_a, N, dtype=torch.bool, device=features.device)
        self_mask[torch.arange(n_a, device=features.device), anchor_idx] = True
        sim_no_self = sim.masked_fill(self_mask, -1e4)

        # Positive mask: same class as anchor, not self
        pos_mask = (labels_all.unsqueeze(0) == labels_a.unsqueeze(1)) & ~self_mask  # (n_a, N)
        has_pos = pos_mask.sum(dim=1) > 0
        if not has_pos.any():
            return torch.tensor(0.0, device=features.device)

        log_denom = torch.logsumexp(sim_no_self, dim=1)              # (n_a,)
        pos_count = pos_mask.float().sum(dim=1).clamp(min=1.0)
        pos_sim_sum = (sim * pos_mask.float()).sum(dim=1)

        loss_per_anchor = log_denom - pos_sim_sum / pos_count        # (n_a,)
        return (self.weight * self.ramp_factor) * loss_per_anchor[has_pos].mean()


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
            self.ramp_factor = _epoch_progress(epoch, total_epochs)
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


