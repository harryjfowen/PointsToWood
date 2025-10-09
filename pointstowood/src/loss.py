import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

class DynamicEdgeFocalLoss(nn.Module):
    def __init__(self,
                 min_gamma: float = 1.0,
                 max_gamma: float = 3.0,
                 sharpness: float = 2.0,
                 alpha_min: float = 0.1,
                 alpha_max: float = 1.0,
                 ramp_start: float = 0.75,
                 reduction: str = "mean"):
        super().__init__()
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.sharpness = sharpness
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.ramp_start = ramp_start
        self.reduction = reduction
        self.current_epoch = 0
        self.total_epochs = 100

    def forward(self, logits: Tensor, labels: Tensor, edge_scores: Optional[Tensor] = None):
        probs = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
        pt = probs * labels + (1 - probs) * (1 - labels)
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        if edge_scores is None:
            edge_scores = torch.zeros_like(pt)

        ramp_epoch = int(self.ramp_start * self.total_epochs)
        if self.current_epoch >= ramp_epoch:
            progress = (self.current_epoch - ramp_epoch) / (self.total_epochs - ramp_epoch)
            alpha_eff = self.alpha_min + (self.alpha_max - self.alpha_min) * 0.5 * (1 - torch.cos(torch.tensor(progress * torch.pi)))
        else:
            alpha_eff = self.alpha_min

        semantic_diff = (1 - pt.detach())
        d_eff = (semantic_diff + alpha_eff * edge_scores) / (1 + alpha_eff)
        d_eff = torch.clamp(d_eff, 0.0, 1.0)

        gamma = self.min_gamma + (self.max_gamma - self.min_gamma) * d_eff.pow(self.sharpness)
        focal_weight = torch.pow(1 - pt, gamma)
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def set_epoch(self, epoch: int, total_epochs: int = None):
        self.current_epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs