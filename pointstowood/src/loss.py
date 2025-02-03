import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Poly1FocalLoss(nn.Module):
    def __init__(
        self,
        epsilon: float = 0.1,  # Reduced from 1.0 to improve stability
        gamma: float = 2.0,
        alpha: float = 0.25,  # Added default for imbalance
        reduction: str = "none",
        weight: Tensor = None,
        label_smoothing: float = None,
        eps: float = 1e-6
    ):
        super(Poly1FocalLoss, self).__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.eps = eps

    def forward(self, logits: Tensor, labels: Tensor, label_weights: Tensor) -> Tensor:
        logits = torch.clamp(logits, min=-10, max=10)
        
        if self.label_smoothing is not None:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        p = torch.sigmoid(logits)
        p = torch.clamp(p, min=self.eps, max=1 - self.eps)
        
        ce_loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=labels,
            reduction="none",
            pos_weight=label_weights  # Use label_weights as the weight parameter
        )
                
        pt = labels * p + (1 - labels) * (1 - p)
        pt = torch.clamp(pt, min=self.eps, max=1)
        
        focal_weight = torch.pow(1 - pt, self.gamma)
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()
        
        return focal_loss, self.gamma