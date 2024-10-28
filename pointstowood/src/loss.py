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

    def forward(self, logits: Tensor, labels: Tensor, label_weights: Tensor = None) -> Tensor:
        # Prevent extreme logits
        logits = torch.clamp(logits, min=-10, max=10)
        
        # Apply label smoothing if specified
        if self.label_smoothing is not None:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Compute probabilities
        p = torch.sigmoid(logits)
        p = torch.clamp(p, min=self.eps, max=1 - self.eps)
        
        # Compute CE loss
        ce_loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=labels,
            reduction="none",
            weight=self.weight,
        )
        ce_loss = torch.clamp(ce_loss, max=100.0)  # Prevent extreme loss values
        
        # Compute pt
        pt = labels * p + (1 - labels) * (1 - p)
        pt = torch.clamp(pt, min=self.eps, max=1 - self.eps)
        
        # Compute focal weight with gradient clipping
        focal_weight = torch.pow(1 - pt, self.gamma)
        focal_weight = torch.clamp(focal_weight, max=2.0)  # Prevent extreme weights
        
        # Compute focal loss
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha if specified
        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            focal_loss = alpha_t * focal_loss
        
        # Add poly term with stability
        poly_term = self.epsilon * torch.pow(1 - pt, self.gamma + 1)
        poly_term = torch.clamp(poly_term, max=100.0)  # Prevent extreme values
        
        loss = focal_loss + poly_term
        
        # Final safety clamp
        loss = torch.clamp(loss, min=0.0, max=100.0)
        
        # Handle any remaining NaN values
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        return loss, self.gamma