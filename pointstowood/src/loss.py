import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FocalLoss(nn.Module):
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
        super(FocalLoss, self).__init__()
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

class CyclicalFocalLoss(nn.Module):
    def __init__(
        self,
        gamma_hc: float = 3.0,    # gamma for high-confidence term
        gamma_lc: float = 0.5,    # gamma for low-confidence term
        fc: float = 4.0,          # cyclical factor (paper uses 4)
        num_epochs: int = None,
        alpha: float = 0.25,
        reduction: str = "mean",  # Changed default to "mean"
        label_smoothing: float = None,
        eps: float = 1e-6
    ):
        super(CyclicalFocalLoss, self).__init__()
        self.gamma_hc = gamma_hc
        self.gamma_lc = gamma_lc
        self.fc = fc
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.eps = eps

    def get_xi(self) -> float:
        """Calculate mixing factor ξ according to paper's equation 8"""
        if self.num_epochs is None:
            return 0.5
            
        ei = self.current_epoch
        en = self.num_epochs
        fc = self.fc
        
        if fc * ei <= en:
            return 1.0 - (fc * ei / en)
        else:
            return (fc * ei / en - 1.0) / (fc - 1.0)

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
            pos_weight=label_weights
        )
                
        pt = labels * p + (1 - labels) * (1 - p)
        pt = torch.clamp(pt, min=self.eps, max=1)
        
        # Get current xi value
        xi = self.get_xi()
        
        # High confidence term (equation 2 in paper)
        focal_weight_hc = torch.pow(1 - pt, self.gamma_hc)
        loss_hc = focal_weight_hc * ce_loss
        
        # Low confidence term (equation 1 in paper)
        focal_weight_lc = torch.pow(1 - pt, self.gamma_lc)
        loss_lc = focal_weight_lc * ce_loss
        
        # Combine using equation 9
        focal_loss = xi * loss_hc + (1 - xi) * loss_lc
        
        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            focal_loss = alpha_t * focal_loss
        
        # Always reduce to scalar before returning
        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()
        else:  # Force mean reduction if none specified
            focal_loss = focal_loss.mean()
        
        return focal_loss, xi

