import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

class FocalLoss(nn.Module):
    def __init__(
        self,
        epsilon: float = None, 
        gamma: float = 2.0,
        alpha: float = 0.5,  
        reduction: str = "mean",
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
        
        self.register_buffer('running_pos_weight', torch.tensor(1.0))
        self.momentum = 0.9  

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        logits = torch.clamp(logits, min=-10, max=10)
        
        if self.label_smoothing is not None:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        num_neg = (labels == 0).sum().float()
        num_pos = (labels == 1).sum().float()
        current_pos_weight = num_neg / num_pos if num_pos > 0 else torch.tensor(1.0, device=labels.device)
        current_pos_weight = torch.clamp(current_pos_weight, min=1.0, max=2.0)
        
        with torch.no_grad():  
            self.running_pos_weight = (
                self.momentum * self.running_pos_weight + 
                (1 - self.momentum) * current_pos_weight
            )
        
        smoothed_pos_weight = self.running_pos_weight.clone()
        
        p = torch.sigmoid(logits)
        p = torch.clamp(p, min=self.eps, max=1 - self.eps)
        
        ce_loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=labels,
            reduction="none",
            pos_weight=smoothed_pos_weight  # Use smoothed weight
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
        gamma_lc: float = 0.5,    # LOW gamma for confident predictions (early/late)
        gamma_hc: float = 3.0,    # HIGH gamma for hard examples (middle)
        fc: float = 4.0,          
        num_epochs: int = None,
        alpha: float = None,
        reduction: str = "mean",
        label_smoothing: float = None,
        eps: float = 1e-7
    ):
        super().__init__()
        self.gamma_lc = gamma_lc
        self.gamma_hc = gamma_hc
        self.fc = fc
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.alpha = alpha
        self._reduction = reduction
        self.label_smoothing = label_smoothing
        self.eps = eps
        
        self.register_buffer('running_pos_weight', torch.tensor(1.0))
        self.momentum = 0.9

    def get_xi(self) -> float:
        if self.num_epochs is None:
            return 0.5
            
        ei = self.current_epoch
        en = self.num_epochs
        fc = self.fc
        
        if fc * ei <= en:
            xi = 1.0 - (fc * ei / en)
        else:
            xi = (fc * ei / en - 1.0) / (fc - 1.0)
        return xi

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        logits = torch.clamp(logits, min=-10, max=10)
        
        if self.label_smoothing is not None:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        num_neg = (labels == 0).sum().float()
        num_pos = (labels == 1).sum().float()
        current_pos_weight = num_neg / num_pos if num_pos > 0 else torch.tensor(1.0, device=labels.device)
        current_pos_weight = torch.clamp(current_pos_weight, min=1.0, max=2.0)
        
        with torch.no_grad():
            self.running_pos_weight = (
                self.momentum * self.running_pos_weight + 
                (1 - self.momentum) * current_pos_weight
            )
        
        p = torch.sigmoid(logits)
        p = torch.clamp(p, min=self.eps, max=1 - self.eps)
        
        ce_loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=labels,
            reduction="none",
            pos_weight=self.running_pos_weight
        )
                
        pt = labels * p + (1 - labels) * (1 - p)
        pt = torch.clamp(pt, min=self.eps, max=1)
        
        xi = self.get_xi()
        
        current_gamma = xi * self.gamma_lc + (1 - xi) * self.gamma_hc
        
        modulating_factor = torch.pow(1 - pt, current_gamma)
        loss = modulating_factor * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            loss = alpha_t * loss
        
        if self._reduction == "mean":
            loss = loss.mean()
        elif self._reduction == "sum":
            loss = loss.sum()
        
        return loss, current_gamma

    @property
    def reduction(self):
        return self._reduction

    @reduction.setter
    def reduction(self, value):
        self._reduction = value

    def set_epoch(self, epoch):
        self.current_epoch = epoch

