from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def _check_nan(*tensors: Tensor, name: str = "") -> None:
    """Print a warning if any tensor contains NaNs (debug aid)."""
    for t in tensors:
        if torch.isnan(t).any():
            print(f"[NaN] detected in {name}")

def _apply_asymmetric_smoothing(labels: Tensor, pos_eps: float | None, neg_eps: float | None) -> Tensor:
    """Return label tensor with optional asymmetric label-smoothing applied.

    Args:
        labels: binary ground-truth tensor with values 0/1.
        pos_eps: smoothing factor for positive class (wood). If ``None`` no smoothing.
        neg_eps: smoothing factor for negative class (leaf). If ``None`` falls back
                 to ``pos_eps`` to keep backward compatibility with previous code.
    Returns:
        New tensor of the same shape with smoothed labels (float32).
    """
    if pos_eps is None and neg_eps is None:
        return labels.float()

    pos_eps = float(pos_eps or 0.0)
    neg_eps = float(neg_eps if neg_eps is not None else pos_eps)

    smoothed = labels.clone().float()
    smoothed[labels == 1] = 1.0 - pos_eps  
    smoothed[labels == 0] = neg_eps        
    return smoothed

class Poly1FocalLoss(nn.Module):
    def __init__(
        self,
                 epsilon: float = 0.1,
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 reduction: str = "none",
        weight: Tensor = None,
        pos_smoothing: float = None,
        neg_smoothing: float = None,
        eps: float = 1e-7
    ):
        super(Poly1FocalLoss, self).__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.weight = weight
        self.pos_smoothing = pos_smoothing
        self.neg_smoothing = neg_smoothing if neg_smoothing is not None else pos_smoothing
        self.eps = eps

    def forward(self, logits: Tensor, labels: Tensor, edge_scores: Tensor = None) -> Tensor:
        _check_nan(logits, labels, edge_scores if edge_scores is not None else torch.tensor([]), name="Poly1FocalLoss inputs")
        
        logits = torch.clamp(logits, min=-5, max=5)
        
        labels = _apply_asymmetric_smoothing(labels, self.pos_smoothing, self.neg_smoothing)
        
        probs = torch.sigmoid(logits)
        
        pt = probs * labels + (1 - probs) * (1 - labels)
        focal_weight = torch.pow(1 - pt, self.gamma)
        
        ce_loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=labels,
            reduction="none"
        )
        
        loss = focal_weight * ce_loss
        
        if edge_scores is not None:
            edge_weight = 1 + (self.gamma / 2.0) * edge_scores
            edge_weight = torch.clamp(edge_weight, min=1.0, max=2.0)
            loss = loss * edge_weight
        
        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            loss = alpha_t * loss

        poly_term = self.epsilon * torch.pow(1 - pt, self.gamma + 1)
        poly_term = torch.clamp(poly_term, min=1e-5, max=1e5)
        loss = loss + poly_term
        
        _check_nan(loss, name="Poly1FocalLoss output")
        loss = ce_loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        return loss, self.gamma

class CyclicalFocalLoss(nn.Module):
    def __init__(
        self,
        gamma_lc: float = 2.5,    
        gamma_hc: float = 0.5,    
        fc: float = 4.0,          
        num_epochs: int = None,
        alpha: float = None,
        reduction: str = "mean",
        pos_smoothing: float = None,
        neg_smoothing: float = None,
        epsilon: float = 0.1,
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
        self.pos_smoothing = pos_smoothing
        self.neg_smoothing = neg_smoothing if neg_smoothing is not None else pos_smoothing
        self.epsilon = epsilon
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

    def forward(self, logits: Tensor, labels: Tensor, edge_scores: Tensor = None) -> Tensor:
        _check_nan(logits, labels, edge_scores if edge_scores is not None else torch.tensor([]), name="CyclicalFocalLoss inputs")
        
        logits = torch.clamp(logits, min=-5, max=5)
        
        labels = _apply_asymmetric_smoothing(labels, self.pos_smoothing, self.neg_smoothing)
        
        xi = self.get_xi()
        
        probs = torch.sigmoid(logits)
        
        pt_hc = probs * labels + (1 - probs) * (1 - labels)
        pt_lc = pt_hc  
        
        focal_weight_hc = torch.pow(1 - pt_hc, self.gamma_hc)
        focal_weight_lc = torch.pow(1 - pt_lc, self.gamma_lc)
        
        ce_loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=labels,
            reduction="none"
        )
        
        Lhc = focal_weight_hc * ce_loss
        Llc = focal_weight_lc * ce_loss
        
        poly_term_hc = self.epsilon * torch.pow(1 - pt_hc, self.gamma_hc + 1)
        poly_term_lc = self.epsilon * torch.pow(1 - pt_lc, self.gamma_lc + 1)
        
        poly_term_hc = torch.clamp(poly_term_hc, min=1e-5, max=1e5)
        poly_term_lc = torch.clamp(poly_term_lc, min=1e-5, max=1e5)
        
        Lhc = Lhc + poly_term_hc
        Llc = Llc + poly_term_lc
        
        loss = xi * Lhc + (1 - xi) * Llc
        
        if edge_scores is not None:
            gamma = xi * self.gamma_hc + (1 - xi) * self.gamma_lc
            edge_weight = 1 + (gamma / 2.0) * edge_scores
            edge_weight = torch.clamp(edge_weight, min=1.0, max=2.0)
            loss = loss * edge_weight
        
        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            loss = alpha_t * loss
        
        _check_nan(loss, name="CyclicalFocalLoss output")
        loss = ce_loss
        
        if self._reduction == "mean":
            loss = loss.mean()
        elif self._reduction == "sum":
            loss = loss.sum()
        
        gamma = xi * self.gamma_hc + (1 - xi) * self.gamma_lc  
        return loss, gamma

    @property
    def reduction(self):
        return self._reduction

    @reduction.setter
    def reduction(self, value):
        self._reduction = value

    def set_epoch(self, epoch):
        self.current_epoch = epoch