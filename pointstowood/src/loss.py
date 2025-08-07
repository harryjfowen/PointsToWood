from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union


def _check_nan(tensor: Tensor, name: str = "tensor") -> None:
    """Check for NaN values in tensor and raise error if found."""
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN values detected in {name}")


def _apply_asymmetric_smoothing(
    labels: Tensor, 
    pos_smoothing: Optional[float], 
    neg_smoothing: Optional[float]
) -> Tensor:
    """Apply asymmetric label smoothing to binary labels."""
    if pos_smoothing is None and neg_smoothing is None:
        return labels
    
    smoothed_labels = labels.clone()
    
    if pos_smoothing is not None:
        pos_mask = labels == 1
        smoothed_labels[pos_mask] = 1 - pos_smoothing
    
    if neg_smoothing is not None:
        neg_mask = labels == 0
        smoothed_labels[neg_mask] = neg_smoothing
    
    return smoothed_labels


class CyclicalFocalLoss(nn.Module):
    """
    Cyclical Focal Loss that transitions between high-capacity (low gamma) and 
    low-capacity (high gamma) focal loss based on training progress.
    
    Args:
        gamma_lc: Gamma value for low-capacity phase (higher focusing)
        gamma_hc: Gamma value for high-capacity phase (lower focusing)  
        fc: Cycle factor controlling transition speed
        num_epochs: Total number of training epochs for xi calculation
        alpha: Class balancing weight (None for no balancing)
        reduction: Loss reduction method ('mean', 'sum', or 'none')
        pos_smoothing: Label smoothing for positive samples
        neg_smoothing: Label smoothing for negative samples
        eps: Small epsilon for numerical stability
    """
    
    def __init__(
        self,
        gamma_lc: float = 2.5,
        gamma_hc: float = 0.5,
        fc: float = 4.0,
        num_epochs: Optional[int] = None,
        alpha: Optional[float] = None,
        reduction: str = "mean",
        pos_smoothing: Optional[float] = None,
        neg_smoothing: Optional[float] = None,
        eps: float = 1e-7
    ):
        super().__init__()
        
        # Validate inputs
        if gamma_lc <= 0 or gamma_hc <= 0:
            raise ValueError("Gamma values must be positive")
        if fc <= 0:
            raise ValueError("Cycle factor (fc) must be positive")
        if alpha is not None and not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1")
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'")
        if pos_smoothing is not None and not (0 <= pos_smoothing <= 1):
            raise ValueError("pos_smoothing must be between 0 and 1")
        if neg_smoothing is not None and not (0 <= neg_smoothing <= 1):
            raise ValueError("neg_smoothing must be between 0 and 1")
            
        self.gamma_lc = gamma_lc
        self.gamma_hc = gamma_hc
        self.fc = fc
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.reduction = reduction
        self.pos_smoothing = pos_smoothing
        self.neg_smoothing = neg_smoothing if neg_smoothing is not None else pos_smoothing
        self.eps = eps
        
        self.current_epoch = 0

    def get_xi(self) -> float:
        """
        Calculate the interpolation factor xi based on current epoch.
        
        Returns:
            xi: Weight for high-capacity loss (0 = pure low-capacity, 1 = pure high-capacity)
        """
        if self.num_epochs is None:
            return 0.5
            
        ei = self.current_epoch
        en = self.num_epochs
        fc = self.fc
        
        # Clamp epoch to valid range
        ei = max(0, min(ei, en))
        
        if fc * ei <= en:
            xi = 1.0 - (fc * ei / en)
        else:
            xi = (fc * ei / en - 1.0) / (fc - 1.0)
            
        # Ensure xi is in valid range [0, 1]
        return max(0.0, min(1.0, xi))

    def forward(
        self, 
        logits: Tensor, 
        labels: Tensor, 
        edge_scores: Optional[Tensor] = None
    ) -> Union[Tensor, tuple[Tensor, float]]:
        """
        Forward pass of cyclical focal loss.
        
        Args:
            logits: Raw model outputs (before sigmoid)
            labels: Binary ground truth labels
            edge_scores: Optional edge importance scores [0, 1]
            
        Returns:
            loss: Computed loss tensor
            gamma: Current effective gamma value (if reduction != 'none')
        """
        # Input validation
        if logits.shape != labels.shape:
            raise ValueError(f"Logits shape {logits.shape} doesn't match labels shape {labels.shape}")
        
        # Check for NaN inputs
        _check_nan(logits, "logits")
        _check_nan(labels, "labels")
        if edge_scores is not None:
            _check_nan(edge_scores, "edge_scores")
            if edge_scores.shape != logits.shape:
                raise ValueError(f"Edge scores shape {edge_scores.shape} doesn't match logits shape {logits.shape}")
        
        # Clamp logits for numerical stability
        logits = torch.clamp(logits, min=-10, max=10)
        
        # Apply label smoothing if specified
        labels = _apply_asymmetric_smoothing(labels, self.pos_smoothing, self.neg_smoothing)
        
        # Get interpolation factor
        xi = self.get_xi()
        
        # Calculate probabilities
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1 - self.eps)
        
        # Calculate pt for focal weighting
        pt = probs * labels + (1 - probs) * (1 - labels)
        
        # Calculate focal weights for both phases
        focal_weight_hc = torch.pow(1 - pt, self.gamma_hc)
        focal_weight_lc = torch.pow(1 - pt, self.gamma_lc)
        
        # Calculate base cross-entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=labels,
            reduction="none"
        )
        
        # Apply focal weighting for both phases
        loss_hc = focal_weight_hc * ce_loss
        loss_lc = focal_weight_lc * ce_loss
        
        # Interpolate between high and low capacity losses
        loss = xi * loss_hc + (1 - xi) * loss_lc
        
        # Apply edge weighting if provided
        if edge_scores is not None:
            edge_scores = torch.clamp(edge_scores, min=0.0, max=1.0)
            
            # Calculate current effective gamma
            current_gamma = xi * self.gamma_hc + (1 - xi) * self.gamma_lc
            
            # Create edge weights (1.0 to 1.5x multiplier based on edge importance)
            edge_weight = 1.0 + 0.5 * edge_scores * (current_gamma / 2.5)
            edge_weight = torch.clamp(edge_weight, min=1.0, max=2.0)
            
            loss = loss * edge_weight
        
        # Apply class balancing if specified
        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            loss = alpha_t * loss
        
        # Final NaN check
        _check_nan(loss, "final loss")
        
        # Apply reduction
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        # else: reduction == "none", return unreduced loss
        
        # Calculate current effective gamma for monitoring
        current_gamma = xi * self.gamma_hc + (1 - xi) * self.gamma_lc
        
        # Return loss and gamma for monitoring purposes
        if self.reduction == "none":
            return loss
        else:
            return loss, current_gamma

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch for xi calculation."""
        if epoch < 0:
            raise ValueError("Epoch must be non-negative")
        self.current_epoch = epoch

    def get_current_gamma(self) -> float:
        """Get the current effective gamma value."""
        xi = self.get_xi()
        return xi * self.gamma_hc + (1 - xi) * self.gamma_lc
    
    def get_phase_info(self) -> dict:
        """Get information about the current training phase."""
        xi = self.get_xi()
        current_gamma = self.get_current_gamma()
        
        if xi > 0.7:
            phase = "high_capacity"
        elif xi < 0.3:
            phase = "low_capacity"
        else:
            phase = "transition"
            
        return {
            "epoch": self.current_epoch,
            "xi": xi,
            "current_gamma": current_gamma,
            "phase": phase,
            "hc_weight": xi,
            "lc_weight": 1 - xi
        }

    def __repr__(self) -> str:
        return (f"CyclicalFocalLoss(gamma_lc={self.gamma_lc}, gamma_hc={self.gamma_hc}, "
                f"fc={self.fc}, alpha={self.alpha}, reduction='{self.reduction}')")

        
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
            edge_scores = torch.clamp(edge_scores, min=0.0, max=1.0)
            edge_weight = 1 + (self.gamma / 2.0) * edge_scores
            edge_weight = torch.clamp(edge_weight, min=1.0, max=2.0)
            loss = loss * edge_weight
        
        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            loss = alpha_t * loss

        # poly_term = self.epsilon * torch.pow(1 - pt, self.gamma + 1)
        # poly_term = torch.clamp(poly_term, min=1e-5, max=1e5)
        # loss = loss + poly_term
        
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
        
        loss = xi * Lhc + (1 - xi) * Llc
        
        if edge_scores is not None:
            edge_scores = torch.clamp(edge_scores, min=0.0, max=1.0)  # Add this
            gamma = xi * self.gamma_hc + (1 - xi) * self.gamma_lc
            edge_weight = 1 + (gamma / 2.0) * edge_scores
            edge_weight = torch.clamp(edge_weight, min=1.0, max=2.0)
            loss = loss * edge_weight
        
        if self.alpha is not None:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            loss = alpha_t * loss
        
        _check_nan(loss, name="CyclicalFocalLoss output")
        
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

class DifficultyAwareFocalLoss(nn.Module):
    """
    Focal loss with adaptive gamma based on both:
    - edge_scores: structural (spatial) difficulty
    - model confidence (pt): semantic difficulty

    Optionally scales the loss using semantic and edge weighting.
    """

    def __init__(self,
                 min_gamma: float = 1.0,
                 max_gamma: float = 3.0,
                 sharpness: float = 2.0,
                 alpha: float = 0.5,   # edge weight multiplier
                 beta: float = 1.0,    # semantic weight multiplier
                 use_weight_scaling: bool = True,  # toggle for loss scaling
                 reduction: str = "mean"):
        super().__init__()
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.sharpness = sharpness
        self.alpha = alpha
        self.beta = beta
        self.use_weight_scaling = use_weight_scaling
        self.reduction = reduction

    def forward(self, logits: Tensor, labels: Tensor, edge_scores: Optional[Tensor] = None):
        # Compute base probabilities
        probs = torch.sigmoid(logits).clamp(min=1e-6, max=1 - 1e-6)
        pt = probs * labels + (1 - probs) * (1 - labels)  # P(correct class)

        # BCE base loss
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        # If edge scores are given, use them for adaptive gamma
        if edge_scores is not None:
            edge_scores = torch.clamp(edge_scores, 0.0, 1.0)

            # Total difficulty = structural + semantic
            semantic_difficulty = (1.0 - pt.detach())  # detach to avoid gradient through pt
            total_difficulty = edge_scores + semantic_difficulty

            # Normalize total difficulty to [0, 1]
            total_difficulty = torch.clamp(total_difficulty / 2.0, 0.0, 1.0)

            # Gamma based on total difficulty
            adaptive_gamma = self.min_gamma + (self.max_gamma - self.min_gamma) * total_difficulty.pow(self.sharpness)
        else:
            # Use average gamma if no edge scores available
            adaptive_gamma = (self.min_gamma + self.max_gamma) / 2

        # Focal weighting
        focal_weight = torch.pow(1 - pt, adaptive_gamma)

        # Optional loss weighting based on difficulty
        if self.use_weight_scaling and edge_scores is not None:
            edge_weight = 1.0 + self.alpha * edge_scores
            semantic_weight = 1.0 + self.beta * (1.0 - pt.detach())
            total_weight = edge_weight * semantic_weight
        else:
            total_weight = 1.0

        # Final loss
        loss = total_weight * focal_weight * ce_loss

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss