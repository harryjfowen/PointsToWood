import torch
import torch.nn as nn

class DifferentiableSampler(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, batch, ratio=0.5, tau=1.0):
        """
        Vectorized differentiable point sampling
        Args:
            x (Tensor): Point features [N, C]
            batch (Tensor): Batch assignments [N]
            ratio (float): Sampling ratio
            tau (float): Temperature for Gumbel-Softmax
        Returns:
            Tensor: Selected point indices [N * ratio]
        """
        # Get logits (unnormalized scores)
        logits = self.score_net(x).squeeze(-1)
        
        selected_indices = []
        unique_batches = torch.unique(batch)
        
        for b in unique_batches:
            batch_mask = (batch == b)
            batch_logits = logits[batch_mask]
            
            # Convert to probability simplex using softmax
            probs = torch.softmax(batch_logits, dim=0)
            
            if self.training:
                # Draw Gumbel noise
                gumbel_noise = -torch.empty_like(probs).exponential_().log()
                gumbel_noise = -torch.empty_like(gumbel_noise).exponential_().log()
                
                # Gumbel-Softmax with temperature (Eq. 3)
                y_soft = torch.softmax((torch.log(probs + 1e-10) + gumbel_noise) / tau, dim=0)
                
                # Get number of samples for this batch
                num_samples = max(1, int(batch_mask.sum() * ratio))
                
                # Select top-k based on soft probabilities
                _, top_indices = torch.topk(y_soft, k=num_samples)
                
            else:
                # Gumbel-Max for test phase (Eq. 4)
                gumbel_noise = -torch.empty_like(probs).exponential_().log()
                y_hard = torch.log(probs + 1e-10) + gumbel_noise
                
                # Get number of samples for this batch
                num_samples = max(1, int(batch_mask.sum() * ratio))
                
                # Select top-k using hard Gumbel-Max
                _, top_indices = torch.topk(y_hard, k=num_samples)
            
            # Map back to original indices
            global_indices = torch.where(batch_mask)[0][top_indices]
            selected_indices.append(global_indices)
        
        return torch.cat(selected_indices)

    def extra_repr(self) -> str:
        """String representation"""
        return f'score_net={self.score_net}'