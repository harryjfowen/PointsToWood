import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class reflectancegate(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_epochs, tau_init=1.0, tau_min=0.1):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Changed to accept single feature
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.decision_net = nn.Linear(hidden_dim, 2)
        
        self.num_epochs = num_epochs
        self.register_buffer('tau', torch.tensor(tau_init, dtype=torch.float))
        self.register_buffer('epoch', torch.tensor(0, dtype=torch.long))
        self.tau_init = tau_init
        self.tau_min = tau_min
    
    def forward(self, x, batch):
        if self.training:
            progress = self.epoch / self.num_epochs
            tau_value = self.tau_min + 0.5 * (self.tau_init - self.tau_min) * (1 + math.cos(progress * math.pi))
            self.tau = torch.tensor(tau_value, device=self.tau.device, dtype=torch.float)
            self.epoch += 1

        # Reshape reflectance to [N, 1]
        x = x.view(-1, 1)
        
        batch = batch.long()
        h = self.feature_net(x.float())
        
        attn_weights = self.attention(h).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=0)
        
        num_samples = batch.max().item() + 1
        pooled_features = torch.zeros(num_samples, h.size(1), device=x.device, dtype=h.dtype)
        weighted_features = h * attn_weights.unsqueeze(-1)
        pooled_features.scatter_add_(0, batch.unsqueeze(-1).expand(-1, h.size(1)), weighted_features)
        
        sample_counts = torch.bincount(batch, minlength=num_samples)
        pooled_features = pooled_features / torch.clamp(sample_counts.unsqueeze(-1).float(), min=1)
        
        logits = self.decision_net(pooled_features)
        decisions = F.gumbel_softmax(logits, tau=self.tau if self.training else self.tau_min, hard=True)
        return decisions[:, 0][batch]