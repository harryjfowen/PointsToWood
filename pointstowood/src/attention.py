import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectanceGatingAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_net = nn.Linear(4, hidden_dim)  # 3 for XYZ + 1 for Reflectance
        self.key_net = nn.Linear(4, hidden_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.normal_(self.query_net.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.key_net.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.query_net.bias)
        nn.init.zeros_(self.key_net.bias)
    
    def forward(self, xyz, reflectance, batch):
        """
        xyz: [N, 3] - XYZ coordinates
        reflectance: [N] - Reflectance values
        batch: [N] - Batch indices
        """
        # Ensure reflectance is [N, 1]
        if reflectance.dim() == 1:
            reflectance = reflectance.unsqueeze(-1)  # [N] -> [N, 1]
        
        # Concatenate XYZ and reflectance
        features = torch.cat([xyz, reflectance], dim=-1)  # [N, 4]
        
        # Learn attention patterns
        q = self.query_net(features)  # [N, hidden_dim]
        k = self.key_net(features)    # [N, hidden_dim]
        
        # Compute attention scores
        attn_scores = (q * k).sum(dim=-1, keepdim=True)  # [N, 1]
        
        # Apply sigmoid to gate reflectance
        attn_weights = torch.sigmoid(attn_scores)
        
        # Reweight reflectance based on attention weights
        refined_reflectance = reflectance * attn_weights
        
        return refined_reflectance.squeeze(-1)
    

