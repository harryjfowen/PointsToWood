import torch
from torch import nn
from torch_geometric.nn import knn
from torch_geometric.nn import voxel_grid, knn
from torch_scatter import scatter_max, scatter_min
import torch.nn.init as init

# class mixed_KNN(nn.Module):
#     def __init__(self, feature_dim, resolution, k=32):
#         super().__init__()
#         self.resolution = resolution
#         self.k = k
#         self.feature_dim = feature_dim + 1
#         self.alpha = nn.Parameter(torch.tensor(0.5))  
#         self.feature_compressor = nn.Sequential(
#             nn.Linear(feature_dim + 1, 16),
#             nn.ReLU(),
#             nn.Linear(16, 3)
#         )
    
#     def forward(self, pos, x, batch):
#         idx = self.voxelsample(pos, batch)
#         x = torch.cat([x, pos[:, 3].unsqueeze(1)], dim=1)
#         compressed_x = self.feature_compressor(x)
#         compressed_x = compressed_x / compressed_x.max(dim=0).values        
#         pos_normalized = pos[:, :3] / pos[:, :3].max(dim=0).values
#         compressed_x = (compressed_x - compressed_x.mean(dim=0)) / compressed_x.std(dim=0)
#         fused_vector = self.alpha * compressed_x + (1 - self.alpha) * pos_normalized
#         row, col = knn(fused_vector, fused_vector[idx], k=self.k, batch_x=batch, batch_y=batch[idx])
#         return row, col, idx

#     def voxelsample(self, pos, batch):
#         voxel_indices = voxel_grid(pos[:, :3], self.resolution, batch)
#         _, remapped_indices = torch.unique(voxel_indices, return_inverse=True)
#         _, max_indices = scatter_max(pos[:, 3], remapped_indices, dim=0)
#         idx = max_indices[max_indices != -1]
#         return idx


class mixed_KNN(nn.Module):
    def __init__(self, feature_dim, resolution, k=32):
        super().__init__()
        self.resolution = resolution
        self.k = k
        self.feature_dim = feature_dim + 1
        
        # Feature compression module
        self.feature_compressor = nn.Sequential(
            nn.Linear(feature_dim + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        
        # Dynamic weight model for per-point weighting
        self.weight_model = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)  # Ensure w_xyz + w_f = 1
        )
        # Initialize the weight model with custom values for the penultimate layer
        self.initialize_weights()
            
    def initialize_weights(self):
        # Apply Xavier initialization to the first Linear layer (with 8 input features)
        init.xavier_uniform_(self.weight_model[0].weight)
        init.zeros_(self.weight_model[0].bias)
        
        # Apply Xavier initialization to the second Linear layer (penultimate)
        init.xavier_uniform_(self.weight_model[2].weight)
        init.zeros_(self.weight_model[2].bias)
        
        # Manually initialize the final Linear layer (to make positional info dominant)
        with torch.no_grad():
            # Initialize weights with a much larger weight for w_xyz (positional component)
            self.weight_model[2].weight.data[0, :] = torch.tensor([10.0] * 16)  # Strong weight for w_xyz (10 times larger)
            
            # Initialize weights for w_f (feature component) to be much smaller
            self.weight_model[2].weight.data[1, :] = torch.tensor([0.0] * 16)  # No contribution from features at the start
            
            # Set bias for w_f to zero (no initial influence)
            self.weight_model[2].bias.data = torch.tensor([10.0, 0.0])  # Strong bias for w_xyz, no bias for w_f
            
    def forward(self, pos, x, batch):
        # Voxel-based downsampling
        idx = self.voxelsample(pos, batch)
        
        # Include intensity or extra positional info in features
        x = torch.cat([x, pos[:, 3].unsqueeze(1)], dim=1)

        # Compress features and normalize
        compressed_x = self.feature_compressor(x)
        
        # Compute dynamic weights
        weights = self.weight_model(torch.cat([pos[:, :3], compressed_x], dim=1))
    
        w_f = weights[:, 1].unsqueeze(1)  
        w_xyz = weights[:, 0].unsqueeze(1)  

        # Feature scaling to match positional magnitude
        feature_scale = pos[:, :3].norm(dim=1, keepdim=True).mean() / (compressed_x.norm(dim=1, keepdim=True).mean() + 1e-8)

        # Prepare the 6D tensor (position + scaled features)
        scaled_features = compressed_x * feature_scale * w_f
        fused_vector = torch.cat([pos[:, :3], scaled_features], dim=1)

        # Perform kNN search using the 6D tensor
        row, col = knn(fused_vector, fused_vector[idx], k=self.k, batch_x=batch, batch_y=batch[idx])
        return row, col, idx

    def voxelsample(self, pos, batch):
        # Voxel grid sampling for efficiency
        voxel_indices = voxel_grid(pos[:, :3], self.resolution, batch)
        _, remapped_indices = torch.unique(voxel_indices, return_inverse=True)
        _, max_indices = scatter_max(pos[:, 3], remapped_indices, dim=0)
        idx = max_indices[max_indices != -1]
        return idx