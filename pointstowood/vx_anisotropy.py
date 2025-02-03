import torch

from torch_scatter import scatter_mean
import torch.backends.cuda
from src.io import load_file, save_file
from torch_geometric.nn import voxel_grid
 

def compute_voxel_anisotropy_scatter(points, voxel_size=0.1, min_points=3):
    # Set preferred backend
    torch.backends.cuda.preferred_linalg_library("cusolver")
    
    voxel_idx = voxel_grid(points, size=voxel_size)
    
    # Make indices consecutive
    unique_voxels, inverse_indices = torch.unique(voxel_idx, return_inverse=True)
    
    # Count points per voxel
    counts = torch.bincount(inverse_indices)
    valid_mask = counts >= min_points
    
    # Create default anisotropy (0.5)
    anisotropy = torch.full_like(voxel_idx, 0.5, dtype=torch.float32)
    
    if not valid_mask.any():
        return anisotropy
    
    # Compute voxel centers and centered points
    voxel_centers = scatter_mean(points, inverse_indices, dim=0)
    centered_points = points - voxel_centers[inverse_indices]
    
    # Compute covariance with additional checks
    cov_components = centered_points.unsqueeze(2) * centered_points.unsqueeze(1)
    covariance = scatter_mean(cov_components.view(-1, 9), inverse_indices, dim=0)
    covariance = covariance.view(-1, 3, 3)
    
    # Add small diagonal term for numerical stability
    covariance = covariance + torch.eye(3, device=covariance.device) * 1e-6
    
    # Check for NaN/Inf values
    valid_cov = ~torch.isnan(covariance).any(dim=(1,2)) & ~torch.isinf(covariance).any(dim=(1,2))
    valid_mask = valid_mask & valid_cov
    
    if not valid_mask.any():
        return anisotropy
    
    try:
        # Compute eigenvalues only for valid voxels
        valid_covariance = covariance[valid_mask]
        eigenvalues = torch.linalg.eigvalsh(valid_covariance)
        eigenvalues, _ = torch.sort(eigenvalues, dim=1, descending=True)
        
        # Compute anisotropy for valid voxels
        valid_anisotropy = (eigenvalues[:, 0] - eigenvalues[:, 2]) / (eigenvalues[:, 0] + 1e-8)
        
        # Create mapping back to original indices
        valid_idx = torch.where(valid_mask)[0]
        idx_map = torch.full((len(valid_mask),), -1, device=points.device)
        idx_map[valid_idx] = torch.arange(len(valid_idx), device=points.device)
        
        # Map anisotropy back to points
        valid_points_mask = idx_map[inverse_indices] >= 0
        anisotropy[valid_points_mask] = valid_anisotropy[idx_map[inverse_indices[valid_points_mask]]]
        
    except RuntimeError as e:
        print(f"Warning: Error in eigenvalue computation: {e}")
        return anisotropy
    
    return anisotropy


pc_data, headers = load_file(filename='/home/harryowen/Desktop/pol21.ply', additional_headers=True, verbose=True)
points = torch.tensor(pc_data[['x','y','z']].values, dtype=torch.float32).to('cuda')
anisotropy = compute_voxel_anisotropy_scatter(points, voxel_size=0.15, min_points=16)
pc_data['anisotropy'] = anisotropy.cpu().numpy()
save_file('/home/harryowen/Desktop/anisotropy.ply', pc_data, additional_fields=['anisotropy'])
