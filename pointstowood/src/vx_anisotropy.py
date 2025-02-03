import torch
from torch_scatter import scatter_mean as scatter
from torch_geometric.nn import voxel_grid
from src.io import load_file, save_file

def compute_voxel_anisotropy_scatter(points, voxel_size=0.1, min_points=64):
    voxel_idx = voxel_grid(points, size=voxel_size)
    # Compute voxel centers and centered points
    voxel_centers = scatter(points, voxel_idx, dim=0, reduce='mean')
    centered_points = points - voxel_centers[voxel_idx]
    # Compute covariance
    cov_components = centered_points.unsqueeze(2) * centered_points.unsqueeze(1)
    covariance = scatter(cov_components.view(-1, 9), voxel_idx, dim=0, reduce='mean')
    covariance = covariance.view(-1, 3, 3)
    # Count points per voxel
    unique_voxels, counts = torch.unique(voxel_idx, return_counts=True)
    # Compute eigenvalues for all voxels
    eigenvalues = torch.linalg.eigvalsh(covariance)
    eigenvalues, _ = torch.sort(eigenvalues, dim=1, descending=True)
    # Compute anisotropy for all voxels
    voxel_anisotropy = (eigenvalues[:, 0] - eigenvalues[:, 2]) / (eigenvalues[:, 0] + 1e-8)
    # Create default anisotropy (0.5) and update valid voxels
    anisotropy = torch.full_like(voxel_idx, 0.5, dtype=torch.float32)
    valid_mask = counts >= min_points
    # Map anisotropy back to points without loop
    anisotropy = torch.where(
        valid_mask[voxel_idx],
        voxel_anisotropy[voxel_idx],
        anisotropy
    )
    return anisotropy

pc_data, headers = load_file(filename='/home/harryowen/Desktop/test.ply', additional_headers=True, verbose=True)
points = torch.tensor(pc_data[:, :3], dtype=torch.float32).to('cuda')
anisotropy = compute_voxel_anisotropy_scatter(points, voxel_size=0.1, min_points=64)
points = torch.cat([points, anisotropy.view(-1, 1)], dim=1).cpu().numpy()
save_file('output_clustered.ply', pc_data, additional_fields=['cluster_label'])