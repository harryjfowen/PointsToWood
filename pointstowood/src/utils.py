import torch
from torch import Tensor
from typing import List, Optional, Tuple
import gc
import os

import torch_geometric
from torch_geometric.nn import voxel_grid, knn
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils import scatter
from torch_scatter import scatter_add, scatter_max

from src.point_sampling import VoxelSampling, VoxelSamplingMax, RandomSampling

def configure_threads(num_procs: int) -> int:
    if num_procs is None or num_procs < 1:
        num_procs = os.cpu_count() or 1

    torch.set_num_threads(num_procs)

    try:
        import numba as _nb
        _nb.set_num_threads(num_procs)
    except Exception:
        pass

    os.environ["OMP_NUM_THREADS"] = str(num_procs)
    return num_procs

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def minmax_normalize_reflectance(reflectance_tensor: Tensor) -> Tensor:
    device = reflectance_tensor.device
    
    if torch.isnan(reflectance_tensor).any():
        reflectance_tensor = torch.nan_to_num(reflectance_tensor, nan=0.0)
    
    q1, q3 = torch.quantile(reflectance_tensor, torch.tensor([0.01, 0.99], device=device))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    clipped_reflectance = torch.clamp(reflectance_tensor, lower_bound, upper_bound)
    
    min_val = torch.min(clipped_reflectance)
    max_val = torch.max(clipped_reflectance)
    normalized_reflectance = 2 * (clipped_reflectance - min_val) / (max_val - min_val) - 1
    
    return normalized_reflectance

def quantile_normalize_reflectance(reflectance_tensor: Tensor) -> Tensor:
    if torch.isnan(reflectance_tensor).any():
        raise ValueError("Input reflectance tensor contains NaN values.")
    
    _, indices = torch.sort(reflectance_tensor)
    ranks = torch.argsort(indices)
    
    empirical_quantiles = (ranks.float() + 1) / (len(ranks) + 1)
    empirical_quantiles = torch.clamp(empirical_quantiles, 1e-7, 1 - 1e-7)
    
    normalized_reflectance = torch.erfinv(2 * empirical_quantiles - 1) * torch.sqrt(torch.tensor(2.0)).to(reflectance_tensor.device)
    
    min_val = normalized_reflectance.min()
    max_val = normalized_reflectance.max()
    scaled_reflectance = 2 * (normalized_reflectance - min_val) / (max_val - min_val) - 1
    
    return scaled_reflectance

def downsample_points(pos: Tensor, spacing: float) -> Tensor:
    with torch.no_grad():
        has_reflectance = pos.shape[1] > 3
        has_label = pos.shape[1] > 4

        cluster = voxel_grid(pos[:, :3], spacing)
        cluster, _ = consecutive_cluster(cluster)

        dims = 4 if has_reflectance else 3
        mean_feats = scatter(pos[:, :dims], cluster, dim=0, reduce="mean")
        if not has_reflectance:
            mean_feats = torch.cat([mean_feats, torch.zeros(mean_feats.size(0), 1, device=pos.device)], dim=1)

        if has_label:
            labels = pos[:, 4]
            if labels.dtype != torch.long:
                labels = labels.long()
            approx_mode = scatter(labels.float(), cluster, dim=0, reduce="mean").round().long().float().unsqueeze(1)
            return torch.cat([mean_feats, approx_mode], dim=1)
        else:
            return mean_feats

def downsample_points_max(pos: Tensor, spacing: float) -> Tensor:
    """
    Downsample using voxel_grid + consecutive_cluster by selecting, for each voxel,
    the single point with the maximum reflectance. The representative's XYZ (and
    label if present) are taken from that point.

    Output columns: [x, y, z, reflectance] and optionally label as the last column.
    If reflectance is absent, zeros are used and the first point per voxel is
    selected implicitly by the segmented max.
    """
    with torch.no_grad():
        has_reflectance = pos.shape[1] > 3
        has_label = pos.shape[1] > 4

        # Build voxel clusters
        cluster = voxel_grid(pos[:, :3], spacing)
        cluster, _ = consecutive_cluster(cluster)

        # Prepare reflectance vector for max selection
        if has_reflectance:
            reflectance_values = pos[:, 3]
            reflectance_values = torch.nan_to_num(reflectance_values, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            reflectance_values = torch.zeros(pos.shape[0], device=pos.device)

        # Segmented argmax over reflectance to find representative indices per voxel
        _, representative_indices = scatter_max(reflectance_values, cluster, dim=0)

        # Gather representative features
        selected_xyz = pos[representative_indices, :3]
        if has_reflectance:
            selected_reflectance = pos[representative_indices, 3:4]
        else:
            selected_reflectance = torch.zeros(selected_xyz.size(0), 1, device=pos.device)

        if has_label:
            selected_label = pos[representative_indices, 4].float().unsqueeze(1)
            return torch.cat([selected_xyz, selected_reflectance, selected_label], dim=1)
        else:
            return torch.cat([selected_xyz, selected_reflectance], dim=1)

def create_point_grid(pos: Tensor, grid_sizes: List[float], min_points: int = 512, max_points: int = 9999999) -> List[Tensor]:
    def _collect_voxels(voxelised):
        local = []
        for vx in torch.unique(voxelised):
            voxel = (voxelised == vx).nonzero(as_tuple=True)[0]
            if voxel.size(0) < min_points:
                continue
            if voxel.size(0) > max_points:
                voxel = voxel[torch.randint(0, voxel.size(0), (max_points,))]
            local.append(voxel.to('cpu'))
        return local

    indices_list: List[Tensor] = []

    for size in grid_sizes:
        voxelised = voxel_grid(pos[:, :3], size)
        indices_list += _collect_voxels(voxelised)
            
    return indices_list


def compute_knn_edge_scores(point_cloud: Tensor, k: int = 16) -> Tensor:
    pos = point_cloud[:, :3]
    labels = point_cloud[:, 4]
    batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    
    row, col = knn(pos, pos, k=k, batch_x=batch, batch_y=batch)
    
    neighbor_labels = labels[row]
    wood_ratios = scatter_add(neighbor_labels, col, dim=0, dim_size=pos.shape[0]) / k
    
    edge_scores = 4.0 * wood_ratios * (1.0 - wood_ratios)
    
    return torch.clamp(edge_scores, 0.0, 1.0)


