import torch
from torch import Tensor
from typing import List, Optional, Tuple
import gc
import os

import torch_geometric
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils import scatter
from torch_scatter import scatter_add, scatter_max

from src.point_sampling import VoxelSampling, VoxelSamplingMax, RandomSampling

def configure_threads(num_procs: int) -> int:
    """Set the maximum number of CPU threads used by Torch, Numba and OpenMP.

    Parameters
    ----------
    num_procs : int
        Positive integer – explicit thread count; -1 (or <0) uses all
        available CPU cores.

    Returns
    -------
    int
        The thread count that was actually set (useful for logging).
    """

    if num_procs is None or num_procs < 1:
        num_procs = os.cpu_count() or 1

    # PyTorch intra-op threads
    torch.set_num_threads(num_procs)

    # Numba – optional
    try:
        import numba as _nb
        _nb.set_num_threads(num_procs)
    except Exception:
        pass

    # OpenMP / other C libraries that honour OMP_NUM_THREADS
    os.environ["OMP_NUM_THREADS"] = str(num_procs)

    return num_procs

def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    torch.cuda.empty_cache()

def minmax_normalize_reflectance(reflectance_tensor: Tensor) -> Tensor:
    """Normalize reflectance values using min-max normalization with outlier clipping."""
    device = reflectance_tensor.device
    
    # Check for NaN values
    if torch.isnan(reflectance_tensor).any():
        reflectance_tensor = torch.nan_to_num(reflectance_tensor, nan=0.0)
    
    # Clip outliers using percentiles
    q1, q3 = torch.quantile(reflectance_tensor, torch.tensor([0.01, 0.99], device=device))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Clip values
    clipped_reflectance = torch.clamp(reflectance_tensor, lower_bound, upper_bound)
    
    # Min-max normalization to [-1, 1]
    min_val = torch.min(clipped_reflectance)
    max_val = torch.max(clipped_reflectance)
    normalized_reflectance = 2 * (clipped_reflectance - min_val) / (max_val - min_val) - 1
    
    return normalized_reflectance

def quantile_normalize_reflectance(reflectance_tensor: Tensor) -> Tensor:
    """
    Normalize reflectance values using quantile normalization.
    
    Args:
        reflectance_tensor (Tensor): Input reflectance values
        
    Returns:
        Tensor: Normalized reflectance values in range [-1, 1]
    """
    if torch.isnan(reflectance_tensor).any():
        raise ValueError("Input reflectance tensor contains NaN values.")
    
    # Get ranks for each value
    _, indices = torch.sort(reflectance_tensor)
    ranks = torch.argsort(indices)
    
    # Convert ranks to empirical quantiles
    empirical_quantiles = (ranks.float() + 1) / (len(ranks) + 1)
    empirical_quantiles = torch.clamp(empirical_quantiles, 1e-7, 1 - 1e-7)
    
    # Transform to standard normal
    normalized_reflectance = torch.erfinv(2 * empirical_quantiles - 1) * torch.sqrt(torch.tensor(2.0)).to(reflectance_tensor.device)
    
    # Scale to [-1, 1]
    min_val = normalized_reflectance.min()
    max_val = normalized_reflectance.max()
    scaled_reflectance = 2 * (normalized_reflectance - min_val) / (max_val - min_val) - 1
    
    return scaled_reflectance

def downsample_points(pos: Tensor, spacing: float) -> Tensor:
    """
    Downsample point cloud using voxel grid.
    Uses max reflectance sampling if reflectance values are available,
    otherwise uses standard mean-based voxel sampling.
    
    Args:
        pos (Tensor): Point positions with optional reflectance and label [N, 3+]
        spacing (float): Voxel spacing for downsampling
        
    Returns:
        Tensor: Downsampled point cloud
    """
    with torch.no_grad():
        use_reflectance = pos.shape[1] > 3 and pos[:, 3].sum() != 0
        has_label = pos.shape[1] > 4  # Check if there's a label column
        
        try:
            if use_reflectance:
                sampler = VoxelSamplingMax()
                reflectance = pos[:, 3]
                
                # Extract and pass label if it exists
                y = None
                if has_label:
                    y = pos[:, 4]
                    # Convert to long to avoid one_hot encoding issues
                    if y.dtype != torch.long:
                        y = y.long()
                
                # Call sampler with appropriate parameters
                if has_label:
                    results = sampler(
                        pos=pos[:, :3], 
                        reflectance=reflectance, 
                        size=spacing,
                        y=y
                    )
                    # Unpack the results - 4 values returned when y is provided
                    pos_out, reflectance_out, batch_out, y_out = results
                    
                    # Combine all columns
                    result = torch.cat([pos_out, reflectance_out.view(-1, 1), y_out.view(-1, 1)], dim=1)
                    return result
                    
                else:
                    # No y provided, only 3 values returned
                    pos_out, reflectance_out, batch_out = sampler(
                        pos=pos[:, :3], 
                        reflectance=reflectance, 
                        size=spacing
                    )
                    
                    # Combine only position and reflectance
                    result = torch.cat([pos_out, reflectance_out.view(-1, 1)], dim=1)
                    return result
                    
            else:
                sampler = VoxelSampling()
                
                # Extract and pass label if it exists
                y = None
                if has_label:
                    y = pos[:, 4] 
                    # Convert to long to avoid one_hot encoding issues
                    if y.dtype != torch.long:
                        y = y.long()
                
                # Call sampler with appropriate parameters
                if has_label:
                    results = sampler(pos=pos[:, :3], size=spacing, y=y)
                    
                    # Unpack the results - 4 values returned when y is provided
                    if len(results) == 4:
                        pos_out, _, batch_out, y_out = results
                        
                        # Create a zero reflectance column since reflectance is not being used
                        reflectance_out = torch.zeros(pos_out.shape[0], dtype=torch.float, device=pos_out.device)
                        
                        # Combine position, reflectance (zeros), and label
                        result = torch.cat([pos_out, reflectance_out.view(-1, 1), y_out.view(-1, 1)], dim=1)
                        return result
                else:
                    # No y provided, only 3 values returned
                    pos_out, _, batch_out = sampler(pos=pos[:, :3], size=spacing)
                    
                    # Create a zero reflectance column since reflectance is not being used
                    reflectance_out = torch.zeros(pos_out.shape[0], dtype=torch.float, device=pos_out.device)
                    
                    # Combine position and reflectance (zeros)
                    result = torch.cat([pos_out, reflectance_out.view(-1, 1)], dim=1)
                    return result
                
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                torch.cuda.empty_cache()
                
                # Move to CPU and retry
                device = pos.device
                pos_cpu = pos.cpu()
                
                if use_reflectance:
                    sampler = VoxelSamplingMax()
                    reflectance = pos_cpu[:, 3]
                    
                    # Extract and pass label if it exists
                    y = None
                    if has_label:
                        y = pos_cpu[:, 4]
                        # Convert to long to avoid one_hot encoding issues
                        if y.dtype != torch.long:
                            y = y.long()
                    
                    # Call sampler with appropriate parameters
                    if has_label:
                        results = sampler(
                            pos=pos_cpu[:, :3], 
                            reflectance=reflectance, 
                            size=spacing,
                            y=y
                        )
                        # Unpack the results - 4 values returned when y is provided
                        pos_out, reflectance_out, batch_out, y_out = results
                        
                        # Combine all columns
                        result = torch.cat([pos_out, reflectance_out.view(-1, 1), y_out.view(-1, 1)], dim=1)
                        return result.to(device)
                        
                    else:
                        # No y provided, only 3 values returned
                        pos_out, reflectance_out, batch_out = sampler(
                            pos=pos_cpu[:, :3], 
                            reflectance=reflectance, 
                            size=spacing
                        )
                        
                        # Combine only position and reflectance
                        result = torch.cat([pos_out, reflectance_out.view(-1, 1)], dim=1)
                        return result.to(device)
                else:
                    sampler = VoxelSampling()
                    
                    # Extract and pass label if it exists
                    y = None
                    if has_label:
                        y = pos_cpu[:, 4] if use_reflectance else pos_cpu[:, 3]
                        # Convert to long to avoid one_hot encoding issues
                        if y.dtype != torch.long:
                            y = y.long()
                    
                    # Call sampler with appropriate parameters
                    if has_label:
                        results = sampler(pos=pos_cpu[:, :3], size=spacing, y=y)
                        
                        # Unpack the results - 4 values returned when y is provided
                        if len(results) == 4:
                            pos_out, _, batch_out, y_out = results
                            
                            # Create a zero reflectance column since reflectance is not being used
                            reflectance_out = torch.zeros(pos_out.shape[0], dtype=torch.float, device=pos_out.device)
                            
                            # Combine position, reflectance (zeros), and label
                            result = torch.cat([pos_out, reflectance_out.view(-1, 1), y_out.view(-1, 1)], dim=1)
                            return result.to(device)
                    else:
                        # No y provided, only 3 values returned
                        pos_out, _, batch_out = sampler(pos=pos_cpu[:, :3], size=spacing)
                        return pos_out.to(device)
            else:
                raise e

def create_point_grid(
    pos: Tensor,
    grid_sizes: List[float],
    min_points: int = 512,
    max_points: int = 9999999,
    overlap: float = 0.0,
) -> List[Tensor]:
    """Voxelise a point cloud (optionally with overlapping grids).

    Parameters
    ----------
    pos : Tensor [N, 3+]
        XYZ(+extras) point cloud tensor.
    grid_sizes : List[float]
        Edge length(s) of the voxels.
    min_points, max_points : int
        Voxel size filtering thresholds.
    overlap : float, optional
        Fraction of *grid_size* by which a **second** grid is shifted.
        0.0 → original behaviour (no overlap). 0.5 → 50 % overlap.

    Returns
    -------
    List[Tensor]
        A list of index tensors, one per voxel (both grids).
    """

    assert 0.0 <= overlap < 1.0, "overlap must be in [0,1)."

    def _collect_voxels(voxelised):
        local = []
        for vx in torch.unique(voxelised):
            voxel = (voxelised == vx).nonzero(as_tuple=True)[0]
            if voxel.size(0) < min_points:
                continue
            if voxel.size(0) > max_points:
                if pos.shape[1] > 3 and not torch.all(pos[:, 3] == 0):
                    weight = pos[voxel, 3] - pos[voxel, 3].min() + 1e-8
                    voxel = voxel[torch.multinomial(weight, max_points)]
                else:
                    voxel = voxel[torch.randint(0, voxel.size(0), (max_points,))]
            local.append(voxel.to('cpu'))
        return local

    indices_list: List[Tensor] = []

    for size in grid_sizes:
        # Base grid
        voxelised = voxel_grid(pos[:, :3], size)
        indices_list += _collect_voxels(voxelised)

        # Overlapping grid (shifted)
        if overlap > 0.0:
            shift_val = torch.as_tensor(size, device=pos.device) * overlap
            shifted_pos = pos[:, :3] + shift_val
            voxelised_shift = voxel_grid(shifted_pos, size)
            indices_list += _collect_voxels(voxelised_shift)
            
    return indices_list

