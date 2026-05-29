import torch
from torch import Tensor
from typing import List, Optional, Tuple, Union
import gc
import os
import numpy as np

import torch_geometric
from torch_geometric.nn import voxel_grid, knn
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils import scatter
from torch_scatter import scatter_add, scatter_max


def preprocess_point_cloud_data(df, zero_reflectance: bool = False, drop_predictions: bool = True):
    """Unified preprocessing for train and predict. Canonical column names, ordering, optional zero reflectance.

    Returns:
        df: DataFrame with columns [x,y,z,reflectance] and optionally label
        headers: list of non-xyz column names (for predict output)
        has_reflectance: bool
    """
    canon_map = {
        'label': ['label'],
        'reflectance': ['reflectance', 'refl', 'intensity'],
    }
    new_columns = {}
    for col in df.columns:
        clean = col.lower().replace('scalar_', '')
        mapped = None
        for target, aliases in canon_map.items():
            if any(alias in clean for alias in aliases):
                mapped = target
                break
        new_columns[col] = mapped if mapped is not None else clean

    df = df.rename(columns=new_columns)

    if 'truth' in df.columns and 'label' in df.columns:
        df = df.drop(columns=['label'])
    if 'label' not in df.columns and 'truth' in df.columns:
        df = df.rename(columns={'truth': 'label'})

    df = df.loc[:, ~df.columns.duplicated()]

    if drop_predictions:
        drop_tokens = ["prediction", "pwood"]
        cols_to_drop = [c for c in df.columns if any(tok in c for tok in drop_tokens)]
        if len(cols_to_drop):
            df = df.drop(columns=cols_to_drop, errors='ignore')

    if 'reflectance' not in df.columns:
        df['reflectance'] = np.zeros(len(df))
        if hasattr(np, 'getLogger'):
            pass  # suppress print in lib
        # print('No reflectance detected, column added with zeros.')
    else:
        pass  # print('Reflectance detected')

    if zero_reflectance:
        df['reflectance'] = np.zeros(len(df))
        # print('Reflectance set to zeros as requested.')

    xyz_cols = ['x', 'y', 'z']
    required_order = xyz_cols + ['reflectance']
    if 'label' in df.columns:
        required_order = required_order + ['label']
    other_cols = [col for col in df.columns if col not in required_order]
    final_cols = required_order + other_cols
    df = df[[c for c in final_cols if c in df.columns]]

    headers = [c for c in df.columns if c not in xyz_cols]
    has_reflectance = (df['reflectance'] != 0).any() if 'reflectance' in df.columns else False
    return df, headers, has_reflectance


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
    """Min-max to [-1, 1]. No clipping — preserves relative differences (model uses relative contrast only)."""
    if torch.isnan(reflectance_tensor).any():
        reflectance_tensor = torch.nan_to_num(reflectance_tensor, nan=0.0)

    min_val = torch.min(reflectance_tensor)
    max_val = torch.max(reflectance_tensor)
    span = max_val - min_val
    if span < 1e-8:
        return torch.zeros_like(reflectance_tensor)
    return 2 * (reflectance_tensor - min_val) / span - 1


def quantile_normalize_reflectance(reflectance_tensor: Tensor) -> Tensor:
    """Rank-transform reflectance to uniform on [-1, 1] via the empirical CDF.

    Sensor-agnostic by construction: only order is preserved, so signed dB
    (RIEGL), 8-bit amplitude, and anything monotone collapse to the same
    uniform marginal. AnisotropicConv consumes only |refl - neighbourhood_median|,
    and under a uniform marginal those local-contrast magnitudes are directly
    comparable across clouds regardless of the source sensor.

    Ties get the average rank (standard quantile normalisation), so saturated
    plateaus (e.g. many points pinned at amplitude 255) don't produce
    spurious micro-contrast within the saturated group.
    """
    if reflectance_tensor.numel() == 0:
        return reflectance_tensor

    original_dtype = reflectance_tensor.dtype
    x = reflectance_tensor.to(torch.float32)

    finite_mask = torch.isfinite(x)
    if not finite_mask.any():
        return torch.zeros_like(x, dtype=original_dtype)
    if not finite_mask.all():
        x = x.clone()
        x[~finite_mask] = torch.median(x[finite_mask])

    if (x.max() - x.min()).abs() < 1e-8:
        return torch.zeros_like(x, dtype=original_dtype)

    # Rank with tie-averaging: for each value, avg of its first and last
    # positions in the sorted array — the standard bisection-based mean rank.
    sorted_vals, sort_idx = torch.sort(x)
    first_eq = torch.searchsorted(sorted_vals, sorted_vals, right=False).to(torch.float32)
    last_eq = (torch.searchsorted(sorted_vals, sorted_vals, right=True) - 1).to(torch.float32)
    avg_ranks_sorted = (first_eq + last_eq) * 0.5

    ranks = torch.empty_like(x)
    ranks[sort_idx] = avg_ranks_sorted

    denom = max(x.numel() - 1, 1)
    normalized = (ranks / denom) * 2.0 - 1.0
    return normalized.to(original_dtype)

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

def _extract_valid_voxels(
    sorted_order: Tensor,
    counts: Tensor,
    boundaries: Tensor,
    min_points: int,
    max_points: Optional[int] = None
) -> List[Tensor]:
    """Extract indices for valid voxels (optimized with batched GPU ops).

    If ``max_points`` is positive, apply an early uniform cap while extracting
    indices. Pass ``None`` or ``<=0`` to keep full voxel membership and defer
    any later sampling policy to the caller.
    """
    # Find valid voxel IDs upfront
    valid_mask = counts >= min_points
    valid_ids = valid_mask.nonzero(as_tuple=True)[0]

    if len(valid_ids) == 0:
        return []

    # Batch fetch boundaries to avoid per-voxel .item() calls
    starts = boundaries[valid_ids]
    ends = boundaries[valid_ids + 1]
    voxel_counts = ends - starts

    # Move to CPU once for iteration (unavoidable for variable-length slicing)
    starts_cpu = starts.cpu().numpy()
    ends_cpu = ends.cpu().numpy()
    counts_cpu = voxel_counts.cpu().numpy()
    sorted_order_cpu = sorted_order.cpu()

    results = []
    for i in range(len(valid_ids)):
        start_idx, end_idx, count = starts_cpu[i], ends_cpu[i], counts_cpu[i]
        voxel_indices = sorted_order_cpu[start_idx:end_idx]

        # Optional early uniform subsample.
        if max_points is not None and max_points > 0 and count > max_points:
            subsample = torch.randperm(count)[:max_points]
            voxel_indices = voxel_indices[subsample]

        results.append(voxel_indices)

    return results


def create_point_grid_with_overlap(
    pos: Tensor,
    grid_size: float,
    min_points: int = 512,
    max_points: Optional[int] = None,
    num_offsets: int = 4
) -> List[Tensor]:
    """Create overlapping voxel grids by shifting grid origin in XY.

    Args:
        pos: Point cloud tensor [N, 3+]
        grid_size: Voxel size in meters
        min_points: Minimum points per voxel
        max_points: Optional early cap applied during extraction. ``None`` keeps
            the full voxel and lets the caller handle any later sampling policy.
        num_offsets: Number of XY offset directions (4 or 8)
            4 = 50% overlap (2x2 grid of origins)
            8 = 4 + cardinal edges for denser coverage

    Returns:
        List of index tensors, one per valid voxel across all offsets
    """
    # Define XY offsets as fractions of grid_size
    if num_offsets == 2:
        xy_offsets = torch.tensor([
            [0.0, 0.0], [0.5, 0.5]
        ])
    elif num_offsets == 4:
        xy_offsets = torch.tensor([
            [0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]
        ])
    elif num_offsets == 8:
        xy_offsets = torch.tensor([
            [0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
            [0.25, 0.0], [0.75, 0.0], [0.0, 0.25], [0.0, 0.75]
        ])
    else:
        xy_offsets = torch.tensor([[0.0, 0.0]])

    # Pre-compute bounds once
    pos_xyz = pos[:, :3]
    pos_min = pos_xyz.min(dim=0).values
    device = pos.device

    # Pre-allocate start tensor (reuse across offsets)
    start = torch.zeros(3, device=device)
    start[2] = pos_min[2]  # Z never changes

    indices_list: List[Tensor] = []

    for offset in xy_offsets:
        # Update only XY components
        start[0] = pos_min[0] - offset[0].item() * grid_size
        start[1] = pos_min[1] - offset[1].item() * grid_size

        # Voxelize with shifted origin
        voxelised = voxel_grid(pos_xyz, grid_size, start=start)
        voxelised, _ = consecutive_cluster(voxelised)

        # Sort once, compute boundaries
        sorted_order = torch.argsort(voxelised)
        counts = torch.bincount(voxelised[sorted_order])
        boundaries = torch.zeros(len(counts) + 1, dtype=torch.long, device=device)
        boundaries[1:] = torch.cumsum(counts, dim=0)

        # Extract valid voxels (only iterates over valid ones)
        indices_list.extend(_extract_valid_voxels(
            sorted_order, counts, boundaries, min_points, max_points
        ))

    return indices_list


def create_point_grid(pos: Tensor, grid_sizes: List[float], min_points: int = 512, max_points: Optional[int] = None) -> List[Tensor]:
    """Efficiently partition points into voxels using sorted indices.

    ``max_points`` is an optional early cap. Pass ``None`` or ``<=0`` to keep
    full voxel membership and defer sampling to a later stage.
    """
    indices_list: List[Tensor] = []
    device = pos.device

    for size in grid_sizes:
        voxelised = voxel_grid(pos[:, :3], size)
        voxelised, _ = consecutive_cluster(voxelised)

        # Sort and compute boundaries
        sorted_order = torch.argsort(voxelised)
        counts = torch.bincount(voxelised[sorted_order])
        boundaries = torch.zeros(len(counts) + 1, dtype=torch.long, device=device)
        boundaries[1:] = torch.cumsum(counts, dim=0)

        # Extract valid voxels (reuses optimized helper)
        indices_list.extend(_extract_valid_voxels(
            sorted_order, counts, boundaries, min_points, max_points
        ))

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
