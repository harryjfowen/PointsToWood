import torch
import glob
import os
import numpy as np
from tqdm import tqdm

from src.utils import (
    clear_gpu_memory,
    quantile_normalize_reflectance,
    downsample_points,
    downsample_points_max,
    create_point_grid,
    create_point_grid_with_overlap,
)
from src.memory_utils import should_use_cpu_preprocessing


def _sor_voxel(voxel_xyz_np, k=10, std_mult=1.0):
    """Per-voxel SOR on small point set (fast: O(n log n) with n ≤ maxpoints). Returns bool mask (keep inliers)."""
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return np.ones(voxel_xyz_np.shape[0], dtype=bool)
    n = voxel_xyz_np.shape[0]
    if n < k + 2:
        return np.ones(n, dtype=bool)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree', n_jobs=1).fit(voxel_xyz_np)
    dists, _ = nn.kneighbors(voxel_xyz_np)
    mean_d = np.mean(dists[:, 1:], axis=1)
    thresh = np.mean(mean_d) + std_mult * (np.std(mean_d) + 1e-8)
    return mean_d <= thresh


class Voxelise:
    def __init__(self, pos, vxpath, minpoints=512, maxpoints=9999999, gridsize=[2.0, 4.0], pointspacing=None, overlap: float = 0.0, grid_method: str = 'mean', sor=False, sor_k=10, sor_std=1.0, file_prefix: str = None):
        """
        Initialize the voxelization process.

        Args:
            pos (Tensor): Point cloud positions and optional reflectance
            vxpath (str): Output path for voxel files
            minpoints (int): Minimum points required per voxel
            maxpoints (int): Maximum points per voxel
            gridsize (List[float]): List of grid sizes to use
            pointspacing (float, optional): Spacing for downsampling
            sor (bool): If True, run per-voxel SOR before writing (fast, local to each voxel)
            sor_k (int): SOR neighbours (default 10)
            sor_std (float): SOR threshold = mean + sor_std*std (default 1.0)
            file_prefix (str, optional): Prefix for output files (e.g., source filename)
        """
        self.pos = pos
        self.vxpath = vxpath
        self.minpoints = minpoints
        self.maxpoints = maxpoints
        self.gridsize = gridsize
        self.overlap = overlap
        self.pointspacing = pointspacing
        self.grid_method = grid_method
        self.sor = sor
        self.sor_k = sor_k
        self.sor_std = sor_std
        self.file_prefix = file_prefix
    
    def downsample(self):
        """Downsample point cloud to specified spacing."""
        if self.grid_method == 'max':
            return downsample_points_max(self.pos, self.pointspacing)
        return downsample_points(self.pos, self.pointspacing)
    
    def grid(self):
        """Create grid of voxels from point cloud."""
        return create_point_grid(
            self.pos,
            self.gridsize,
            min_points=self.minpoints,
            max_points=self.maxpoints,
        )
    
    def write_voxels(self):
        """Process and write voxels to disk.

        Two modes:
        - overlap > 0: Single grid size with offset origins for smooth edge coverage
        - overlap == 0: Multi-resolution grids (original behavior)
        """
        if not isinstance(self.pos, torch.Tensor):
            # Adaptive device selection based on memory requirements
            point_count = len(self.pos)
            has_reflectance = self.pos.shape[1] > 3

            use_cpu, estimated_gpu_mem, available_gpu_mem = should_use_cpu_preprocessing(
                point_count, has_reflectance, self.gridsize
            )

            if use_cpu:
                device = 'cpu'
                print(f"Large point cloud ({point_count:,} points, ~{estimated_gpu_mem:.1f}GB) - using CPU for preprocessing")
            else:
                device = 'cuda'
                print(f"Point cloud fits in GPU memory (~{estimated_gpu_mem:.1f}GB) - using GPU for preprocessing")

            self.pos = torch.tensor(self.pos.values, dtype=torch.float).to(device=device)

        if self.file_prefix:
            file_counter = len(glob.glob(os.path.join(self.vxpath, f'{self.file_prefix}_voxel_*.pt')))
        else:
            file_counter = len(glob.glob(os.path.join(self.vxpath, 'voxel_*.pt')))

        # Use overlapping grids mode if overlap > 0
        if self.overlap > 0:
            file_counter = self._write_voxels_with_overlap(file_counter)
        else:
            file_counter = self._write_voxels_multi_resolution(file_counter)

        clear_gpu_memory()
        return file_counter

    def _write_voxels_with_overlap(self, file_counter: int) -> int:
        """Write voxels using offset grid origins for edge coverage.

        Uses a single grid size with 4 or 8 XY offset origins to create
        overlapping blocks. More efficient than multi-resolution and
        ensures edge points are covered by multiple blocks.
        """
        # Use first grid size (or median if multiple specified)
        grid_size = self.gridsize[0] if len(self.gridsize) == 1 else sorted(self.gridsize)[len(self.gridsize) // 2]

        # overlap is now an integer: 4 or 8
        num_offsets = int(self.overlap)

        # Normalize reflectance once before downsampling
        reflectance_not_zero = self.pos.shape[1] > 3 and not torch.all(self.pos[:, 3] == 0)
        if reflectance_not_zero:
            self.pos[:, 3] = quantile_normalize_reflectance(self.pos[:, 3])

        # Downsample once
        spacing = self.pointspacing if (self.pointspacing is not None and self.pointspacing > 0) else (grid_size / 100.0)
        self.pointspacing = spacing
        self.pos = self.downsample()

        # Create overlapping voxels
        voxels = create_point_grid_with_overlap(
            self.pos,
            grid_size,
            min_points=self.minpoints,
            max_points=self.maxpoints,
            num_offsets=num_offsets
        )

        print(f"Overlapping grids: {grid_size}m with {num_offsets} offsets -> {len(voxels)} voxels")

        # Move to CPU for writing
        if self.pos.device.type == 'cpu':
            pos_cpu = self.pos.detach()
        else:
            pos_cpu = self.pos.detach().clone().to('cpu')

        file_counter = self._write_voxel_list(voxels, pos_cpu, reflectance_not_zero, file_counter, f'{grid_size}m overlap', grid_size=grid_size)

        del voxels, pos_cpu, self.pos
        return file_counter

    def _write_voxels_multi_resolution(self, file_counter: int) -> int:
        """Write voxels using multiple grid resolutions (original behavior)."""
        original_pos = self.pos.clone()

        # Normalize reflectance once upfront on full cloud (before downsampling for any grid size)
        reflectance_not_zero = original_pos.shape[1] > 3 and not torch.all(original_pos[:, 3] == 0)
        if reflectance_not_zero:
            original_pos[:, 3] = quantile_normalize_reflectance(original_pos[:, 3])

        for grid_size in self.gridsize:
            # Reset to normalized original before per-grid processing
            self.pos = original_pos.clone()

            # Spacing: resolution>0 = fixed (same for all grid sizes); 0 = adaptive (spacing = grid_size/100 so points-per-voxel scale is consistent)
            spacing = self.pointspacing if (self.pointspacing is not None and self.pointspacing > 0) else (grid_size / 100.0)
            self.pointspacing = spacing
            self.pos = self.downsample()

            # Build voxels for this grid size only
            voxels = create_point_grid(self.pos, [grid_size], min_points=self.minpoints, max_points=self.maxpoints)

            # Only move to CPU if not already there
            if self.pos.device.type == 'cpu':
                pos_cpu = self.pos.detach()
            else:
                pos_cpu = self.pos.detach().clone().to('cpu')

            file_counter = self._write_voxel_list(voxels, pos_cpu, reflectance_not_zero, file_counter, f'{grid_size}m', grid_size=grid_size)

            del voxels, pos_cpu

        del original_pos, self.pos
        return file_counter

    def _write_voxel_list(self, voxels, pos_cpu, reflectance_not_zero: bool, file_counter: int, desc: str, grid_size: float = None) -> int:
        """Write a list of voxels to disk. If grid_size is set, save as dict for resolution-aware batching."""
        for voxel_indices in tqdm(voxels, desc=f'Writing {desc} voxels'):
            if voxel_indices.size(0) == 0:
                continue

            voxel = pos_cpu[voxel_indices]
            if voxel.numel() == 0:
                continue

            # Drop NaN rows
            voxel = voxel[~torch.isnan(voxel).any(dim=1)]
            if voxel.size(0) < self.minpoints:
                continue

            # Per-voxel SOR before subsampling (denoise full voxel, then cap)
            if self.sor:
                mask = _sor_voxel(voxel[:, :3].numpy(), k=self.sor_k, std_mult=self.sor_std)
                voxel = voxel[torch.from_numpy(mask)]
                if voxel.size(0) < self.minpoints:
                    continue

            # Subsample if still over maxpoints (reflectance-weighted if available)
            if voxel.size(0) > self.maxpoints:
                if reflectance_not_zero:
                    try:
                        voxel_weights = voxel[:, 3]
                        voxel_weights = torch.nan_to_num(voxel_weights, nan=0.0, posinf=0.0, neginf=0.0)
                        voxel_weights = voxel_weights - voxel_weights.min() + 1e-8
                        if torch.all(voxel_weights == 0) or torch.any(~torch.isfinite(voxel_weights)):
                            sample_idx = torch.randint(0, voxel.size(0), (self.maxpoints,))
                        else:
                            sample_idx = torch.multinomial(voxel_weights, num_samples=self.maxpoints, replacement=False)
                        voxel = voxel[sample_idx]
                    except Exception:
                        voxel = voxel[torch.randint(0, voxel.size(0), (self.maxpoints,))]
                else:
                    voxel = voxel[torch.randint(0, voxel.size(0), (self.maxpoints,))]

            if self.file_prefix:
                filename = f'{self.file_prefix}_voxel_{file_counter}.pt'
            else:
                filename = f'voxel_{file_counter}.pt'
            to_save = {'point_cloud': voxel, 'grid_size': float(grid_size)} if grid_size is not None else voxel
            torch.save(to_save, os.path.join(self.vxpath, filename))
            file_counter += 1

        return file_counter

def preprocess(args):
    """Process point cloud data based on command-line arguments."""
    maxpoints = args.max_pts if args.max_pts > 0 else 9999999  # 0 = no subsampling
    Voxelise(
        args.pc,
        vxpath=args.vxfile,
        minpoints=args.min_pts,
        maxpoints=maxpoints,
        pointspacing=args.resolution,
        gridsize=args.grid_size,
        overlap=getattr(args, 'overlap', 0.0),
        grid_method=getattr(args, 'grid_method', 'max'),
        sor=getattr(args, 'sor', False),
        sor_k=getattr(args, 'sor_k', 10),
        sor_std=getattr(args, 'sor_std', 1.0),
        file_prefix=getattr(args, 'eval_source_file', None),
    ).write_voxels()