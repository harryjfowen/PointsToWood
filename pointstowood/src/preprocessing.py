import torch
import glob
import os
from tqdm import tqdm

from src.utils import (
    clear_gpu_memory,
    quantile_normalize_reflectance,
    minmax_normalize_reflectance,
    downsample_points,
    downsample_points_max,
    create_point_grid
)

class Voxelise:
    def __init__(self, pos, vxpath, minpoints=512, maxpoints=9999999, gridsize=[2.0, 4.0], pointspacing=None, overlap: float = 0.0, grid_method: str = 'mean'):
        """
        Initialize the voxelization process.
        
        Args:
            pos (Tensor): Point cloud positions and optional reflectance
            vxpath (str): Output path for voxel files
            minpoints (int): Minimum points required per voxel
            maxpoints (int): Maximum points per voxel
            gridsize (List[float]): List of grid sizes to use
            pointspacing (float, optional): Spacing for downsampling
        """
        self.pos = pos
        self.vxpath = vxpath
        self.minpoints = minpoints
        self.maxpoints = maxpoints
        self.gridsize = gridsize
        self.overlap = overlap
        self.pointspacing = pointspacing
        self.grid_method = grid_method
    
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
        """Process and write voxels to disk."""
        if not isinstance(self.pos, torch.Tensor):
            self.pos = torch.tensor(self.pos.values, dtype=torch.float).to(device='cuda')

        original_pos = self.pos.clone()
        file_counter = len(glob.glob(os.path.join(self.vxpath, 'voxel_*.pt')))

        for grid_size in self.gridsize:
            # Reset to original before per-grid processing
            self.pos = original_pos.clone()

            # Decide spacing: fixed if provided (>0), else adaptive per grid
            spacing = self.pointspacing if (self.pointspacing is not None and self.pointspacing > 0) else (grid_size / 100.0)
            self.pointspacing = spacing
            self.pos = self.downsample()

            reflectance_not_zero = self.pos.shape[1] > 3 and not torch.all(self.pos[:, 3] == 0)
            if reflectance_not_zero:
                self.pos[:, 3] = minmax_normalize_reflectance(self.pos[:, 3])

            # Build voxels for this grid size only
            voxels = create_point_grid(self.pos, [grid_size], min_points=self.minpoints, max_points=self.maxpoints)

            pos_cpu = self.pos.detach().clone().to('cpu')

            for _, voxel_indices in enumerate(tqdm(voxels, desc=f'Writing {grid_size}m voxels')):
                if voxel_indices.size(0) == 0:
                    continue

                # Subsample if too many points (reflectance-weighted if available)
                if voxel_indices.size(0) > self.maxpoints:
                    if reflectance_not_zero:
                        try:
                            voxel_weights = pos_cpu[voxel_indices, 3]
                            voxel_weights = torch.nan_to_num(voxel_weights, nan=0.0, posinf=0.0, neginf=0.0)
                            # Shift to positive range
                            voxel_weights = voxel_weights - voxel_weights.min()
                            voxel_weights = voxel_weights + 1e-8
                            if torch.all(voxel_weights == 0) or torch.any(~torch.isfinite(voxel_weights)):
                                sample_idx = torch.randint(0, voxel_indices.size(0), (self.maxpoints,))
                            else:
                                sample_idx = torch.multinomial(voxel_weights, num_samples=self.maxpoints, replacement=False)
                            voxel_indices = voxel_indices[sample_idx]
                        except Exception:
                            voxel_indices = voxel_indices[torch.randint(0, voxel_indices.size(0), (self.maxpoints,))]
                    else:
                        voxel_indices = voxel_indices[torch.randint(0, voxel_indices.size(0), (self.maxpoints,))]

                voxel = pos_cpu[voxel_indices]
                if voxel.numel() == 0:
                    continue

                # Drop NaN rows
                voxel = voxel[~torch.isnan(voxel).any(dim=1)]
                if voxel.size(0) == 0:
                    continue

                # Ensure voxel still meets minimum points requirement after NaN removal
                if voxel.size(0) < self.minpoints:
                    continue

                torch.save(voxel, os.path.join(self.vxpath, f'voxel_{file_counter}.pt'))
                file_counter += 1

            del voxels, pos_cpu

        del original_pos, self.pos
        clear_gpu_memory()
        return -1

def preprocess(args):
    """Process point cloud data based on command-line arguments."""
    Voxelise(
        args.pc, 
        vxpath=args.vxfile, 
        minpoints=args.min_pts, 
        maxpoints=args.max_pts, 
        pointspacing=args.resolution, 
        gridsize=args.grid_size,
        overlap=getattr(args, 'overlap', 0.0),
        grid_method=getattr(args, 'grid_method', 'mean')
    ).write_voxels()