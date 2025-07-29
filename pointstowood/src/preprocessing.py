import torch
import glob
import os
from tqdm import tqdm

from src.utils import (
    clear_gpu_memory,
    quantile_normalize_reflectance,
    minmax_normalize_reflectance,
    downsample_points,
    create_point_grid
)

class Voxelise:
    def __init__(self, pos, vxpath, minpoints=512, maxpoints=9999999, gridsize=[2.0, 4.0], pointspacing=None, overlap: float = 0.0):
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
    
    def downsample(self):
        """Downsample point cloud to specified spacing."""
        return downsample_points(self.pos, self.pointspacing)
    
    def grid(self):
        """Create grid of voxels from point cloud."""
        return create_point_grid(
            self.pos,
            self.gridsize,
            min_points=self.minpoints,
            max_points=self.maxpoints,
            overlap=self.overlap,
        )
    
    def write_voxels(self):
        """Process and write voxels to disk."""
        if not isinstance(self.pos, torch.Tensor):
            self.pos = torch.tensor(self.pos.values, dtype=torch.float).to(device='cuda')

        if self.pointspacing:
            self.pos = self.downsample()

        reflectance_not_zero = self.pos.shape[1] > 3 and not torch.all(self.pos[:, 3] == 0)
        
        if reflectance_not_zero:
            self.pos[:, 3] = minmax_normalize_reflectance(self.pos[:, 3])

        voxels = self.grid()

        if reflectance_not_zero:
            weight = self.pos[:, 3] - self.pos[:, 3].min()
            mask = ~(torch.isnan(weight) | torch.isinf(weight))
            self.pos, weight = self.pos[mask], weight[mask]
            if weight.sum() == 0:
                raise ValueError("All weights are invalid. Check the reflectance values.")
            weight = weight + 1e-8
            weight = weight.detach().to('cpu')
        else:
            weight = None

        self.pos = self.pos.detach().clone().to('cpu')
        
        file_counter = len(glob.glob(os.path.join(self.vxpath, 'voxel_*.pt')))

        for _, voxel_indices in enumerate(tqdm(voxels, desc='Writing voxels')):
            if voxel_indices.size(0) == 0:
                continue  

            if voxel_indices.size(0) > self.maxpoints:
                if reflectance_not_zero:
                    voxel_indices = voxel_indices[torch.multinomial(weight[voxel_indices], self.maxpoints)]
                else:
                    voxel_indices = voxel_indices[torch.randint(0, voxel_indices.size(0), (self.maxpoints,))]
            
            voxel = self.pos[voxel_indices]
            voxel = voxel[~torch.isnan(voxel).any(dim=1)]
            
            torch.save(voxel, os.path.join(self.vxpath, f'voxel_{file_counter}.pt'))
            file_counter += 1
        
        del voxel, voxel_indices, weight, self.pos
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
        overlap=args.overlap
    ).write_voxels()
