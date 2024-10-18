import torch
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
import glob
from src.io import load_file, save_file
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import CSF
from scipy.interpolate import griddata
import torch_scatter

class Voxelise:
    def __init__(self, pos, vxpath, minpoints=512, maxpoints=16384, gridsize=[2.0,4.0], pointspacing=0.01):
        self.pos = pos
        self.vxpath = vxpath
        self.minpoints = minpoints
        self.maxpoints = maxpoints
        self.gridsize = gridsize
        self.pointspacing = min(self.gridsize) / 100.0#pointspacing

    def quantile_normalize_reflectance(self):
        reflectance_tensor = self.pos[:, 3].view(-1)
        _, indices = torch.sort(reflectance_tensor)
        ranks = torch.argsort(indices)
        empirical_quantiles = (ranks.float() + 1) / (len(ranks) + 1)
        normalized_reflectance = torch.erfinv(2 * empirical_quantiles - 1) * torch.sqrt(torch.tensor(2.0)).to(reflectance_tensor.device)
        min_val = normalized_reflectance.min()
        max_val = normalized_reflectance.max()
        scaled_reflectance = 2 * (normalized_reflectance - min_val) / (max_val - min_val) - 1
        return scaled_reflectance
    
    def downsample(self):
        voxelised = voxel_grid(self.pos, self.pointspacing)
        _, idx = consecutive_cluster(voxelised)
        return self.pos[idx]

    def gpu_ground(self):
        x, y, z = self.pos[:, 0], self.pos[:, 1], self.pos[:, 2]
        grid_resolution = 5.0
        x_min, x_max = torch.min(x), torch.max(x)
        y_min, y_max = torch.min(y), torch.max(y)
        x_bins = torch.arange(x_min, x_max + grid_resolution, grid_resolution, device='cuda')
        y_bins = torch.arange(y_min, y_max + grid_resolution, grid_resolution, device='cuda')
        x_indices = torch.bucketize(x, x_bins)
        y_indices = torch.bucketize(y, y_bins)
        grid_indices = x_indices * len(y_bins) + y_indices
        _, grid_inverse_indices = torch.unique(grid_indices, return_inverse=True)
        min_z_values = torch_scatter.scatter_min(z, grid_inverse_indices)[0]
        min_z_per_point = min_z_values[grid_inverse_indices]
        global_max_z = torch.max(z)
        normalized_z = (z - min_z_per_point) / (global_max_z - min_z_per_point + 1e-8)
        normalized_z = normalized_z.view(-1, 1) 
        self.pos = torch.cat((self.pos, normalized_z), dim=1)
        return self.pos
        
    def grid(self):
        indices_list = []
        for size in self.gridsize:
            voxelised = voxel_grid(self.pos, size).to('cpu')
            for vx in torch.unique(voxelised):
                voxel = (voxelised == vx).nonzero(as_tuple=True)[0]
                if voxel.size(0) < self.minpoints:
                    continue
                indices_list.append(voxel)
        return indices_list
    
    def write_voxels(self):
        
        self.pos = torch.tensor(self.pos.values, dtype=torch.float).to(device='cuda')

        reflectance_not_zero = not torch.all(self.pos[:, 3] == 0)

        if reflectance_not_zero:
            self.pos[:, 3] = self.quantile_normalize_reflectance()

        #self.pos = self.downsample()

        print('Height Normalising Point Cloud')
        self.pos = self.gpu_ground()

        voxels = self.grid()
        
        if reflectance_not_zero:
            weight = self.pos[:, 3] - self.pos[:, 3].min()
            mask = ~(torch.isnan(weight) | torch.isinf(weight) | (weight < 0))
            self.pos, weight = self.pos[mask], weight[mask]
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
        return (self.pos[:,-1])

def preprocess(args):
    n_z = Voxelise(args.pc, vxpath=args.vxfile, minpoints=args.min_pts, maxpoints=args.max_pts, pointspacing=args.resolution, gridsize = args.grid_size).write_voxels()
    args.pc['n_z'] = n_z.detach().numpy()



