import torch
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
import glob
import os
from tqdm import tqdm
import torch_scatter

class Voxelise:
    def __init__(self, pos, vxpath, minpoints=512, maxpoints=9999999, gridsize=[1.0,2.0], pointspacing=None):
        self.pos = pos
        self.vxpath = vxpath
        self.minpoints = minpoints
        self.maxpoints = maxpoints
        self.gridsize = gridsize
        self.pointspacing = pointspacing

    def quantile_normalize_reflectance(self):
        reflectance_tensor = self.pos[:, 3].view(-1)
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
    
    def downsample(self):
        if self.pos.shape[1] > 3:
            coords, values = self.pos[:, :3], self.pos[:, 3]
            voxel_indices = voxel_grid(pos=coords, size=self.pointspacing)
            _, remapped_indices = torch.unique(voxel_indices, return_inverse=True)
            _, max_indices = torch_scatter.scatter_max(values, remapped_indices, dim=0)
            valid_max_indices = max_indices[max_indices != -1]
            downsampled_pos = self.pos[valid_max_indices]
        else:
            voxel_indices = voxel_grid(pos=self.pos, size=self.pointspacing)
            _, idx = consecutive_cluster(voxel_indices)
            downsampled_pos = self.pos[idx]
        
        return downsampled_pos

    def grid(self):
        indices_list = []
        for size in self.gridsize:
            voxelised = voxel_grid(self.pos, size)
            for vx in torch.unique(voxelised):
                voxel = (voxelised == vx).nonzero(as_tuple=True)[0]
                if voxel.size(0) < self.minpoints:
                    continue
                indices_list.append(voxel.to('cpu'))
        return indices_list
    
    
    def write_voxels(self):
        
        self.pos = torch.tensor(self.pos.values, dtype=torch.float).to(device='cuda')

        if self.pointspacing:
            print(f'Downsampling to {self.pointspacing}m spacing')
            self.pos = self.downsample()

        reflectance_not_zero = not torch.all(self.pos[:, 3] == 0)
        
        if reflectance_not_zero:
            self.pos[:, 3] = self.quantile_normalize_reflectance()

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
        return -1

def preprocess(args):
    Voxelise(args.pc, vxpath=args.vxfile, minpoints=args.min_pts, maxpoints=args.max_pts, pointspacing=args.resolution, gridsize = args.grid_size).write_voxels()



