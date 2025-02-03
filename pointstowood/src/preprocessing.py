import torch
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
import glob
import os
from tqdm import tqdm
import torch_scatter

class Voxelise:
    def __init__(self, pos, vxpath, minpoints=512, maxpoints=16384, gridsize=[2.0,4.0], pointspacing=None):
        self.pos = pos
        self.vxpath = vxpath
        self.minpoints = minpoints
        self.maxpoints = maxpoints
        self.gridsize = gridsize
        self.pointspacing = None

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
        # -1 to 1
        scaled_reflectance = 2 * (normalized_reflectance - min_val) / (max_val - min_val) - 1
        # 0-1
        # scaled_reflectance = (normalized_reflectance - min_val) / (max_val - min_val)  # Scale to [0, 1]
        return scaled_reflectance
    
    # def downsample(self):
    #     voxelised = voxel_grid(self.pos, self.pointspacing)
    #     _, idx = consecutive_cluster(voxelised)
    #     return self.pos[idx]
    
    def downsample(self):
        coords, values = self.pos[:, :3], self.pos[:, 3]
        voxel_indices = voxel_grid(pos=coords, size=self.pointspacing)
        _, remapped_indices = torch.unique(voxel_indices, return_inverse=True)
        _, max_indices = torch_scatter.scatter_max(values, remapped_indices, dim=0)
        valid_max_indices = max_indices[max_indices != -1]
        downsampled_pos = self.pos[valid_max_indices]
        return downsampled_pos

    def gpu_ground(self):
        self.pos = self.pos.contiguous()
        x, y, z = self.pos[:, 0].contiguous(), self.pos[:, 1].contiguous(), self.pos[:, 2].contiguous()
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
        normalized_z = z - min_z_per_point  # Preserve units in meters
        self.pos = torch.cat((self.pos, normalized_z.view(-1, 1)), dim=1)
        return normalized_z.view(-1, 1).cpu()#self.pos
        
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
    
    def reflectance2intensity(self):
        reflectance_tensor = self.pos[:, 3]
        if torch.isnan(reflectance_tensor).any():
            raise ValueError("Input reflectance tensor contains NaN values.")
        min_reflectance = reflectance_tensor.min()
        max_reflectance = reflectance_tensor.max()
        if max_reflectance == min_reflectance:
            raise ValueError("max_reflectance and min_reflectance are equal, leading to division by zero.")
        scaled_reflectance = (reflectance_tensor - min_reflectance) / (max_reflectance - min_reflectance)
        intensity = scaled_reflectance * 65535
        self.pos[:, 3] = scaled_reflectance
        return self.pos
    
    def write_voxels(self):
        do_ground = 'n_z' not in self.pos.columns
        if not do_ground:
            grd = torch.tensor(self.pos[['n_z']].values, dtype=torch.float)
        else:
            grd = None

        self.pos = torch.tensor(self.pos.values, dtype=torch.float).to(device='cuda')

        if do_ground:
            print('Height Normalising Point Cloud')
            grd = self.gpu_ground()
        else:
            pass
        
        # if self.pointspacing:
        #     self.pos = self.downsample()

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
        return grd

def preprocess(args):
    n_z = Voxelise(args.pc, vxpath=args.vxfile, minpoints=args.min_pts, maxpoints=args.max_pts, pointspacing=args.resolution, gridsize = args.grid_size).write_voxels()
    args.pc['n_z'] = n_z.detach().numpy()



