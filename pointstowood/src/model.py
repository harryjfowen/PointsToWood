import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU, SiLU
from torch_geometric.nn import knn, radius
from src.pointnet2 import PointNetConv
from torch_geometric.nn import Set2Set
from torch_scatter import scatter_max, scatter_mean, scatter_softmax
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
import torch.nn as nn
from src.sampling import DifferentiableSampler
import math

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_uniform_(
                    m.weight, 
                    mode='fan_in',
                    nonlinearity='linear',
                    a=1.0
                )

class DepthwiseSeparableConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise_conv = torch.nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels  
        )
        self.depthwise_bn = torch.nn.BatchNorm1d(in_channels)
        self.pointwise_conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1  
        )
        self.pointwise_bn = torch.nn.BatchNorm1d(in_channels)
        
    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.depthwise_bn(out)
        out = F.silu(out, inplace=True)
        out = self.pointwise_conv(out)
        out = self.pointwise_bn(out)
        out = F.silu(out, inplace=True)
        return out
    
class InvertedResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4):
        super(InvertedResidualBlock, self).__init__()
        self.expansion_factor = expansion_factor
        expanded_channels = in_channels * expansion_factor
        self.expand = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, expanded_channels, kernel_size=1),
            torch.nn.BatchNorm1d(expanded_channels),
            torch.nn.SiLU(),
        )
        self.conv = torch.nn.Sequential(
            DepthwiseSeparableConv1d(expanded_channels, expanded_channels, kernel_size=1),
            torch.nn.BatchNorm1d(expanded_channels),
            torch.nn.SiLU(),
            DepthwiseSeparableConv1d(expanded_channels, expanded_channels, kernel_size=1),  
            torch.nn.BatchNorm1d(expanded_channels),
        )
        self.project = torch.nn.Sequential(
            torch.nn.Conv1d(expanded_channels, out_channels, kernel_size=1),
            torch.nn.BatchNorm1d(out_channels)
        )
        if in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1),
                torch.nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = torch.nn.Sequential()

    def forward(self, x):
        residual = x
        out = x.unsqueeze(0).permute(0, 2, 1)
        out = self.expand(out)
        out = self.conv(out)
        out = self.project(out)
        out = out.permute(0, 2, 1).squeeze(0)
        residual = self.shortcut(residual)
        out += residual
        out = F.silu(out, inplace = True)
        return out
    
class InvertedResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4):
        super().__init__()
        expanded_channels = in_channels * expansion_factor
        
        self.expand = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, expanded_channels, kernel_size=1),
            torch.nn.BatchNorm1d(expanded_channels),
            torch.nn.SiLU(),
        )
        
        self.conv = torch.nn.Sequential(
            DepthwiseSeparableConv1d(expanded_channels, expanded_channels, kernel_size=1),
            torch.nn.BatchNorm1d(expanded_channels),
            torch.nn.SiLU(),
            DepthwiseSeparableConv1d(expanded_channels, expanded_channels, kernel_size=1),
            torch.nn.BatchNorm1d(expanded_channels),
        )
        
        self.project = torch.nn.Sequential(
            torch.nn.Conv1d(expanded_channels, out_channels, kernel_size=1),
            torch.nn.BatchNorm1d(out_channels)
        )
        
        if in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1),
                torch.nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = torch.nn.Sequential()

    def forward(self, x):
        residual = x
        out = x.unsqueeze(0).permute(0, 2, 1)
        out = self.expand(out)
        out = self.conv(out)
        out = self.project(out)
        out = out.permute(0, 2, 1).squeeze(0)
        residual = self.shortcut(residual)
        out += residual
        out = F.silu(out, inplace=True)
        return out

def random_sample(num_points):
    num_samples = int(num_points * 0.5)
    idx = torch.randperm(num_points)[:num_samples]
    idx, _ = torch.sort(idx)
    return idx
    
def sample_voxels(pos, batch, resolution):
    voxel_indices = voxel_grid(pos, resolution, batch)
    _, idx = consecutive_cluster(voxel_indices)
    return idx

def sample_voxels_by_max_feature(pos, features, batch, resolution):
    voxel_indices = voxel_grid(pos[:, :3], resolution, batch)
    max_indices = scatter_max(features, voxel_indices, dim=0)[1]
    mask = max_indices != -1 
    return max_indices[mask]  
    
class GlobalSAModule(torch.nn.Module):
    def __init__(self, NN, num_frequencies=16):
        super().__init__()
        self.set2set = Set2Set(NN[-1], processing_steps=16)
        self.NN = MLP(NN)
        self.norm = torch.nn.LayerNorm(NN[-1])
        self.size = (NN[-1] * 2) + 96 
        self.num_frequencies = num_frequencies
        
    def positional_encoding(self, pos, sf, batch):
        device = pos.device        
        num_batches = batch.max().item() + 1
        normalized_pos = pos[:, :3] / sf[batch].unsqueeze(1)
        batch_counts = torch.bincount(batch, minlength=num_batches)
        centers = torch.zeros(num_batches, 3, device=device)
        centers.index_add_(0, batch, normalized_pos)
        centers = centers / batch_counts.unsqueeze(1).clamp(min=1)
        normalized_pos = normalized_pos - centers[batch]
        frequencies = 1.0 / (10000 ** (2 * torch.arange(self.num_frequencies, device=device).float() / self.num_frequencies))
        pos_enc = normalized_pos.unsqueeze(-1) * frequencies
        return torch.cat([pos_enc.sin(), pos_enc.cos()], dim=-1).view(pos.shape[0], -1)

    def forward(self, x, pos, batch, reflectance, sf):
        pos_enc = self.positional_encoding(pos, sf, batch)
        x = torch.cat([x, pos_enc], dim=1)
        x = self.NN(x)
        x = self.norm(x)
        x = self.set2set(x, batch)
        pos = pos.new_zeros((self.size, 3))
        batch = torch.arange(self.size, device=batch.device)
        reflectance = reflectance.new_zeros(self.size)
        return x, pos, batch, reflectance, sf

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(NN={self.NN}, '
                f'num_frequencies={self.num_frequencies})')

class StemModule(torch.nn.Module):
    def __init__(self, NN, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        pos_enc_dim = 3 * num_frequencies * 2  
        input_dim = pos_enc_dim + 1  
        self.norm = torch.nn.LayerNorm(input_dim)
        self.mlp = MLP(NN)
    
    def positional_encoding(self, pos, sf, batch):
        device = pos.device        
        num_batches = batch.max().item() + 1
        normalized_pos = pos[:, :3] / sf[batch].unsqueeze(1)
        batch_counts = torch.bincount(batch, minlength=num_batches)
        centers = torch.zeros(num_batches, 3, device=device)
        centers.index_add_(0, batch, normalized_pos)
        centers = centers / batch_counts.unsqueeze(1).clamp(min=1)
        normalized_pos = normalized_pos - centers[batch]
        frequencies = 1.0 / (10000 ** (2 * torch.arange(self.num_frequencies, device=device).float() / self.num_frequencies))
        pos_enc = normalized_pos.unsqueeze(-1) * frequencies
        return torch.cat([pos_enc.sin(), pos_enc.cos()], dim=-1).view(pos.shape[0], -1)

        
    def forward(self, pos, reflectance, batch, sf):
        pos_enc = self.positional_encoding(pos, sf, batch)
        x = torch.cat([pos_enc, reflectance.unsqueeze(-1)], dim=1)
        x = self.norm(x)
        return self.mlp(x)
    
class FPModule(torch.nn.Module):
    def __init__(self, k, NN):
        super(FPModule, self).__init__()
        self.k = k
        self.NN = MLP(NN)

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.NN(x)
        return x, pos_skip, batch_skip

class LightKernel(torch.nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        self.register_buffer('kernel_dirs', torch.tensor([
            [1., 1., 1.], [-1., 1., 1.], 
            [1., -1., 1.], [1., 1., -1.],
            [-1., -1., 1.], [-1., 1., -1.], 
            [1., -1., -1.], [-1., -1., -1.]
        ], dtype=torch.float32) / torch.sqrt(torch.tensor(3.0)))
        self.feature_nn = torch.nn.Linear(8, out_channels)
        self.norm = nn.LayerNorm(out_channels)
    def forward(self, relative_pos, edge_index_i):
        if self.kernel_dirs.device != relative_pos.device:
            self.kernel_dirs = self.kernel_dirs.to(relative_pos.device)
            self.feature_nn = self.feature_nn.to(relative_pos.device)   
        directions = relative_pos / (torch.norm(relative_pos, dim=1, keepdim=True) + 1e-8)
        kernel_responses = torch.matmul(directions, self.kernel_dirs.T) 
        neighborhood_shape = scatter_mean(kernel_responses, edge_index_i, dim=0)  
        shape_features = self.feature_nn(neighborhood_shape) 
        shape_features = self.norm(shape_features)
        return shape_features[edge_index_i]  
    
def MLP(channels):
    return Seq(*[
        Seq(
            Lin(channels[i - 1], channels[i]),
            torch.nn.BatchNorm1d(
                num_features=channels[i],
            ),
            torch.nn.SiLU(),
        )
        for i in range(1, len(channels))
    ])

class SAModule(torch.nn.Module):
    def __init__(self, resolution, k, NN, RNN, num_epochs, tau_init=1.0, tau_min=0.1):
        super().__init__()
        self.resolution = resolution
        self.k = k
        self.conv = PointNetConv(local_nn=MLP(NN), global_nn=None, shape_kernel=LightKernel(3), attention_nn=None, add_self_loops=False, radius=radius)
        self.residual_block = InvertedResidualBlock(RNN, RNN)
        self.sampler = DifferentiableSampler(in_channels=NN[0]-7)
        self.ratio = 0.5
        self.num_epochs = num_epochs
        
        self.register_buffer('tau', torch.tensor(tau_init, dtype=torch.float))
        self.register_buffer('epoch', torch.tensor(0, dtype=torch.long))
        self.tau_init = tau_init
        self.tau_min = tau_min

    def forward(self, x, pos, batch, reflectance, sf):
        if self.training:
            progress = self.epoch / self.num_epochs
            tau_value = self.tau_min + 0.5 * (self.tau_init - self.tau_min) * (1 + math.cos(progress * math.pi))
            self.tau = torch.tensor(tau_value, device=self.tau.device, dtype=torch.float)
            self.epoch += 1

        pos = torch.cat([pos[:, :3], reflectance.unsqueeze(-1)], dim=-1)
        
        if self.resolution == 0.04:
            idx = sample_voxels(pos[:, :3], batch, self.resolution)
            row, col = radius(x=pos[:, :3], y=pos[idx, :3], r=self.resolution*2,  batch_x=batch, batch_y=batch[idx], max_num_neighbors=self.k)
        else:
            idx = self.sampler(x, batch, ratio=self.ratio, tau=self.tau)
            row, col = knn(x=pos[:, :3], y=pos[idx, :3], k=self.k, batch_x=batch, batch_y=batch[idx])
        
        edge_index = torch.stack([col, row], dim=0)
        
        x = self.conv((x, x[idx]), (pos, pos[idx]), edge_index, batch[edge_index])

        x = self.residual_block(x)
        pos, batch, reflectance = pos[idx, :3], batch[idx], reflectance[idx]
        return x, pos, batch, reflectance, sf

class Net(torch.nn.Module):
    def __init__(self, num_classes, C=32, num_epochs=None):
        super(Net, self).__init__()
        self.num_epochs = num_epochs
        
        self.stem_module = StemModule(NN=[97, C * 2, C],  num_frequencies=16)

        self.sa1_module = SAModule(0.04, 32, [C + 7, C * 2, C * 4], C * 4, num_epochs)
        self.sa2_module = SAModule(0.08, 32, [C * 4 + 7, C * 6, C * 8], C * 8, num_epochs)
        self.sa3_module = SAModule(0.32, 32, [C * 8 + 7, C * 12, C * 16], C * 16, num_epochs)
        self.sa4_module = GlobalSAModule([C * 16 + 96, C * 16, C * 16])

        self.fp4_module = FPModule(1, [C * 48, C * 32, C * 16])
        self.fp3_module = FPModule(2, [C * 24, C * 20, C * 16])
        self.fp2_module = FPModule(2, [C * 20, C * 16, C * 16])
        self.fp1_module = FPModule(2, [C * 17, C * 16, C * 16])

        self.conv1 = torch.nn.Conv1d(C * 16, C * 16, 1)
        self.conv2 = torch.nn.Conv1d(C * 16, num_classes, 1)

        self.conv1 = torch.nn.Conv1d(C * 16, C * 16, 1)
        self.conv2 = torch.nn.Conv1d(C * 16, num_classes, 1)
        self.norm = torch.nn.BatchNorm1d(C * 16)

        #self.feature_reducer = MLP([C * 16, C * 8, C * 4, C / 2])

        initialize_weights(self)

    def forward(self, data):
        
        data.x = self.stem_module(pos=data.pos, reflectance=data.reflectance, batch=data.batch, sf=data.sf)
        sa0_out = (data.x, data.pos, data.batch, data.reflectance, data.sf)

        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)

        fp4_out = self.fp4_module(*sa4_out[:-2], *sa3_out[:-2])
        fp3_out = self.fp3_module(*fp4_out, *sa2_out[:-2])
        fp2_out = self.fp2_module(*fp3_out, *sa1_out[:-2])
        features, _, _ = self.fp1_module(*fp2_out, *sa0_out[:-2])

        output = self.conv1(features.unsqueeze(dim=0).permute(0, 2, 1))
        output = F.silu(self.norm(output))
        output = torch.squeeze(self.conv2(output)).to(torch.float)

        # if not self.training:
        #     return output, self.feature_reducer(output)
        
        return output
