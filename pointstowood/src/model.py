import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN
from torch_geometric.nn import PointNetConv, radius, voxel_grid, knn
from src.PointNet import PointNetConv
from src.AnisotropicConv import AnisotropicConv
from torch_geometric.nn.pool.consecutive import consecutive_cluster
import torch.nn as nn
import math
from torchvision.ops import stochastic_depth
from torch_scatter import scatter_mean

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False), nn.Sigmoid()
        )

    def forward(self, x, batch):
        z = scatter_mean(x, batch, dim=0)
        s = self.fc(z)[batch]
        return x * s
    
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
        self.leaky_relu = torch.nn.LeakyReLU()
        
    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.depthwise_bn(out)
        out = self.leaky_relu(out)
        out = self.pointwise_conv(out)
        out = self.pointwise_bn(out)
        out = self.leaky_relu(out)
        return out
    
class InvertedResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4, layer_idx=0, total_layers=5, pL=0.5):
        super(InvertedResidualBlock, self).__init__()
        self.expansion_factor = expansion_factor
        expanded_channels = in_channels * expansion_factor
        
        self.expand = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, expanded_channels, kernel_size=1),
            torch.nn.BatchNorm1d(expanded_channels),
            torch.nn.LeakyReLU(),
        )
        self.conv = torch.nn.Sequential(
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

        self.survival_prob = round(0.1 + (layer_idx / (total_layers-1)) * (0.3 - 0.1), 2)

        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x):
        residual = x
        
        out = x.unsqueeze(0).permute(0, 2, 1)
        out = self.expand(out)
        out = self.conv(out)
        out = self.project(out)
        out = out.permute(0, 2, 1).squeeze(0)
        
        out = stochastic_depth(out, p=self.survival_prob, mode="batch", training=self.training)
        
        residual = self.shortcut(residual)
        out += residual
        out = self.leaky_relu(out)
        return out

class SAModule(torch.nn.Module):
    def __init__(self, resolution, k, NN, num_blocks=1, start_layer_idx=0, total_layers=5, num_kernel_points=16):
        super(SAModule, self).__init__()
        self.resolution = resolution
        self.k = k

        self.conv = AnisotropicConv(
            local_nn=MLP(NN),
            global_nn=None,
            add_self_loops=False,
            num_kernel_points=num_kernel_points
        )
        
        self.residual_blocks = nn.ModuleList([
            InvertedResidualBlock(
                NN[-1], 
                NN[-1],
                layer_idx=start_layer_idx + i,
                total_layers=total_layers
            ) 
            for i in range(num_blocks)
        ])

        self.se = SqueezeExcite(NN[-1], reduction=4)

        
    def voxelsample(self, pos, batch, resolution):
        voxel_indices = voxel_grid(pos, resolution, batch)
        _, idx = consecutive_cluster(voxel_indices)
        return idx
    
    def forward(self, x, pos, batch, reflectance, sf):
        pos = torch.cat([pos[:, :3], reflectance.unsqueeze(-1)], dim=-1)

        idx = self.voxelsample(pos[:, :3], batch, self.resolution)

        row, col = knn(pos[:, :3], pos[idx, :3], k=self.k, batch_x=batch, batch_y=batch[idx])
        edge_index = torch.stack([col, row], dim=0)

        pos[:, :3] = pos[:, :3] / sf[batch].unsqueeze(-1)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos[:, :3] = pos[:, :3] * sf[batch].unsqueeze(-1)
        
        for block in self.residual_blocks:
            x = block(x)

        x = self.se(x, batch[idx])
        
        pos, batch, reflectance = pos[idx, :3], batch[idx], reflectance[idx]
        return x, pos, batch, reflectance, sf

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

def MLP(channels):
    return Seq(*[
        Seq(*( [Lin(channels[i - 1], channels[i]), torch.nn.LeakyReLU()] + ([BN(channels[i])] if i != 1 else []) ))
        for i in range(1, len(channels))
    ])

class STEM(torch.nn.Module):
    def __init__(self, k, NN):
        super().__init__()
        self.k = k
        self.conv = PointNetConv(
            local_nn=MLP(NN), 
            global_nn=None, 
            add_self_loops=False,
            radius = None, 
        )
        
    def forward(self, x, pos, batch, reflectance, sf):
        pos = torch.cat([pos[:, :3], reflectance.unsqueeze(-1)], dim=-1)
        row, col = radius(pos[:, :3], pos[:, :3], 0.02 * 2.1, batch, batch, max_num_neighbors=self.k)
        edge_index = torch.stack([col, row], dim=0) 
        pos[:, :3] = pos[:, :3] / sf[batch].unsqueeze(-1)
        x = self.conv(x, (pos, pos), edge_index)
        pos[:, :3] = pos[:, :3] * sf[batch].unsqueeze(-1)
        return x, pos[:, :3], batch, reflectance, sf

    
class NetFull(torch.nn.Module):
    def __init__(self, num_classes, C=32, num_kernel_points=16):
        super(NetFull, self).__init__()

        vx_1 = 0.02 * 1.618 # 0.03236
        vx_2 = vx_1 * 1.618 # 0.05242
        vx_3 = vx_2 * 1.618 # 0.08466
        vx_4 = vx_3 * 1.618 # 0.13684

        sqrt2 = 1.414
        def round_to_power_of_2(x):
            return 2 ** round(math.log2(x))

        C0h = int(C * sqrt2)                  
        C0 = round_to_power_of_2(C0h * sqrt2)

        C1h = int(C0 * sqrt2)                
        C1 = round_to_power_of_2(C1h * sqrt2) 

        C2h = int(C1 * sqrt2)                
        C2 = round_to_power_of_2(C2h * sqrt2)

        C3h = int(C2 * sqrt2)              
        C3 = round_to_power_of_2(C3h * sqrt2) 
        C4h = int(C3 * sqrt2)                

        C4 = round_to_power_of_2(C4h * sqrt2)
        
        total_blocks = 12        
        current_layer_idx = 0

        self.stem = STEM(8, [4, C0h, C0])

        self.sa1_module = SAModule(vx_1, 16, [(C0 + 5) * num_kernel_points, C1h, C1], num_blocks=3, 
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points)
        current_layer_idx += 3 
        
        self.sa2_module = SAModule(vx_2, 16, [(C1 + 5) * num_kernel_points, C2h, C2], num_blocks=6, 
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points)
        current_layer_idx += 6  
        
        self.sa3_module = SAModule(vx_3, 16, [(C2 + 5) * num_kernel_points, C3h, C3], num_blocks=2, 
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points)
        current_layer_idx += 2  
        
        self.sa4_module = SAModule(vx_4, 16, [(C3 + 5) * num_kernel_points, C4h, C4], num_blocks=1, 
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points)
        
        self.fp4_module = FPModule(1, [C4 + C3, C4, C4])
        self.fp3_module = FPModule(1, [C4 + C2, C4, C4])
        self.fp2_module = FPModule(1, [C4 + C1, C4, C4])
        self.fp1_module = FPModule(1, [C4 + C0, C4, C4])

        self.conv1 = torch.nn.Conv1d(C4, C4, 1)
        self.feat_head = torch.nn.Conv1d(C4, 32, 1)
        self.conv2 = torch.nn.Conv1d(C4, num_classes, 1)
        self.norm = torch.nn.BatchNorm1d(C4)

        initialize_weights(self)

    def forward(self, data, return_feats: bool = False):
        
        sa0_out = (data.x, data.pos, data.batch, data.reflectance, data.sf)

        sa0_out = self.stem(*sa0_out)

        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)

        fp4_out = self.fp4_module(*sa4_out[:-2], *sa3_out[:-2])
        fp3_out = self.fp3_module(*fp4_out, *sa2_out[:-2])
        fp2_out = self.fp2_module(*fp3_out, *sa1_out[:-2])
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out[:-2])

        x = self.conv1(x.unsqueeze(dim=0).permute(0, 2, 1))
        x = F.leaky_relu(self.norm(x))                       

        feat32 = torch.squeeze(self.feat_head(x)).to(torch.float)  
        logits = torch.squeeze(self.conv2(x)).to(torch.float)     

        if return_feats:
            return logits, feat32
        return logits
    
class NetLight(torch.nn.Module):
    def __init__(self, num_classes, C=16, num_kernel_points=16):
        super(NetLight, self).__init__()

        vx_1 = 0.02 * 1.618 # 0.03236
        vx_2 = vx_1 * 1.618 # 0.05242
        vx_3 = vx_2 * 1.618 # 0.08466
        vx_4 = vx_3 * 1.618 # 0.13684

        sqrt2 = 1.414
        def round_to_power_of_2(x):
            return 2 ** round(math.log2(x))

        C0h = int(C * sqrt2)                  
        C0 = round_to_power_of_2(C0h * sqrt2)

        C1h = int(C0 * sqrt2)                
        C1 = round_to_power_of_2(C1h * sqrt2) 

        C2h = int(C1 * sqrt2)                
        C2 = round_to_power_of_2(C2h * sqrt2)

        C3h = int(C2 * sqrt2)              
        C3 = round_to_power_of_2(C3h * sqrt2) 
        C4h = int(C3 * sqrt2)                

        C4 = round_to_power_of_2(C4h * sqrt2)
        
        total_blocks = 5        
        current_layer_idx = 0

        self.stem = STEM(8, [4, C0h, C0])

        self.sa1_module = SAModule(vx_1, 16, [(C0 + 5) * num_kernel_points, C1h, C1], num_blocks=1, 
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points)
        current_layer_idx += 1
        
        self.sa2_module = SAModule(vx_2, 16, [(C1 + 5) * num_kernel_points, C2h, C2], num_blocks=2, 
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points)
        current_layer_idx += 2  
        
        self.sa3_module = SAModule(vx_3, 16, [(C2 + 5) * num_kernel_points, C3h, C3], num_blocks=1, 
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points)
        current_layer_idx += 1  
        
        self.sa4_module = SAModule(vx_4, 16, [(C3 + 5) * num_kernel_points, C4h, C4], num_blocks=1, 
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points)
        
        self.fp4_module = FPModule(1, [C4 + C3, C4, C4])
        self.fp3_module = FPModule(1, [C4 + C2, C4, C4])
        self.fp2_module = FPModule(1, [C4 + C1, C4, C4])
        self.fp1_module = FPModule(1, [C4 + C0, C4, C4])

        self.conv1 = torch.nn.Conv1d(C4, C4, 1)
        self.feat_head = torch.nn.Conv1d(C4, 32, 1)
        self.conv2 = torch.nn.Conv1d(C4, num_classes, 1)
        self.norm = torch.nn.BatchNorm1d(C4)

        initialize_weights(self)

    def forward(self, data, return_feats: bool = False):
        
        sa0_out = (data.x, data.pos, data.batch, data.reflectance, data.sf)

        sa0_out = self.stem(*sa0_out)

        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)

        fp4_out = self.fp4_module(*sa4_out[:-2], *sa3_out[:-2])
        fp3_out = self.fp3_module(*fp4_out, *sa2_out[:-2])
        fp2_out = self.fp2_module(*fp3_out, *sa1_out[:-2])
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out[:-2])

        x = self.conv1(x.unsqueeze(dim=0).permute(0, 2, 1))
        x = F.leaky_relu(self.norm(x))                       

        feat32 = torch.squeeze(self.feat_head(x)).to(torch.float)  
        logits = torch.squeeze(self.conv2(x)).to(torch.float)     

        if return_feats:
            return logits, feat32
        return logits




