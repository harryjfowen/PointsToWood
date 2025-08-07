import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN, GroupNorm as GN
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
        self.depthwise_gn = nn.GroupNorm(min(32, in_channels), in_channels)
        self.pointwise_conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1  
        )
        self.pointwise_gn = nn.GroupNorm(min(32, out_channels), out_channels)
        self.leaky_relu = torch.nn.LeakyReLU()
        
    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.depthwise_gn(out)
        out = self.leaky_relu(out)
        out = self.pointwise_conv(out)
        out = self.pointwise_gn(out)
        out = self.leaky_relu(out)
        return out
    

class PointCloudStochasticDepth(nn.Module):
    """Stochastic depth implementation optimized for point cloud data."""
    
    def __init__(self, drop_rate: float = 0.0):
        super().__init__()
        self.drop_rate = drop_rate
    
    def forward(self, x):
        if not self.training or self.drop_rate == 0.0:
            return x
        
        # Bernoulli sampling for survival
        survival_rate = 1.0 - self.drop_rate
        if torch.rand(1, device=x.device).item() < survival_rate:
            return x / survival_rate  # Scale to maintain expected value
        else:
            return torch.zeros_like(x)


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.depthwise_gn = nn.GroupNorm(min(32, in_channels), in_channels)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pointwise_gn = nn.GroupNorm(min(32, out_channels), out_channels)
        self.activation = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.depthwise_gn(x)
        x = self.activation(x)
        x = self.pointwise_conv(x)
        x = self.pointwise_gn(x)
        x = self.activation(x)
        return x


class InvertedResidualBlock(nn.Module):
    """
    Inverted residual block with expansion for point cloud data.
    Compatible with PyTorch Geometric tensor format [num_points, channels].
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension  
        expansion_factor: Channel expansion ratio (default: 2)
        layer_idx: Current layer index for drop rate scheduling
        total_layers: Total number of layers for drop rate scheduling
        max_drop_rate: Maximum stochastic depth drop rate (default: 0.1)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        expansion_factor: int = 2,
        layer_idx: int = 0, 
        total_layers: int = 9,
        max_drop_rate: float = 0.1,
        reverse_drop_schedule: bool = False
    ):
        super().__init__()
        
        expanded_channels = in_channels * expansion_factor
        
        # Expansion phase
        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, expanded_channels), expanded_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # Depthwise convolution phase
        self.depthwise = DepthwiseSeparableConv1d(
            expanded_channels, expanded_channels, kernel_size=1
        )
        
        # Projection phase
        self.project = nn.Sequential(
            nn.Conv1d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels)
        )
        
        # Skip connection
        self.use_skip = in_channels == out_channels
        if not self.use_skip:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(32, out_channels), out_channels)
            )
        
        # Stochastic depth with linear scheduling
        if reverse_drop_schedule:
            # Reverse schedule: early layers get more dropout, later layers get less
            drop_rate = (total_layers - 1 - layer_idx) / max(total_layers - 1, 1) * max_drop_rate
        else:
            # Standard schedule: early layers get less dropout, later layers get more
            drop_rate = layer_idx / max(total_layers - 1, 1) * max_drop_rate
        self.stochastic_depth = PointCloudStochasticDepth(drop_rate)
        
        self.final_activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass for point cloud tensor.
        
        Args:
            x: Input tensor of shape [num_points, in_channels]
            
        Returns:
            Output tensor of shape [num_points, out_channels]
        """
        # Convert to Conv1d format: [num_points, channels] -> [1, channels, num_points]
        x_conv = x.unsqueeze(0).transpose(1, 2)
        
        # Skip connection
        if self.use_skip:
            residual = x
        else:
            residual_conv = self.shortcut(x_conv)
            residual = residual_conv.transpose(1, 2).squeeze(0)
        
        # Main branch
        out = self.expand(x_conv)
        out = self.depthwise(out) 
        out = self.project(out)
        
        # Convert back to point cloud format: [1, channels, num_points] -> [num_points, channels]
        out = out.transpose(1, 2).squeeze(0)
        
        # Apply stochastic depth to main branch
        out = self.stochastic_depth(out)
        
        # Residual connection and final activation
        out = out + residual
        out = self.final_activation(out)
        
        return out


class SAModule(torch.nn.Module):
    def __init__(self, resolution, k, NN, num_blocks=1, start_layer_idx=0, total_layers=5, num_kernel_points=16, reverse_drop_schedule=False):
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
                expansion_factor=4,  # Reduced from 4x to 2x
                layer_idx=start_layer_idx + i,
                total_layers=total_layers, 
                max_drop_rate=0.5,
                reverse_drop_schedule=True
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
        Seq(*( [Lin(channels[i - 1], channels[i]), torch.nn.LeakyReLU(), BN(channels[i])] ))
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
        row, col = radius(pos[:, :3], pos[:, :3], 0.02 * 2.1, batch, batch, max_num_neighbors=self.k)
        edge_index = torch.stack([col, row], dim=0) 
        pos_scaled = pos[:, :3] / sf[batch].unsqueeze(-1)
        x = self.conv(x, (pos_scaled, pos_scaled), edge_index)
        return x, pos[:, :3], batch, reflectance, sf

    
class NetFull(torch.nn.Module):
    def __init__(self, num_classes, C=32, num_kernel_points=16):
        super(NetFull, self).__init__()

        vx_1 = 0.02 * 1.618 # 0.03236
        vx_2 = vx_1 * 1.618 # 0.05242
        vx_3 = vx_2 * 1.618 # 0.08466
        vx_4 = vx_3 * 1.618 # 0.13684

        # KPConv-X progression function
        def square_progression(base_C):
            """Generate KPConv-X style progression: C → 1.5C → 1.33*1.5C → 1.5*1.33*1.5C → 1.33*1.5*1.33*1.5C"""
            C0 = base_C
            C1 = int(C0 * 1.5)  # 1.5x
            C2 = int(C1 * 1.33)  # 1.33x
            C3 = int(C2 * 1.5)   # 1.5x
            C4 = int(C3 * 1.33)  # 1.33x
            return C0, C1, C2, C3, C4
        
        def double_progression(base_C):
            C0 = base_C
            C1 = int(C0 * 2)
            C2 = int(C1 * 2)
            C3 = int(C2 * 2)
            C4 = int(C3 * 2)
            return C0, C1, C2, C3, C4
        
        # Generate progression
        C0, C1, C2, C3, C4 = double_progression(C) # 64, 128, 256, 512, 1024

        total_blocks = 9        
        current_layer_idx = 0

        self.stem = STEM(8, [4, C0 //2, C0])  # 4 channels: 3D relative pos + 1D distance

        self.sa1_module = SAModule(vx_1, 16, [(C0 + 5) * num_kernel_points, (C0 + 5) * num_kernel_points // 2, C1], num_blocks=2, 
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points)
        current_layer_idx += 2
        
        self.sa2_module = SAModule(vx_2, 16, [(C1 + 5) * num_kernel_points, (C1 + 5) * num_kernel_points // 2, C2], num_blocks=4, 
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points)
        current_layer_idx += 4    
        
        self.sa3_module = SAModule(vx_3, 16, [(C2 + 5) * num_kernel_points, (C2 + 5) * num_kernel_points // 2, C3], num_blocks=2, 
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points)
        current_layer_idx += 2
        
        self.sa4_module = SAModule(vx_4, 16, [(C3 + 5) * num_kernel_points, (C3 + 5) * num_kernel_points // 2, C4], num_blocks=1, 
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points)
        
        self.fp4_module = FPModule(1, [C4 + C3, C4, C4])
        self.fp3_module = FPModule(1, [C4 + C2, C4, C4])
        self.fp2_module = FPModule(1, [C4 + C1, C4, C4])
        self.fp1_module = FPModule(1, [C4 + C0, C4, C4])

        self.conv1 = torch.nn.Conv1d(C4, C4, 1)
        self.feat_head = torch.nn.Conv1d(C4, 32, 1)
        self.conv2 = torch.nn.Conv1d(C4, num_classes, 1)
        self.norm = nn.GroupNorm(min(32, C4), C4)

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
    def __init__(self, num_classes, C=8, num_kernel_points=16):
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

        C4 = round_to_power_of_2(C4h * sqrt2) #
        
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
        self.norm = nn.GroupNorm(min(32, C4), C4)

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




