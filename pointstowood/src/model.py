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
    def __init__(self, channels: int, reduction: int = 8, dropout: float = 0.1):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False), 
            nn.Sigmoid()
        )

    def forward(self, x, batch):
        z = scatter_mean(x, batch, dim=0)
        s = self.fc(z)[batch]
        return x * s
    
class DepthwiseSeparableConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise_conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.depthwise_gn = nn.BatchNorm1d(in_channels)
        self.pointwise_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pointwise_gn = nn.BatchNorm1d(out_channels)
        self.leaky_relu = torch.nn.LeakyReLU()
        
    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.depthwise_gn(out)
        out = self.leaky_relu(out)
        out = self.pointwise_conv(out)
        out = self.pointwise_gn(out)
        out = self.leaky_relu(out)
        return out


class DropPathPack(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x, lengths, return_mask: bool = False):
        if self.drop_prob <= 0 or not self.training:
            if return_mask:
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                mask = torch.ones(shape, device=x.device, dtype=x.dtype)
                return x, mask
            return x

        keep_prob = 1.0 - self.drop_prob

        if isinstance(lengths, torch.Tensor):
            lengths_list = lengths.tolist()
        else:
            lengths_list = lengths

        bernoulli = x.new_empty((len(lengths_list),)).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            bernoulli.div_(keep_prob)

        bernoulli_full = torch.cat([x.new_full((l,), b) for l, b in zip(lengths_list, bernoulli)], dim=0)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = bernoulli_full.view(shape)

        if return_mask:
            return x * mask, mask
        return x * mask

class InvertedResidualBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        expansion_factor: int = 4,  
        layer_idx: int = 0, 
        total_layers: int = 9,
        max_drop_rate: float = 0.2,  
        min_drop_rate: float = 0.0   
    ):
        super().__init__()
        
        expanded_channels = in_channels * expansion_factor
        
        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(expanded_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        self.depthwise = DepthwiseSeparableConv1d(
            expanded_channels, expanded_channels, kernel_size=1
        )
        
        self.project = nn.Sequential(
            nn.Conv1d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        self.use_skip = in_channels == out_channels
        if not self.use_skip:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.drop_path_graph = DropPathPack(drop_prob=0.2)
        
        self.final_activation = nn.LeakyReLU(inplace=True)

    def forward(self, x, batch: torch.Tensor = None):
        x_conv = x.unsqueeze(0).transpose(1, 2)
        
        if self.use_skip:
            residual = x
        else:
            residual_conv = self.shortcut(x_conv)
            residual = residual_conv.transpose(1, 2).squeeze(0)
        
        out = self.expand(x_conv)
        out = self.depthwise(out) 
        out = self.project(out)
        
        out = out.transpose(1, 2).squeeze(0)
        
        if self.use_skip:
            if batch is not None:
                lengths = torch.bincount(batch)
                out = self.drop_path_graph(out, lengths)
            out = out + residual
        else:
            out = out + residual
            
        out = self.final_activation(out)
        
        return out


class SAModule(torch.nn.Module):
    def __init__(self, resolution, k, NN, num_blocks=1, start_layer_idx=0, total_layers=5, num_kernel_points=16, learnable_kernels=False, expansion_factor=4, reverse_drop_schedule=False):
        super(SAModule, self).__init__()
        self.resolution = resolution
        self.k = k

        self.conv = AnisotropicConv(
            local_nn=MLP(NN),
            global_nn=None,
            add_self_loops=False,
            num_kernel_points=num_kernel_points,
            learnable_kernels=learnable_kernels
        )
        
        self.residual_blocks = nn.ModuleList([
            InvertedResidualBlock(
                NN[-1], 
                NN[-1],
                expansion_factor=expansion_factor,
                layer_idx=start_layer_idx + i,
                total_layers=total_layers, 
                max_drop_rate=0.5,
                min_drop_rate=0.1
            ) 
            for i in range(num_blocks)
        ])

        self.se = SqueezeExcite(NN[-1], reduction=8, dropout=0.1)
        
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
            x = block(x, batch[idx])

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

def MLPWithDropout(channels, dropout=0.1):
    layers = []
    for i in range(1, len(channels)):
        layers.append(Lin(channels[i - 1], channels[i]))
        layers.append(torch.nn.LeakyReLU())
        layers.append(BN(channels[i]))
        if i < len(channels) - 1:
            layers.append(torch.nn.Dropout(dropout))
    return Seq(*layers)

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
        pos_scaled = torch.cat([pos_scaled, reflectance.unsqueeze(-1)], dim=-1)
        
        x = self.conv(x, (pos_scaled, pos_scaled), edge_index)
        return x, pos[:, :3], batch, reflectance, sf


class NetFull(torch.nn.Module):
    def __init__(self, num_classes, C=32, num_kernel_points=32, learnable_kernels=False):
        super(NetFull, self).__init__()

        vx_1 = 0.02 * 1.618
        vx_2 = vx_1 * 1.618
        vx_3 = vx_2 * 1.618
        vx_4 = vx_3 * 1.618

        def double_progression(base_C):
            C0 = base_C
            C1 = int(C0 * 2)
            C2 = int(C1 * 2)
            C3 = int(C2 * 2)
            C4 = int(C3 * 2)
            return C0, C1, C2, C3, C4


        def kpconvx_progression(base_C):
            C0 = base_C
            C1 = int(base_C * 1.5)
            C2 = base_C * 2
            C3 = base_C * 3
            C4 = base_C * 4
            return C0, C1, C2, C3, C4

        C0, C1, C2, C3, C4 = kpconvx_progression(C)

        total_blocks = 8        
        current_layer_idx = 0

        self.stem = STEM(8, [4, C0//2, C0])

        self.sa1_module = SAModule(vx_1, 16, [(C0 + 5) * num_kernel_points, C1 * 4, C1], num_blocks=2, learnable_kernels=learnable_kernels,
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points, expansion_factor=4)
        current_layer_idx += 3

        self.sa2_module = SAModule(vx_2, 16, [(C1 + 5) * num_kernel_points, C2 * 4, C2], num_blocks=4, learnable_kernels=learnable_kernels,
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points, expansion_factor=4)
        current_layer_idx += 6

        self.sa3_module = SAModule(vx_3, 16, [(C2 + 5) * num_kernel_points, C3 * 4, C3], num_blocks=1, learnable_kernels=learnable_kernels,
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points, expansion_factor=4)
        current_layer_idx += 1

        self.sa4_module = SAModule(vx_4, 16, [(C3 + 5) * num_kernel_points, C4 * 4, C4], num_blocks=1, learnable_kernels=learnable_kernels,
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points, expansion_factor=4)
        
        self.fp4_module = FPModule(3, [C4 + C3, C4, C4])
        self.fp3_module = FPModule(3, [C4 + C2, C4, C4])
        self.fp2_module = FPModule(3, [C4 + C1, C4, C4])
        self.fp1_module = FPModule(3, [C4 + C0, C4, C4])

        self.seg_head = nn.Sequential(
            nn.Conv1d(C4, C4*4, 1),
            nn.BatchNorm1d(C4*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(C4*4, C4*2, 1),
            nn.BatchNorm1d(C4*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(C4*2, num_classes, 1)
        )        

        initialize_weights(self)

    def forward(self, data, return_feats: bool = False):
        sa0_out = (None, data.pos, data.batch, data.reflectance, data.sf)
        sa0_out = self.stem(*sa0_out)

        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)

        fp4_out = self.fp4_module(*sa4_out[:-2], *sa3_out[:-2])
        fp3_out = self.fp3_module(*fp4_out, *sa2_out[:-2])
        fp2_out = self.fp2_module(*fp3_out, *sa1_out[:-2])
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out[:-2])

        x = x.unsqueeze(dim=0).permute(0, 2, 1)
        logits = torch.squeeze(self.seg_head(x)).to(torch.float)

        return logits
    
class NetLight(torch.nn.Module):
    def __init__(self, num_classes, C=16, num_kernel_points=8, learnable_kernels=True):
        super(NetLight, self).__init__()

        vx_1 = 0.02 * 1.618
        vx_2 = vx_1 * 1.618
        vx_3 = vx_2 * 1.618
        vx_4 = vx_3 * 1.618

        def light_kpconvx_progression(base_C):
            C0 = base_C
            C1 = int(base_C * 1.5)
            C2 = base_C * 2
            C3 = base_C * 3
            C4 = base_C * 4
            return C0, C1, C2, C3, C4

        C0, C1, C2, C3, C4 = light_kpconvx_progression(C)
        
        total_blocks = 4        
        current_layer_idx = 0

        self.stem = STEM(8, [4, C0//2, C0])

        self.sa1_module = SAModule(vx_1, 16, [(C0 + 5) * num_kernel_points, C1 * 4, C1], num_blocks=1,
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points, learnable_kernels=learnable_kernels, expansion_factor=4)
        current_layer_idx += 1
        
        self.sa2_module = SAModule(vx_2, 16, [(C1 + 5) * num_kernel_points, C2 * 4, C2], num_blocks=2,
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points, learnable_kernels=learnable_kernels, expansion_factor=4)
        current_layer_idx += 2  
        
        self.sa3_module = SAModule(vx_3, 16, [(C2 + 5) * num_kernel_points, C3 * 4, C3], num_blocks=1,
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points, learnable_kernels=learnable_kernels, expansion_factor=4)
        current_layer_idx += 1  
        
        self.sa4_module = SAModule(vx_4, 16, [(C3 + 5) * num_kernel_points, C4 * 4, C4], num_blocks=1,
                                  start_layer_idx=current_layer_idx, total_layers=total_blocks, num_kernel_points=num_kernel_points, learnable_kernels=learnable_kernels, expansion_factor=4)
        
        self.fp4_module = FPModule(3, [C4 + C3, C4, C4])
        self.fp3_module = FPModule(3, [C4 + C2, C4, C4])
        self.fp2_module = FPModule(3, [C4 + C1, C4, C4])
        self.fp1_module = FPModule(3, [C4 + C0, C4, C4])

        self.seg_head = nn.Sequential(
            nn.Conv1d(C4, C4*4, 1),  
            nn.BatchNorm1d(C4*4),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(C4*4, C4*2, 1), 
            nn.BatchNorm1d(C4*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(C4*2, num_classes, 1)
        )

        initialize_weights(self)

    def forward(self, data, return_feats: bool = False):
        sa0_out = (None, data.pos, data.batch, data.reflectance, data.sf)
        sa0_out = self.stem(*sa0_out)

        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)

        fp4_out = self.fp4_module(*sa4_out[:-2], *sa3_out[:-2])
        fp3_out = self.fp3_module(*fp4_out, *sa2_out[:-2])
        fp2_out = self.fp2_module(*fp3_out, *sa1_out[:-2])
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out[:-2])

        x = x.unsqueeze(dim=0).permute(0, 2, 1)
        logits = torch.squeeze(self.seg_head(x)).to(torch.float)

        return logits
