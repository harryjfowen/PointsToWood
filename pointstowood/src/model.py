import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate
from torch.nn import Sequential as Seq, Linear as Lin
from torch_geometric.nn import knn
from src.pointnet import PointNetConv
from torch_geometric.nn import Set2Set
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
import torch.nn as nn
from src.sampling import DifferentiableSampler
from src.attention import ReflectanceGatingAttention

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
        super().__init__()
        expanded_channels = in_channels * expansion_factor
        self.expand = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, expanded_channels, kernel_size=1),
            torch.nn.BatchNorm1d(expanded_channels),
            torch.nn.SiLU(inplace=True),
        )
        self.conv = torch.nn.Sequential(
            DepthwiseSeparableConv1d(expanded_channels, expanded_channels, kernel_size=1),
            torch.nn.BatchNorm1d(expanded_channels),
            torch.nn.SiLU(inplace=True),
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

class GlobalSAModule(torch.nn.Module):
    def __init__(self, NN):
        super().__init__()
        self.NN = MLP(NN)
        self.norm = torch.nn.LayerNorm(NN[-1])
        self.size = (NN[-1] * 2)
        self.set2set = Set2Set(NN[-1], processing_steps=8)
        
    def forward(self, x, pos, batch, reflectance, sf):
        x = self.NN(x)
        x = self.norm(x)
        x = self.set2set(x, batch)
        pos = pos.new_zeros((self.size, 3))
        batch = torch.arange(self.size, device=batch.device)
        reflectance = reflectance.new_zeros(self.size)
        return x, pos, batch, reflectance, sf

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(in_channels={self.set2set.in_channels}, processing_steps=32)'
    
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
        Seq(
            Lin(channels[i - 1], channels[i]),
            torch.nn.BatchNorm1d(
                num_features=channels[i],
            ),
            torch.nn.SiLU(inplace=True),
        )
        for i in range(1, len(channels))
    ])

def apply_spatial_dropout(tensor, dropout_layer):
    coord_channels = tensor[:3, :]
    feature_channels = tensor[3:, :].unsqueeze(0).permute(0, 2, 1)
    feature_channels = dropout_layer(feature_channels).permute(0, 2, 1).squeeze(0)
    return torch.cat((coord_channels, feature_channels), dim=0)
    
class SAModule(torch.nn.Module):
    def __init__(self, resolution, k, NN, RNN, num_epochs, tau_init=1.0, tau_min=0.1):
        super().__init__()
        self.resolution = resolution
        self.k = k
        self.sampler = DifferentiableSampler(in_channels=(NN[0]), tau_init=tau_init, tau_min=tau_min, num_epochs=num_epochs)
        self.conv = PointNetConv(local_nn=MLP(NN), global_nn=None, shape_kernel=None, add_self_loops=False, radius=None)
        self.residual_block = InvertedResidualBlock(RNN, RNN)
        self.ratio = 0.5
        self.num_epochs = num_epochs
    
    def voxelsample(self, pos, batch, resolution):
        voxel_indices = voxel_grid(pos, resolution, batch)
        _, idx = consecutive_cluster(voxel_indices)
        return idx

    def forward(self, x, pos, batch, reflectance, sf):

        pos = torch.cat([pos[:, :3], reflectance.unsqueeze(-1)], dim=-1)

        idx = self.voxelsample(pos[:, :3], batch, self.resolution)
        row, col = knn(x=pos[:, :3], y=pos[idx, :3], k=self.k, batch_x=batch, batch_y=batch[idx])

        edge_index = torch.stack([col, row], dim=0)
        
        if x is not None:
            x = self.conv((x, x[idx]), (pos, pos[idx]), edge_index, batch[edge_index])
        else:
            x = self.conv((x, x), (pos, pos[idx]), edge_index, batch[edge_index])

        x = self.residual_block(x)

        pos, batch, reflectance = pos[idx, :3], batch[idx], reflectance[idx]
        return x, pos, batch, reflectance, sf

class Net(torch.nn.Module):
    def __init__(self, num_classes, C=32, num_epochs=None):
        super(Net, self).__init__()
        self.num_epochs = num_epochs

        self.reflectance_attention = ReflectanceGatingAttention(hidden_dim=C)

        self.sa1_module = SAModule(0.04, 32, [4, C * 4, C * 8], C * 8, num_epochs)#71
        self.sa2_module = SAModule(0.08, 32, [(C * 8) + 4, C * 8, C * 8], C * 8, num_epochs)#263
        self.sa3_module = GlobalSAModule([C * 8, C * 8, C * 8])#512

        self.fp3_module = FPModule(1, [C * 24, C * 20, C * 16])
        self.fp2_module = FPModule(2, [C * 24, C * 20, C * 16])
        self.fp1_module = FPModule(2, [C * 16, C * 16, C * 16])

        self.spatial_dropout1 = torch.nn.Dropout(p=0.1)
        self.spatial_dropout2 = torch.nn.Dropout(p=0.1)
        self.spatial_dropout3 = torch.nn.Dropout(p=0.1)
        self.dropout = torch.nn.Dropout(p=0.1)

        self.feature_reducer = MLP([C * 16, C * 8, C * 4, C])

        self.conv1 = torch.nn.Conv1d(C * 16, C * 16, 1)
        self.conv2 = torch.nn.Conv1d(C * 16, num_classes, 1)
        self.norm = torch.nn.BatchNorm1d(C * 16)

        initialize_weights(self)

    def forward(self, data):
        
        data.reflectance = self.reflectance_attention(data.pos, data.reflectance, data.batch)
        
        sa0_out = (data.x, data.pos, data.batch, data.reflectance, data.sf)

        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        sa1_out = (apply_spatial_dropout(sa1_out[0], self.spatial_dropout1),) + sa1_out[1:]
        sa2_out = (apply_spatial_dropout(sa2_out[0], self.spatial_dropout2),) + sa2_out[1:]
        sa3_out = (apply_spatial_dropout(sa3_out[0], self.spatial_dropout3),) + sa3_out[1:]

        fp3_out = self.fp3_module(*sa3_out[:-2], *sa2_out[:-2])
        fp2_out = self.fp2_module(*fp3_out, *sa1_out[:-2])
        fp1_out, _, _ = self.fp1_module(*fp2_out, *sa0_out[:-2])
                
        with torch.no_grad():
            compressed = self.feature_reducer(fp1_out)

        output = self.conv1(fp1_out.unsqueeze(dim=0).permute(0, 2, 1))
        output = self.dropout(output)
        output = F.silu(self.norm(output))
        output = torch.squeeze(self.conv2(output)).to(torch.float)

        # if not self.training:
        #     return output, compressed

        return output
        
