import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate
from torch_geometric.nn import knn
from src.pointnet_attention import AttentivePointNetConv
from torch_geometric.nn import Set2Set, voxel_grid, knn, global_max_pool
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter_add   
from src.kan import KAN

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            
            if fan_in == 0 or fan_out == 0:
                continue
                
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_uniform_(
                    m.weight, 
                    mode='fan_in',
                    nonlinearity='linear',
                    a=1.0
                )
            else: 
                torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
                
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

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
        out = F.relu(out, inplace=True)
        out = self.pointwise_conv(out)
        out = self.pointwise_bn(out)
        return out
    
class InvertedResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4):
        super().__init__()
        expanded_channels = in_channels * expansion_factor
        self.expand = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, expanded_channels, kernel_size=1),
            torch.nn.BatchNorm1d(expanded_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.conv = torch.nn.Sequential(
            DepthwiseSeparableConv1d(expanded_channels, expanded_channels, kernel_size=1),
            torch.nn.BatchNorm1d(expanded_channels),
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
        out = F.relu(out, inplace=True)
        return out

class GlobalSAModule(torch.nn.Module):
    def __init__(self, NN):
        super().__init__()
        self.NN = KAN(NN)
        self.norm = torch.nn.LayerNorm(NN[-1])
        self.size = (NN[-1] * 2)
        self.set2set = Set2Set(NN[-1], processing_steps=8)
        
    def forward(self, x, pos, batch, reflectance):
        x = self.NN(x)
        x = self.norm(x)
        x = self.set2set(x, batch)
        pos = pos.new_zeros((self.size, 3))
        batch = torch.arange(self.size, device=batch.device)
        reflectance = reflectance.new_zeros(self.size)
        return x, pos, batch, reflectance

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(in_channels={self.set2set.in_channels}, processing_steps=32)'

class FPModule(torch.nn.Module):
    def __init__(self, k, NN):
        super().__init__()
        self.k = k
        self.NN = KAN(NN)

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        identity = x
        x = self.NN(x)
        if x.shape == identity.shape:  # Add residual if shapes match
            x = x + identity
        return x, pos_skip, batch_skip

def apply_spatial_dropout(tensor, dropout_layer):
    if tensor.size(0) <= 4: 
        return tensor
    coord_channels = tensor[:4]
    feature_channels = tensor[4:].unsqueeze(0).permute(0, 2, 1)
    feature_channels = dropout_layer(feature_channels).permute(0, 2, 1).squeeze(0)
    return torch.cat((coord_channels, feature_channels), dim=0)
    
class SAModule(torch.nn.Module):
    def __init__(self, resolution, k, NN, RNN, num_epochs):
        super().__init__()
        self.resolution = resolution 
        self.k = k
        self.base_ratio = 0.2 
        self.ratio_range = 0.05 
        
        in_channels = max(1, NN[0] - 4) 
        
        self.reflectance_gate = BinaryReflectanceGate(hidden_dim=NN[0])
        
        self.conv = AttentivePointNetConv(
            local_nn=KAN(NN), 
            attn_dim=in_channels//2, 
            in_channels=in_channels,
            global_nn=None, 
            add_self_loops=False,
            attention_type='softmax'
        )
        self.residual_block = InvertedResidualBlock(RNN, RNN)
        self.num_epochs = num_epochs
    
    def voxelsample(self, pos, batch, resolution):
        voxel_indices = voxel_grid(pos, resolution, batch)
        _, idx = consecutive_cluster(voxel_indices)
        return idx
    
    def random_sample(self, pos, batch, ratio):

        rand_ratio = ratio + (torch.rand(1).item() * 2 - 1) * self.ratio_range 
        rand_ratio = max(0.15, min(0.25, rand_ratio)) 
        
        device = pos.device
        batch_size = batch.max() + 1
        ones = torch.ones_like(batch)
        counts = scatter_add(ones, batch, dim=0, dim_size=batch_size)        
        samples_per_batch = (counts * rand_ratio).long() 
        rand = torch.rand(batch.size(0), device=device)
        rand = rand + batch.float()        
        sorted_indices = rand.argsort()        
        batch_positions = torch.arange(len(batch), device=device) - torch.zeros_like(batch).scatter_add(0, batch, ones).cumsum(0)[batch] + ones.scatter_add(0, batch, ones)[batch]        
        keep_mask = batch_positions <= samples_per_batch[batch]
        return sorted_indices[keep_mask]

    def forward(self, x, pos, batch, reflectance):
        if reflectance is not None and torch.abs(reflectance).mean() > 1e-6:
            reflectance = self.reflectance_gate(pos, reflectance, batch)

        if pos.size(1) == 3:  
            pos_with_refl = torch.cat([pos, reflectance.unsqueeze(-1)], dim=-1)
        else:
            pos_with_refl = pos  

        # if self.training: 
        #     idx = self.random_sample(pos[:, :3], batch, self.base_ratio)
        # else:
        #     idx = self.voxelsample(pos[:, :3], batch, self.resolution)
        
        idx = self.voxelsample(pos[:, :3], batch, self.resolution)

        row, col = knn(x=pos[:, :3], y=pos[idx, :3], k=self.k, batch_x=batch, batch_y=batch[idx])
        
        edge_index = torch.stack([col, row], dim=0)
        
        if x is not None:
            x = self.conv((x, x[idx]), (pos_with_refl, pos_with_refl[idx]), edge_index)
        else:
            x = self.conv((x, x), (pos_with_refl, pos_with_refl[idx]), edge_index)
        
        x = self.residual_block(x)
        
        pos, batch, reflectance = pos[idx, :3], batch[idx], reflectance[idx]
        return x, pos, batch, reflectance

class BinaryReflectanceGate(torch.nn.Module):
    def __init__(self, hidden_dim=16, temperature=1.0):
        super().__init__()
        self.hidden_dim, self.temperature = hidden_dim, temperature
        self.point_nn = torch.nn.Sequential(torch.nn.Linear(4, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU())
        self.gate_nn = torch.nn.Linear(hidden_dim, 2)
        self.gate_nn.bias.data = torch.tensor([0.0, 2.0])

    def gumbel_softmax(self, logits, tau=1.0, hard=True):
        gumbels = -torch.empty_like(logits).exponential_().log()
        y_soft = F.softmax((logits + gumbels) / tau, dim=-1)
        if hard:
            index = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            return (y_hard - y_soft).detach() + y_soft
        return y_soft

    def forward(self, pos, reflectance, batch):
        features = torch.cat([pos, reflectance.unsqueeze(-1)], dim=-1)
        point_features = self.point_nn(features)
        sample_features = global_max_pool(point_features, batch)
        gate_logits = self.gate_nn(sample_features)
        gate_decision = self.gumbel_softmax(gate_logits, tau=self.temperature, hard=False)
        use_reflectance = gate_decision[:, 1][batch].unsqueeze(-1)
        return (use_reflectance * reflectance.unsqueeze(-1)).squeeze(-1)

    def update_temperature(self, epoch, max_epochs):
        self.temperature = max(1.0 * (1 - epoch / (2 * max_epochs)), 0.1)

class Net(torch.nn.Module):
    def __init__(self, num_classes, C=16, num_epochs=None):
        super().__init__()
        self.num_epochs = num_epochs
        self.C = C

        self.sa1_module = SAModule(0.04, 32, [4, C * 4], C * 4, num_epochs)
        self.sa2_module = SAModule(0.08, 32, [C * 4 + 4, C * 8], C * 8, num_epochs)
        self.sa3_module = SAModule(0.16, 32, [C * 8 + 4, C * 8], C * 8, num_epochs)
        self.sa4_module = GlobalSAModule([C * 8, C * 8])

        self.fp4_module = FPModule(1, [C * 24, C * 8])
        self.fp3_module = FPModule(2, [C * 16, C * 8])
        self.fp2_module = FPModule(2, [C * 12, C * 8])
        self.fp1_module = FPModule(2, [C * 8, C * 8])

        self.feature_reducer = KAN([C * 8, C * 8, C])
        self.conv1 = torch.nn.Conv1d(C * 8, C * 8, 1)
        self.norm = torch.nn.BatchNorm1d(C * 8)
        self.conv2 = torch.nn.Conv1d(C * 8, num_classes, 1)

        initialize_weights(self)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch, data.reflectance)
        
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)

        fp4_out = self.fp4_module(sa4_out[0], sa4_out[1], sa4_out[2], 
                                sa3_out[0], sa3_out[1], sa3_out[2])
        fp3_out = self.fp3_module(fp4_out[0], fp4_out[1], fp4_out[2],
                                sa2_out[0], sa2_out[1], sa2_out[2])
        fp2_out = self.fp2_module(fp3_out[0], fp3_out[1], fp3_out[2], 
                                sa1_out[0], sa1_out[1], sa1_out[2])
        fp1_out, _, _ = self.fp1_module(fp2_out[0], fp2_out[1], fp2_out[2],
                                      sa0_out[0], sa0_out[1], sa0_out[2])

        with torch.no_grad():
            compressed = self.feature_reducer(fp1_out)

        output = self.conv1(fp1_out.unsqueeze(dim=0).permute(0, 2, 1))
        output = F.relu(self.norm(output))
        output = torch.squeeze(self.conv2(output)).to(torch.float)

        return output

    def train(self, mode=True):
        return super().train(mode)
        