import torch
from opt_einsum import contract

class KANLayer(torch.nn.Module):
    def __init__(self, d_input: int, d_output: int, n_order: int = 3):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.n_order = n_order
        
        # More stable initialization for KAN weights
        std = 0.02  # Smaller initial std for more stable start
        self.weights = torch.nn.Parameter(
            torch.randn(n_order + 1, d_input, d_output) * std
        )

        self.register_buffer('ones', None, persistent=False)

    def forward(self, x):
        # x: [batch_size, d_input]
        x = torch.tanh(x)

        if self.ones is None or self.ones.shape != x.shape:
            self.ones = x.new_ones(x.shape)
        
        x_sq = x.pow(2)

        # Chebyshev up to T3
        polys = torch.stack([
            self.ones,            
            x,                    
            2 * x_sq - 1,          
            4 * x * x_sq - 3 * x   
        ], dim=-1)
        
        # Replace torch.einsum with optimized contract
        output = contract('bin,nid->bd', polys, self.weights)
        
        return output

class KANSequential(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        
        for i in range(1, len(channels)):
            self.layers.append(KANLayer(channels[i-1], channels[i]))
            if i < len(channels) - 1:
                self.layers.append(torch.nn.BatchNorm1d(channels[i]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def KAN(channels):
    return KANSequential(channels) 