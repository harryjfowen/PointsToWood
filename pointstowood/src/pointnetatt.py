class AttentivePointNetConv(MessagePassing):
    def __init__(
        self, 
        in_channels: int,
        local_nn: Optional[Callable] = None,
        global_nn: Optional[Callable] = None,
        attention_division: int = 2,
        add_self_loops: bool = True, 
        **kwargs
    ):
        self.radius = kwargs.pop('radius', None)
        kwargs.setdefault('aggr', 'max')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.attention_dims = max(in_channels // attention_division, 32)
        
        # Combined projection for Q,K,V to reduce memory usage
        self.qkv_proj = nn.Linear(in_channels + 4, 3 * self.attention_dims, bias=False)
        
        # Single output projection
        self.out_proj = nn.Linear(self.attention_dims, in_channels, bias=False)
        
        # Single layer norm (reduced from 2)
        self.norm = nn.LayerNorm(self.attention_dims)
        
        # Simplified MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.attention_dims, self.attention_dims, bias=False),
            nn.ReLU()
        )
        
        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.mlp[0].bias is not None:
            nn.init.zeros_(self.mlp[0].bias)

    @torch.cuda.amp.autocast(enabled=True)  # Enable AMP for better performance
    def message(self, x_j: Optional[Tensor], pos_i: Tensor,
                pos_j: Tensor, edge_index_i: Tensor) -> Tensor:
        
        # Efficient relative position calculation
        relative_pos = pos_j[:, :3] - pos_i[:, :3]
        
        # Use torch.norm with keepdim for efficiency
        distances = torch.norm(relative_pos, p=2, dim=1, keepdim=True)
        max_distances = scatter_max(distances, edge_index_i, dim=0)[0][edge_index_i]
        
        # Normalize positions (avoid division by zero)
        normalized_pos = relative_pos / (max_distances + 1e-8)
        
        # Combine all features at once
        if x_j is not None:
            combined_features = torch.cat([normalized_pos, pos_j[:, 3:4], x_j], dim=-1)
        else:
            combined_features = torch.cat([normalized_pos, pos_j[:, 3:4], torch.zeros_like(normalized_pos)], dim=-1)
        
        # Single QKV projection
        qkv = self.qkv_proj(combined_features)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Efficient attention computation
        scale = float(self.attention_dims) ** -0.5
        attn = (q * k) * scale  # Dot product attention
        
        # Neighborhood-wise softmax
        attn = scatter_softmax(attn.sum(-1), edge_index_i, dim=0)
        
        # Apply attention and MLP in one go
        out = v * attn.unsqueeze(-1)
        out = self.norm(out)
        out = self.mlp(out)  # Simplified MLP
        out = self.out_proj(out)
        
        # Optional local_nn
        if self.local_nn is not None:
            out = self.local_nn(out)
            
        return out

    def forward(
        self,
        x: Union[OptTensor, PairOptTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:
        if not isinstance(x, tuple):
            x = (x, None)
        if isinstance(pos, Tensor):
            pos = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, 
                    num_nodes=min(pos[0].size(0), pos[1].size(0))
                )

        out = self.propagate(edge_index, x=x, pos=pos)
        
        if self.global_nn is not None:
            out = self.global_nn(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'attention_dims={self.attention_dims})')