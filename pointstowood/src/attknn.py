import torch
import faiss
import torch.nn as nn  # Import the torch.nn module


class AttentionBasedKNN(nn.Module):
    def __init__(self, num_heads=1):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, x, pos, topk):
        # x: [n, input_dim], pos: [n, 3]
        
        # Compute dynamic query based on x (features)
        query = x  # Directly use x as the query tensor
        
        # Normalize the query
        normalized_query = query / torch.norm(query, dim=-1, keepdim=True)

        # Normalize the position (spatial) information
        normalized_pos = pos / torch.norm(pos, dim=-1, keepdim=True)

        # Combine query with position (or other feature information)
        query_with_pos = torch.cat([normalized_query, normalized_pos], dim=-1)  # Concatenate features and position
        
        # Flatten query for FAISS search
        flatten_query = query_with_pos.reshape(-1, query_with_pos.shape[-1])
        
        # Perform the KNN search using FAISS
        distances, indices = self.index.search(flatten_query.cpu().numpy(), topk)
        distances = torch.tensor(distances, device=query.device)
        indices = torch.tensor(indices, device=query.device)

        # Scaling the distances to compute attention scores
        sqrt_dk = query_with_pos.shape[-1] ** 0.5
        scaled_product = sqrt_dk - distances * (sqrt_dk / 2)
        attention_scores = torch.nn.functional.softmax(scaled_product, dim=-1)

        # Use indices to select weighted values
        weighted_values = torch.einsum("...k,...kv->...v", attention_scores, self.weight_values[indices])

        result = weighted_values.flatten(-2, -1)
        return result
