import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from pykdtree.kdtree import KDTree
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn import voxel_grid

def rotate_3d(pos):
    """Rotate point cloud using random rotation matrices"""
    rotations = torch.deg2rad(torch.rand(3) * 180 - 90)
    cos, sin = torch.cos(rotations), torch.sin(rotations)
    roll_mat = torch.tensor([[1, 0, 0], [0, cos[0], -sin[0]], [0, sin[0], cos[0]]], dtype=torch.float32, device=pos.device)
    pitch_mat = torch.tensor([[cos[1], 0, sin[1]], [0, 1, 0], [-sin[1], 0, cos[1]]], dtype=torch.float32, device=pos.device)
    yaw_mat = torch.tensor([[cos[2], -sin[2], 0], [sin[2], cos[2], 0], [0, 0, 1]], dtype=torch.float32, device=pos.device)
    
    # Apply rotation to XYZ coordinates only
    rotated_pos = pos.clone()
    rotated_pos[:, :3] = rotated_pos[:, :3].view(-1, 3) @ roll_mat @ pitch_mat @ yaw_mat
    return rotated_pos

def voxel_downsample(pos, reflectance, resolution=0.02):
    voxel_indices = voxel_grid(pos, resolution, batch=None)
    _, idx = consecutive_cluster(voxel_indices)
    return pos.clone()[idx], reflectance.clone()[idx]

def sor_filter(pos, reflectance, k=16, std_threshold=1.0):
    """Statistical Outlier Removal filter"""
    # Check if pos has at least 2014 points
    if pos.shape[0] < 4096:
        return pos, reflectance
        
    tree = KDTree(pos[:, :3].cpu().numpy())  
    distances, _ = tree.query(pos[:, :3].cpu().numpy(), k=k)
    distances = torch.from_numpy(distances).to(pos.device)  
    mean_distances = torch.mean(distances, dim=1)
    mean = torch.mean(mean_distances)
    std = torch.std(mean_distances)
    threshold = mean + std_threshold * std
    mask = mean_distances < threshold
    
    filtered_pos = pos[mask]
    filtered_reflectance = reflectance[mask] if reflectance is not None else None
    
    return filtered_pos, filtered_reflectance

def zero_reflectance(reflectance):
    """Set reflectance values to zero"""
    return torch.zeros_like(reflectance)

def generate_perspectives(data):
    """
    Generate specified perspectives of a point cloud for inference.
    
    For each sample (ALL WITH ZERO REFLECTANCE):
    - Original perspective
    - SOR filtered with std=1.0
    - SOR filtered with std=0.5
    - SOR filtered with std=2.0
    - Zero reflectance (original geometry)
    - Zero reflectance + SOR filtered with std=1.0
    
    Args:
        data: A torch_geometric.data.Data object containing point cloud
        
    Returns:
        List of Data objects, each containing a different perspective
    """
    perspectives = []
    
    # 1. Original data with zero reflectance
    original_zero_ref_data = Data(
        pos=data.pos.clone(),
        reflectance=zero_reflectance(data.reflectance),
        local_shift=data.local_shift,
        sf=data.sf
    )
    perspectives.append(original_zero_ref_data)

    #2. SOR filtered with std=1.0 and zero reflectance
    try:
        sor_pos_1, sor_ref_1 = sor_filter(data.pos, data.reflectance, k=16, std_threshold=1.0)
        sor_data_1 = Data(
            pos=sor_pos_1,
            reflectance=zero_reflectance(sor_ref_1),
            local_shift=data.local_shift,
            sf=data.sf
        )
        perspectives.append(sor_data_1)
    except Exception as e:
        print(f"Error applying SOR filter (std=1.0): {e}")
    
    #3. SOR filtered with std=0.5 and zero reflectance
    try:
        sor_pos_05, sor_ref_05 = sor_filter(data.pos, data.reflectance, k=16, std_threshold=0.5)
        sor_data_05 = Data(
            pos=sor_pos_05,
            reflectance=zero_reflectance(sor_ref_05),
            local_shift=data.local_shift,
            sf=data.sf
        )
        perspectives.append(sor_data_05)

    except Exception as e:
        print(f"Error applying SOR filter (std=0.5): {e}")

    # 4. SOR filtered with std=2.0 and zero reflectance
    try:
        sor_pos_2, sor_ref_2 = sor_filter(data.pos, data.reflectance, k=16, std_threshold=2.0)
        sor_data_2 = Data(
            pos=sor_pos_2,
            reflectance=zero_reflectance(sor_ref_2),
            local_shift=data.local_shift,
            sf=data.sf
        )
        perspectives.append(sor_data_2)

    except Exception as e:
        print(f"Error applying SOR filter (std=2.0): {e}")

    
    # 5. Zero reflectance (duplicate of #1, kept for compatibility)
    zero_ref_data = Data(
        pos=data.pos.clone(),
        reflectance=zero_reflectance(data.reflectance),
        local_shift=data.local_shift,
        sf=data.sf
    )

    perspectives.append(zero_ref_data)

    # 6. Zero reflectance + SOR filtered with std=1.0 (duplicate of #2, kept for compatibility)
    zero_ref_sor_1_data = Data(
        pos=sor_pos_1,
        reflectance=zero_reflectance(sor_ref_1),
        local_shift=data.local_shift,
        sf=data.sf
    )
    perspectives.append(zero_ref_sor_1_data)
    
    return perspectives

class MultiPerspectiveDataset(Dataset):
    """
    Dataset wrapper that generates multiple perspectives for each item and flattens them.
    This creates 7x more point clouds that will be handled by the PointCloudClassifier KNN.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.perspective_count = 3  # Original + SOR + Zero reflectance
        
        # Pre-calculate the mapping from flattened index to (original_idx, perspective_idx)
        self.index_map = []
        for i in range(len(base_dataset)):
            for j in range(self.perspective_count):
                self.index_map.append((i, j))
        
    def __len__(self):
        return len(self.base_dataset) * self.perspective_count
    
    def __getitem__(self, idx):
        # Get the original sample index and perspective index
        original_idx, perspective_idx = self.index_map[idx]
        
        # Get the original sample
        base_sample = self.base_dataset[original_idx]
        
        # Generate all perspectives
        perspectives = generate_perspectives(base_sample)
        
        # Return the requested perspective, or original if perspective generation failed
        if perspective_idx < len(perspectives):
            return perspectives[perspective_idx]
        else:
            print(f"Perspective {perspective_idx} not available, returning original")
            return perspectives[0] 