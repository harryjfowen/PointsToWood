import torch

def rotate_3d(points):
    rotations = torch.deg2rad(torch.rand(3) * 180 - 90)
    cos, sin = torch.cos(rotations), torch.sin(rotations)
    roll_mat = torch.tensor([[1, 0, 0], [0, cos[0], -sin[0]], [0, sin[0], cos[0]]], dtype=torch.float32)
    pitch_mat = torch.tensor([[cos[1], 0, sin[1]], [0, 1, 0], [-sin[1], 0, cos[1]]], dtype=torch.float32)
    yaw_mat = torch.tensor([[cos[2], -sin[2], 0], [sin[2], cos[2], 0], [0, 0, 1]], dtype=torch.float32)
    points = points.view(-1, 3) @ roll_mat @ pitch_mat @ yaw_mat
    return points

def random_scale_change(points, min_multiplier, max_multiplier):
    scale_factor = torch.FloatTensor(1).uniform_(min_multiplier, max_multiplier).to(points.device)
    return points * scale_factor

def perturb_leaf_reflectance(feature, label):
    feature_new = feature.clone()    
    leaf_mask = (label == 0)
    if leaf_mask.sum() > 0:
        feature_new[leaf_mask] = torch.rand(leaf_mask.sum()) * 0.3 + 0.7
    return feature_new

def perturb_wood_reflectance(feature, label):
    feature_new = feature.clone()    
    wood_mask = (label == 1)
    if wood_mask.sum() > 0:
        wood_noise = torch.randn_like(feature[wood_mask]) * 0.1
        feature_new[wood_mask] += wood_noise
    feature_new = torch.clamp(feature_new, min=-1.0, max=1.0)
    return feature_new

def selective_downsample(points, reflectance, label, min_points=4096):
    keep_mask = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    
    leaf_mask = (label == 0)
    if leaf_mask.sum() > min_points:
        leaf_indices = torch.nonzero(leaf_mask).squeeze(1)
        leaf_keep_ratio = torch.FloatTensor(1).uniform_(0.3, 0.7).item()
        num_leaf_keep = max(int(leaf_indices.size(0) * leaf_keep_ratio), min_points)
        
        perm = torch.randperm(leaf_indices.size(0))
        leaf_drop_indices = leaf_indices[perm[num_leaf_keep:]]
        if len(leaf_drop_indices) > 0:
            keep_mask[leaf_drop_indices] = False
    
    wood_mask = (label == 1)
    if wood_mask.sum() > min_points:
        wood_indices = torch.nonzero(wood_mask).squeeze(1)
        wood_drop_ratio = torch.FloatTensor(1).uniform_(0.0, 0.10).item()
        num_wood_drop = int(wood_indices.size(0) * wood_drop_ratio)
        
        if num_wood_drop > 0:
            perm = torch.randperm(wood_indices.size(0))
            wood_drop_indices = wood_indices[perm[:num_wood_drop]]
            keep_mask[wood_drop_indices] = False
    
    return points[keep_mask], reflectance[keep_mask], label[keep_mask]

def augmentations(pos, reflectance, label, mode='train'):
    if mode == 'train':

        rand_val_pos = torch.rand(1)

        if rand_val_pos < 0.25:
            pos = rotate_3d(pos)
        
        if rand_val_pos >= 0.25 and rand_val_pos < 0.50:
            pos, reflectance, label = selective_downsample(pos, reflectance, label)
            
        if rand_val_pos >= 0.50 and rand_val_pos < 0.75:
            pos = random_scale_change(pos, 0.8, 2.0)
            
        rand_val_refl = torch.rand(1)

        if rand_val_refl < 0.125:
            reflectance = perturb_leaf_reflectance(reflectance, label)
        
        if rand_val_refl >= 0.125 and rand_val_refl < 0.25:
            reflectance = perturb_wood_reflectance(reflectance, label)

        if rand_val_refl >= 0.25 and rand_val_refl < 0.50:
            reflectance = torch.zeros_like(reflectance)
        
        assert not torch.isnan(pos).any(), "NaN detected in positions"
        assert not torch.isnan(reflectance).any(), "NaN detected in reflectance"
        assert not torch.isnan(label).any(), "NaN detected in labels"
            
    return pos, reflectance, label

