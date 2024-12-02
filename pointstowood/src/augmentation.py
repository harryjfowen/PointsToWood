import torch
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster

def rotate_3d(points):
    rotations = torch.deg2rad(torch.rand(3) * 180 - 90)
    cos, sin = torch.cos(rotations), torch.sin(rotations)
    roll_mat = torch.tensor([[1, 0, 0], [0, cos[0], -sin[0]], [0, sin[0], cos[0]]], dtype=torch.float32)
    pitch_mat = torch.tensor([[cos[1], 0, sin[1]], [0, 1, 0], [-sin[1], 0, cos[1]]], dtype=torch.float32)
    yaw_mat = torch.tensor([[cos[2], -sin[2], 0], [sin[2], cos[2], 0], [0, 0, 1]], dtype=torch.float32)
    points = points.view(-1, 3) @ roll_mat @ pitch_mat @ yaw_mat
    return points

def random_noise_addition(points):
    max_std_dev = 0.00333 
    random_noise_std_dev = torch.clamp(torch.rand(1) * max_std_dev, min=0.0001, max=max_std_dev)
    points = points + torch.normal(torch.zeros_like(points), random_noise_std_dev)
    return points

def random_rescale(points, scale_range=(0.9, 1.1)):
    scale_factor = torch.clamp(torch.rand(1) * (scale_range[1] - scale_range[0]) + scale_range[0], min=0.9, max=1.1)
    points = points * scale_factor
    return points

def silence_reflectance(feature):
    feature = torch.zeros_like(feature)
    return feature

def perturb_reflectance(feature):
    noise = torch.normal(mean=0.0, std=0.1, size=feature.size())
    feature = feature + noise
    return feature

def random_downsample(points, reflectance, label):
    resolution = torch.rand(1).item() * 0.03 + 0.01
    shuffled_points = points[torch.randperm(points.size(0))]
    voxel_indices = voxel_grid(shuffled_points, resolution)
    _, idx = consecutive_cluster(voxel_indices)
    return points[idx], reflectance[idx], label[idx]

def augmentations(pos, reflectance, label, mode='train'):
    rand_val_refl = torch.rand(1)
    rand_val_pos = torch.rand(1)
    if rand_val_refl < 0.25:
        reflectance = silence_reflectance(reflectance)
    if mode == 'train':
        if rand_val_refl >= 0.25 and rand_val_refl < 0.5:
            reflectance = perturb_reflectance(reflectance)
    if rand_val_pos < 0.25:
        pos = rotate_3d(pos)
    # if rand_val_pos > 0.25 and rand_val_pos <= 0.375:
    #     pos = random_rescale(pos)
    # if (rand_val_pos) > 0.375 and rand_val_pos <= 0.50:
    #     pos = random_noise_addition(pos)
    return pos, reflectance, label

