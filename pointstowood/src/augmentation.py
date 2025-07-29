import torch
from pykdtree.kdtree import KDTree


def sor_filter(pos, reflectance, label, k=16, std_threshold=None):
    if std_threshold is None:
        std_threshold = torch.rand(1, device=pos.device) * 2.0 + 1.0
    tree = KDTree(pos.cpu().numpy())  
    distances, _ = tree.query(pos.cpu().numpy(), k=k)
    distances = torch.from_numpy(distances).to(pos.device)  
    mean_distances = torch.mean(distances, dim=1)
    mean = torch.mean(mean_distances)
    std = torch.std(mean_distances)
    threshold = mean + std_threshold * std
    mask = mean_distances < threshold
    pos_filtered = pos[mask]
    reflectance_filtered = reflectance[mask] if reflectance is not None else None
    label_filtered = label[mask] if label is not None else None
    return pos_filtered, reflectance_filtered, label_filtered

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

def jitter_points(points, std):
    noise = torch.normal(mean=0.0, std=std, size=points.size())
    points = points + noise
    return points

def random_flip(points):
    if torch.rand(1) < 0.5:
        points = points.clone()
        points[:, 0] = -points[:, 0]
    return points

def perturb_reflectance(feature):
    noise = torch.normal(mean=0.0, std=0.01, size=feature.size())
    feature = feature + noise
    return feature
    
def match_leaf_to_wood_reflectance(reflectance, label):
    wood_mask = label == 1
    leaf_mask = label == 0
    
    if wood_mask.sum() > 10 and leaf_mask.sum() > 10: 
        wood_mean = reflectance[wood_mask].mean()
        wood_std = reflectance[wood_mask].std()
        
        leaf_noise = torch.randn_like(reflectance[leaf_mask]) * wood_std
        leaf_reflectance = leaf_noise + wood_mean
        
        reflectance = reflectance.clone()
        reflectance[leaf_mask] = leaf_reflectance
            
    return reflectance

def random_point_dropout(pos, reflectance=None, label=None, drop_ratio: float = 0.10):
    
    if drop_ratio <= 0.0:
        return pos, reflectance, label

    keep_mask = torch.rand(len(pos), device=pos.device) > drop_ratio
    pos = pos[keep_mask]
    reflectance = reflectance[keep_mask] if reflectance is not None else None
    label = label[keep_mask] if label is not None else None
    return pos, reflectance, label

def augmentations(pos, reflectance, label, mode: str = "train"):
    """Apply geometry and reflectance augmentations.

    Geometry augs and reflectance augs are *independent*; one of each can
    fire in the same call.
    """

    # ---------------- Geometry branch ----------------
    if mode == "train":
        p_geom = torch.rand(1)

        if p_geom < 0.20:
            pos = rotate_3d(pos)

        elif p_geom > 0.20 and p_geom < 0.40:
            pos = random_scale_change(pos, 0.95, 1.05)

        elif p_geom > 0.40 and p_geom < 0.60:
            if len(pos) > 4096:
                pos, reflectance, label = sor_filter(pos, reflectance, label)

        elif p_geom > 0.60 and p_geom < 0.80:
            pos = jitter_points(pos, 0.005)

        elif p_geom > 0.90:
            pos = random_flip(pos)

    # ---------------- Reflectance branch ----------------
    p_refl = torch.rand(1)

    if mode == "train":
        if p_refl < 0.20:
            reflectance = torch.zeros_like(reflectance)

        elif p_refl > 0.20 and p_refl < 0.30:
            reflectance = perturb_reflectance(reflectance)

    elif mode == "test":

        if p_refl < 0.20:
            reflectance = torch.zeros_like(reflectance)

    return pos, reflectance, label
