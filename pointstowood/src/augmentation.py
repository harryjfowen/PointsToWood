import torch


def _density_downsample_fixed_spacing(pos, reflectance, label, spacing):
    """Downsample (pos, reflectance, label) with fixed grid spacing. No random draw."""
    if pos is None or pos.size(0) == 0:
        return pos, reflectance, label
    from src.utils import downsample_points_max
    pos_full = torch.cat([pos, reflectance.unsqueeze(1), label.unsqueeze(1)], dim=1)
    out = downsample_points_max(pos_full, spacing)
    return out[:, :3], out[:, 3], out[:, 4]


def random_density_downsample(pos, reflectance, label, spacing_min=0.01, spacing_max=0.04, prob=0.5):
    """Random voxel-grid downsampling to simulate different point densities (e.g. 1–4 cm).

    With probability prob, downsample (pos, reflectance, label) with a random grid spacing
    in [spacing_min, spacing_max]. Per voxel: select the point with max reflectance (same
    as preprocessing / sample_bright); if reflectance is all zero, scatter_max picks one
    representative per voxel (consecutive cluster). Use when training on dense voxels so
    the model sees sparser densities and generalizes to sparse scans (e.g. UK).
    """
    if pos is None or pos.size(0) == 0 or (prob <= 0 or torch.rand(1).item() >= prob):
        return pos, reflectance, label
    spacing = spacing_min + (spacing_max - spacing_min) * torch.rand(1).item()
    return _density_downsample_fixed_spacing(pos, reflectance, label, spacing)


def random_density_downsample_batch(data, spacing_min=0.01, spacing_max=0.04, prob=0.5, edge_voxel_size=0.25):
    """Apply one random spacing per batch so the whole batch has the same effective resolution.

    With probability prob, draw one spacing in [spacing_min, spacing_max] and downsample
    every sample in the batch with that same spacing. Recomputes sf and edge_scores.
    Use when the model is resolution-sensitive and batches must be single-resolution.
    """
    if prob <= 0 or torch.rand(1).item() >= prob:
        return
    spacing = spacing_min + (spacing_max - spacing_min) * torch.rand(1).item()
    batch_vec = data.batch
    device = data.pos.device
    pos_list, refl_list, y_list, sf_list = [], [], [], []
    offset = 0
    for b in batch_vec.unique(sorted=True):
        mask = batch_vec == b
        pos_b = data.pos[mask]
        refl_b = data.reflectance[mask]
        y_b = data.y[mask]
        pos_b, refl_b, y_b = _density_downsample_fixed_spacing(pos_b, refl_b, y_b, spacing)
        pos_list.append(pos_b)
        refl_list.append(refl_b)
        y_list.append(y_b)
        sf_b = torch.sqrt((pos_b ** 2).sum(dim=1)).max().clamp(min=1e-8)
        # Keep sf as one value per sample (not per point), matching DataLoader output and model indexing sf[batch].
        sf_list.append(sf_b)
    data.pos = torch.cat(pos_list, dim=0)
    data.reflectance = torch.cat(refl_list, dim=0)
    data.y = torch.cat(y_list, dim=0)
    data.sf = torch.stack(sf_list, dim=0)
    # Rebuild batch vector
    n_per = [p.size(0) for p in pos_list]
    data.batch = torch.cat([torch.full((n,), i, dtype=torch.long, device=device) for i, n in enumerate(n_per)], dim=0)
    # Recompute edge scores (per-sample mixed-label voxels)
    from src.pointcutmix import recompute_edge_scores
    data.edge_scores = recompute_edge_scores(data.pos, data.y, batch=data.batch, voxel_size=edge_voxel_size)


def rotate_3d(points):
    """Full yaw (360°) + small pitch/roll (±20°).

    Yaw: full rotation — tree canopy is rotationally symmetric around vertical.
    Pitch/roll: ±20° covers terrain slope, sensor tilt, and off-nadir scan angles
    without producing unrealistic upside-down trees.
    """
    device = points.device
    yaw   = torch.deg2rad(torch.rand(1, device=device) * 360.0)
    pitch = torch.deg2rad((torch.rand(1, device=device) - 0.5) * 40.0)  # ±20°
    roll  = torch.deg2rad((torch.rand(1, device=device) - 0.5) * 40.0)  # ±20°

    cy, sy = torch.cos(yaw).squeeze(),   torch.sin(yaw).squeeze()
    cp, sp = torch.cos(pitch).squeeze(), torch.sin(pitch).squeeze()
    cr, sr = torch.cos(roll).squeeze(),  torch.sin(roll).squeeze()

    zero = torch.zeros(1, device=device).squeeze()
    one  = torch.ones(1, device=device).squeeze()

    Rz = torch.stack([cy, -sy, zero,  sy,  cy, zero, zero, zero,  one]).view(3, 3)
    Ry = torch.stack([cp, zero,  sp, zero, one,  zero, -sp, zero,  cp]).view(3, 3)
    Rx = torch.stack([one, zero, zero, zero,  cr, -sr, zero,  sr,  cr]).view(3, 3)

    return points.view(-1, 3) @ (Rz @ Ry @ Rx).T

def random_scale_change(points, min_multiplier, max_multiplier):
    scale_factor = torch.FloatTensor(1).uniform_(min_multiplier, max_multiplier).to(points.device)
    return points * scale_factor

def jitter_points(points, std):
    noise = torch.normal(mean=0.0, std=std, size=points.size(), device=points.device)
    points = points + noise
    return points

def perturb_reflectance(feature):
    noise = torch.normal(mean=0.0, std=0.02, size=feature.size(), device=feature.device)
    feature = feature + noise
    return feature

    
def scale_shift_reflectance(reflectance, scale_range=(0.5, 1.5), shift_range=(-0.2, 0.2)):
    """Per-sample scale and shift of reflectance. Varies absolute level; contrast (deltas) preserved.
    Encourages the model to use relative reflectance, not absolute brightness."""
    scale = torch.empty(1, device=reflectance.device).uniform_(*scale_range).item()
    shift = torch.empty(1, device=reflectance.device).uniform_(*shift_range).item()
    result = reflectance * scale + shift
    return result.clamp(-1.0, 1.0)


def match_wood_reflectance_augmentation(reflectance, labels, ratio=0.5):
    """Copy wood reflectance onto leaf points. Simulates 'bright leaf that looks like wood' - the UK FP case.

    For `ratio` of leaf points, assign reflectance from a random wood point in this sample.
    Forces the model to rely on geometry when reflectance is misleading.
    """
    leaf_mask = labels < 0.5
    wood_mask = labels >= 0.5
    n_leaf = leaf_mask.sum().item()
    n_wood = wood_mask.sum().item()
    if n_leaf == 0 or n_wood == 0:
        return reflectance
    reflectance = reflectance.clone()
    n_match = max(1, int(ratio * n_leaf))
    leaf_indices = torch.where(leaf_mask)[0]
    wood_indices = torch.where(wood_mask)[0]
    selected_leaf = leaf_indices[torch.randperm(n_leaf, device=reflectance.device)[:n_match]]
    donors = wood_indices[torch.randint(n_wood, (n_match,), device=reflectance.device)]
    reflectance[selected_leaf] = reflectance[donors]
    return reflectance


def reflectance_outlier_augmentation(reflectance, labels, shift_range=(0.3, 0.6), outlier_ratio_range=(0.1, 0.3)):
    """Simulate scattered bright leaf returns from sensor/surface effects.

    Physical justification:
      - Specular reflection from waxy/glossy leaf surfaces
      - Leaf orientation perpendicular to beam returns more energy
      - Wet foliage increases reflectance
      - Multi-path effects in dense canopy
      - Near-range sensor saturation

    Randomly elevates reflectance of 10-30% of leaf points with per-point
    variation, creating scattered bright outliers that overlap with wood
    distribution. Teaches model to rely on geometry when reflectance is ambiguous.
    """
    leaf_mask = labels == 0

    if leaf_mask.sum() == 0:
        return reflectance

    reflectance = reflectance.clone()

    # Select random subset of leaves as "outliers"
    n_leaves = leaf_mask.sum().item()
    outlier_ratio = torch.empty(1, device=reflectance.device).uniform_(*outlier_ratio_range).item()
    n_outliers = max(1, int(n_leaves * outlier_ratio))

    leaf_indices = torch.where(leaf_mask)[0]
    perm = torch.randperm(n_leaves, device=reflectance.device)[:n_outliers]
    outlier_indices = leaf_indices[perm]

    # Brighten selected leaves with per-point variation (more realistic)
    shifts = torch.empty(n_outliers, device=reflectance.device).uniform_(*shift_range)
    reflectance[outlier_indices] = reflectance[outlier_indices] + shifts

    # Clamp to valid range (quantile-normalized reflectance is [-1, 1])
    reflectance = reflectance.clamp(-1.0, 1.0)

    return reflectance


def augmentations(pos, reflectance, label, mode: str = "train"):
    """Apply geometry and reflectance augmentations.

    Geometry augs are applied independently (can stack):
      30% 3D rotation (full yaw + ±20° pitch/roll), 30% scale (0.95-1.05x)

    Reflectance augs are mutually exclusive (one drawn per sample):
      20% zero, 10% match wood, 8% mild scale/shift, 7% leaf outliers,
      5% small additive noise, 50% clean.
    """

    # ---------------- Geometry branch (independent, can stack) ----------------
    if mode == "train":
        if torch.rand(1) < 0.3:
            pos = rotate_3d(pos)

        if torch.rand(1) < 0.3:
            pos = random_scale_change(pos, 0.90, 1.10)

        # Jitter removed: point clouds are noisy enough; extra spatial noise not needed.

    # ---------------- Reflectance branch (mutually exclusive) ----------------
    # Keep majority clean so model still learns true reflectance cues.
    # Add low-probability corruptions for robustness to site/sensor variation.
    if mode == "train":
        p_refl = torch.rand(1).item()

        if p_refl < 0.20:
            # (i) Dropout: forces geometry-only fallback for cross-sensor robustness
            reflectance = torch.zeros_like(reflectance)
        elif p_refl < 0.40:
            # (ii) Bright leaf simulation: specular/wet foliage returns that overlap wood distribution
            reflectance = reflectance_outlier_augmentation(
                reflectance,
                label,
                shift_range=(0.15, 0.50),
                outlier_ratio_range=(0.05, 0.25),
            )
        elif p_refl < 0.50:
            # (iii) Global scale/shift: inter-sensor calibration differences
            reflectance = scale_shift_reflectance(
                reflectance,
                scale_range=(0.85, 1.15),
                shift_range=(-0.08, 0.08),
            )
        # else: 50% clean — preserves true reflectance signal

    elif mode == "val_with_reflectance":
        pass

    elif mode == "val_no_reflectance":
        reflectance = torch.zeros_like(reflectance)

    elif mode == "test":
        pass

    return pos, reflectance, label
