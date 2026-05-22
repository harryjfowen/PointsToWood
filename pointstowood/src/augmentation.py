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


def random_density_downsample_batch(data, spacing_min=0.01, spacing_max=0.04, prob=0.5,
                                    edge_voxel_size=0.25, difficulty=None,
                                    hard_skip_threshold=0.75, difficulty_power=2.0):
    """Apply one random spacing per batch so the whole batch has the same effective resolution.

    With probability prob, draw one spacing in [spacing_min, spacing_max] and downsample
    every sample in the batch with that same spacing. Recomputes sf and edge_scores.
    Use when the model is resolution-sensitive and batches must be single-resolution.
    """
    effective_prob = float(prob)
    if difficulty is None:
        difficulty = getattr(data, 'difficulty', None)
    if difficulty is not None:
        d = difficulty.detach().float().view(-1)
        if d.numel() > 0:
            d = d[torch.isfinite(d)].clamp(0.0, 1.0)
            if d.numel() > 0:
                hardest = float(d.max().item())
                if hardest >= float(hard_skip_threshold):
                    return
                effective_prob *= max(0.0, 1.0 - hardest) ** float(difficulty_power)

    if effective_prob <= 0 or torch.rand(1).item() >= effective_prob:
        return

    native_voxel_size = getattr(data, 'voxel_size', None)
    native_spacing = None
    if isinstance(native_voxel_size, torch.Tensor) and native_voxel_size.numel() == 1:
        native_spacing = float(native_voxel_size.item()) / 100.0
    elif native_voxel_size is not None:
        native_spacing = float(native_voxel_size) / 100.0

    draw_min = max(float(spacing_min), native_spacing) if native_spacing is not None else float(spacing_min)
    draw_max = float(spacing_max)
    if draw_max < draw_min:
        return
    spacing = draw_min + (draw_max - draw_min) * torch.rand(1).item()

    batch_vec = data.batch
    device = data.pos.device
    pos_list, refl_list, y_list, sf_list = [], [], [], []
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
        sf_list.append(sf_b)
    data.pos = torch.cat(pos_list, dim=0)
    data.reflectance = torch.cat(refl_list, dim=0)
    data.y = torch.cat(y_list, dim=0)
    data.sf = torch.stack(sf_list, dim=0)
    n_per = [p.size(0) for p in pos_list]
    data.batch = torch.cat([torch.full((n,), i, dtype=torch.long, device=device) for i, n in enumerate(n_per)], dim=0)
    data.voxel_size = torch.tensor([float(spacing * 100.0)], dtype=torch.float32, device=device)
    from src.pointcutmix import recompute_edge_scores
    data.edge_scores = recompute_edge_scores(data.pos, data.y, batch=data.batch, voxel_size=edge_voxel_size)


def rotate_3d(points):
    """Full yaw (360°) + small pitch/roll (±10°).

    Yaw: trees are rotationally symmetric around vertical — every rotation is valid.
    Pitch/roll: ±10° covers terrain slope and off-nadir scan angles without producing
    unrealistic upside-down trees.
    """
    device = points.device
    yaw   = torch.deg2rad(torch.rand(1, device=device) * 360.0)
    pitch = torch.deg2rad((torch.rand(1, device=device) - 0.5) * 20.0)
    roll  = torch.deg2rad((torch.rand(1, device=device) - 0.5) * 20.0)

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
    return points + noise


def reflectance_constant_dropout(reflectance):
    """Replace all reflectance with a single constant c ~ U(-1, 1).

    Produces M_refl ≡ 0 (no local gradient), forcing geometry-only inference,
    without teaching the model that the value 0.0 specifically means 'no signal'.
    """
    c = torch.empty(1, device=reflectance.device).uniform_(-1.0, 1.0).item()
    return torch.full_like(reflectance, c)


def compress_reflectance_contrast(reflectance, alpha_range=(0.05, 1.0)):
    """Scale reflectance by α ~ U(alpha_range), collapsing wood/leaf separation.

    Simulates low-quality or non-calibrated scans (Cohen's d ~ 0.1–0.5 in German
    TLS plots). Forces the model to rely on geometry when reflectance is weak.
    """
    alpha = torch.empty(1, device=reflectance.device).uniform_(*alpha_range).item()
    return reflectance * alpha


def scale_shift_reflectance(reflectance, scale_range=(0.5, 1.5), shift_range=(-0.2, 0.2)):
    """Per-sample scale and shift. Preserves local contrast; varies absolute level.

    Covers scanner-to-scanner calibration offsets — same tree, different absolute
    intensity, same relative wood/leaf ordering.
    """
    scale = torch.empty(1, device=reflectance.device).uniform_(*scale_range).item()
    shift = torch.empty(1, device=reflectance.device).uniform_(*shift_range).item()
    return (reflectance * scale + shift).clamp(-1.0, 1.0)


def flip_reflectance(reflectance):
    """Negate all reflectance values.

    Simulates uncalibrated phase-shift TLS where raw intensity is not radiometrically
    corrected: bark at grazing incidence can appear darker than leaves, inverting the
    calibrated convention. Gated on Cohen's d ≥ 0.3 so it only fires when there is
    real contrast to invert.
    """
    return -reflectance


def reflectance_label_separation(reflectance, labels, min_class_points=32):
    """|Cohen's d| between wood and leaf reflectance distributions.

    Returns None when both classes are not sufficiently represented (pure samples,
    or under 32 points per class). Used to gate reflectance augmentation intensity:
    high d → preserve discriminative signal; low d or None → treat as uninformative.
    """
    leaf_mask = labels < 0.5
    wood_mask = labels >= 0.5
    if leaf_mask.sum().item() < min_class_points or wood_mask.sum().item() < min_class_points:
        return None
    leaf = reflectance[leaf_mask].float()
    wood = reflectance[wood_mask].float()
    pooled_var = 0.5 * (leaf.var(unbiased=False) + wood.var(unbiased=False))
    if pooled_var <= 1e-8:
        return 0.0
    return float((wood.mean() - leaf.mean()).abs() / pooled_var.sqrt())


def sparse_wood_augmentation(pos, reflectance, label, drop_range=(0.4, 0.75)):
    """Randomly drop wood points to simulate single-scan occlusion and thin-branch returns.

    TLS single-scan geometry (e.g. esp190 Quercus ilex) leaves far-side branches with
    only a handful of returns. Without this augmentation the model trains on dense wood
    clusters and fails on sparse thread-like structures in the field.
    """
    wood_mask = label >= 0.5
    n_wood = wood_mask.sum().item()
    if n_wood < 8:
        return pos, reflectance, label
    drop_frac = torch.empty(1).uniform_(*drop_range).item()
    n_drop = int(n_wood * drop_frac)
    if n_drop == 0:
        return pos, reflectance, label
    wood_indices = torch.where(wood_mask)[0]
    keep_indices = wood_indices[torch.randperm(n_wood, device=pos.device)[n_drop:]]
    leaf_indices = torch.where(~wood_mask)[0]
    all_keep = torch.cat([leaf_indices, keep_indices]).sort().values
    return pos[all_keep], reflectance[all_keep], label[all_keep]


def augmentations(pos, reflectance, label, mode: str = "train", edge_scores=None, difficulty: float = 0.0):
    """Geometry and reflectance augmentations for TLS wood/leaf segmentation.

    Geometry — four independent ops, physically motivated:
      Yaw (always): rotational symmetry of tree canopy.
      Scale (30%, 0.80–1.20×): tree size and scan-distance variation.
      Jitter (15%, 1–6 mm): phase-shift sensor noise; < 20% of SA1 radius (32 mm).
      Sparse wood (20%, mixed only, drop 40–75%): single-scan occlusion of thin branches.

    Reflectance — two regimes, four ops each, one Cohen's d gate:

      Hard mixed (difficulty ≥ 0.4): 80% clean — twig-in-leaf zones need real signal.
        10% constant dropout + 10% scale_shift cover uncalibrated hard-zone samples.

      All other samples: Cohen's d gates dropout intensity.
        refl_info = |Cohen's d| clamped to [0, 1]; None (pure samples) → 0.
        Dropout: 35% at refl_info=0 (uninformative) → ~11% at refl_info=1 (discriminative).
        Compress, scale_shift, flip each at fixed 10%. Flip gated at d ≥ 0.3.

    Reflectance dropout replaces values with a uniform constant c ~ U(-1, 1) rather
    than zeros. A constant input gives M_refl ≡ 0 (no local gradient) without
    teaching the model that 0.0 specifically encodes 'missing signal'.
    """

    if mode == "train":

        # ---- Geometry ----

        # Is this sample boundary-mixed? Used to gate sparse-wood to mixed samples only.
        if edge_scores is not None:
            is_mixed = edge_scores.float().mean().item() >= 0.15
        else:
            dominant_frac = max((label >= 0.5).float().mean().item(),
                               (label < 0.5).float().mean().item())
            is_mixed = dominant_frac <= 0.85

        pos = rotate_3d(pos)

        if torch.rand(1) < 0.3:
            pos = random_scale_change(pos, 0.80, 1.20)

        if torch.rand(1) < 0.15:
            std = torch.empty(1).uniform_(0.001, 0.006).item()
            pos = jitter_points(pos, std)

        # Sparse wood on mixed samples only: thin-branch occlusion occurs in the
        # context of surrounding leaf — dropping wood from pure-trunk samples is
        # a different (less realistic) physics.
        if is_mixed and torch.rand(1) < 0.20:
            pos, reflectance, label = sparse_wood_augmentation(pos, reflectance, label)

        # ---- Reflectance ----

        is_hard = difficulty >= 0.4

        if is_hard:
            # Hard twig-in-leaf zones: mostly clean. Small dropout and scale_shift slots
            # cover uncalibrated scanners where even hard-zone reflectance is unreliable.
            p = torch.rand(1).item()
            if p < 0.10:
                reflectance = reflectance_constant_dropout(reflectance)
            elif p < 0.20:
                reflectance = scale_shift_reflectance(reflectance)
            # else: clean (80%)

        else:
            # Cohen's d between wood and leaf in this sample. None for pure samples
            # (only one class present) → treat as uninformative (refl_info = 0).
            cohens_d = reflectance_label_separation(reflectance, label)
            refl_info = min(1.0, cohens_d) if cohens_d is not None else 0.0

            p = torch.rand(1).item()
            p_dropout  = 0.35 * (1.0 - 0.70 * refl_info)  # 35% → ~11%
            p_compress = p_dropout + 0.10
            p_scale    = p_compress + 0.10
            p_flip     = p_scale + 0.10  # gated: only fires when d ≥ 0.3

            if p < p_dropout:
                reflectance = reflectance_constant_dropout(reflectance)
            elif p < p_compress:
                reflectance = compress_reflectance_contrast(reflectance)
            elif p < p_scale:
                reflectance = scale_shift_reflectance(reflectance)
            elif p < p_flip and cohens_d is not None and cohens_d >= 0.3:
                reflectance = flip_reflectance(reflectance)
            # else: clean

    elif mode == "val_no_reflectance":
        reflectance = torch.zeros_like(reflectance)

    return pos, reflectance, label
