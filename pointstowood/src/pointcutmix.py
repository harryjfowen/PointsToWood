import torch
from torch_geometric.nn import voxel_grid
import torch_scatter


def pointcutmix(pos1, reflectance1, label1, pos2, reflectance2, label2, beta=1.0, method='spatial'):
    """
    PointCutMix augmentation for forest point clouds.

    Args:
        pos1, reflectance1, label1: First point cloud (e.g., wood-heavy voxel)
        pos2, reflectance2, label2: Second point cloud (e.g., leaf-heavy voxel)
        beta: Beta distribution parameter for mixing ratio
        method: 'spatial' (PointCutMix-K) or 'random' (PointCutMix-R)

    Returns:
        Mixed point cloud with realistic wood/leaf boundaries
    """
    device = pos1.device
    N = len(pos1)

    # Sample mixing ratio from Beta distribution
    lam = torch.distributions.Beta(beta, beta).sample().float().to(device)
    n_keep = int(lam * N)

    if n_keep == 0 or n_keep == N:
        return pos1, reflectance1, label1

    # Create binary mask for which points to keep from pos1
    mask = torch.zeros(N, dtype=torch.bool, device=device)

    if method == 'spatial':
        # PointCutMix-K: Keep spatially coherent region (good for wood branches)
        center_idx = torch.randint(0, N, (1,)).item()

        # Find k-nearest neighbors using euclidean distance
        distances = torch.norm(pos1 - pos1[center_idx], dim=1)
        _, nearest_indices = torch.topk(distances, n_keep, largest=False)
        mask[nearest_indices] = True

    else:  # method == 'random'
        # PointCutMix-R: Random selection
        random_indices = torch.randperm(N)[:n_keep]
        mask[random_indices] = True

    # Ensure both point clouds have same size (pad/subsample if needed)
    if len(pos2) != N:
        if len(pos2) > N:
            # Subsample pos2 to match N
            subset_indices = torch.randperm(len(pos2))[:N]
            pos2 = pos2[subset_indices]
            reflectance2 = reflectance2[subset_indices]
            label2 = label2[subset_indices]
        else:
            # Repeat pos2 to match N (for smaller voxels)
            repeat_factor = (N + len(pos2) - 1) // len(pos2)
            pos2 = pos2.repeat(repeat_factor, 1)[:N]
            reflectance2 = reflectance2.repeat(repeat_factor)[:N]
            label2 = label2.repeat(repeat_factor)[:N]

    # Mix point clouds
    mixed_pos = torch.where(mask.unsqueeze(1), pos1, pos2)
    mixed_reflectance = torch.where(mask, reflectance1, reflectance2)

    # Create spatially-explicit soft labels based on actual point origins
    # Points from pos1 keep their original labels, points from pos2 keep theirs
    mixed_label = torch.where(mask, label1.float(), label2.float())

    return mixed_pos, mixed_reflectance, mixed_label


def apply_pointcutmix_batch(batch_pos, batch_reflectance, batch_label, prob=0.5, beta=1.0, method='spatial'):
    """
    Apply PointCutMix to a batch by randomly pairing voxels.

    Args:
        batch_*: List of voxels in the batch
        prob: Probability of applying PointCutMix to each voxel
        beta: Beta distribution parameter
        method: 'spatial' or 'random' selection method

    Returns:
        Augmented batch with some PointCutMix samples
    """
    if len(batch_pos) < 2:
        return batch_pos, batch_reflectance, batch_label

    augmented_pos, augmented_reflectance, augmented_label = [], [], []

    for i in range(len(batch_pos)):
        # Only apply to mono-label samples (low variance)
        label_var = torch.var(batch_label[i].float())
        if torch.rand(1) < prob and label_var < 0.01:
            # Prefer mixing with opposite mono-label samples
            j_candidates = []
            current_is_wood = torch.mean(batch_label[i].float()) > 0.5

            for k in range(len(batch_pos)):
                if k != i:
                    other_var = torch.var(batch_label[k].float())
                    other_is_wood = torch.mean(batch_label[k].float()) > 0.5

                    # Prefer opposite mono samples
                    if other_var < 0.01 and other_is_wood != current_is_wood:
                        j_candidates.append(k)

            # Only mix if we found opposite mono samples
            if j_candidates:
                j = j_candidates[torch.randint(0, len(j_candidates), (1,)).item()]
            else:
                # No good pairing found, keep original
                augmented_pos.append(batch_pos[i])
                augmented_reflectance.append(batch_reflectance[i])
                augmented_label.append(batch_label[i])
                continue

            # Mix current voxel with random other voxel
            mixed_pos, mixed_refl, mixed_label = pointcutmix(
                batch_pos[i], batch_reflectance[i], batch_label[i],
                batch_pos[j], batch_reflectance[j], batch_label[j],
                beta=beta, method=method
            )

            augmented_pos.append(mixed_pos)
            augmented_reflectance.append(mixed_refl)
            augmented_label.append(mixed_label)
        else:
            # Keep original
            augmented_pos.append(batch_pos[i])
            augmented_reflectance.append(batch_reflectance[i])
            augmented_label.append(batch_label[i])

    return augmented_pos, augmented_reflectance, augmented_label


def recompute_edge_scores(pos, labels, voxel_size=0.25):
    """
    Recompute edge scores after PointCutMix based on new mixed labels.

    Args:
        pos: Mixed point positions
        labels: Mixed labels (can be soft)
        voxel_size: Size for voxel grid clustering

    Returns:
        edge_scores: New edge scores reflecting the mixed boundaries
    """
    # Create voxel clusters
    cluster = voxel_grid(pos, size=voxel_size, batch=None)

    # For each voxel, compute proportion of wood points (label > 0.5)
    wood_points = (labels > 0.5).float()
    pos_sum = torch_scatter.scatter_add(wood_points, cluster, dim=0)
    count = torch_scatter.scatter_add(torch.ones_like(labels), cluster, dim=0)
    pos_prop = pos_sum / (count + 1e-6)

    # Edge scores: voxels with mixed content (between 0 and 1)
    edge_scores = ((pos_prop[cluster] > 0) & (pos_prop[cluster] < 1)).float()

    return edge_scores


def apply_edge_aware_label_smoothing(labels, edge_scores, smoothing_factor=0.1):
    """
    Apply classical label smoothing with intensity based on edge scores.

    Args:
        labels: Binary labels (0 or 1)
        edge_scores: Edge uncertainty scores (0 = clear, 1 = boundary)
        smoothing_factor: Maximum smoothing amount (0.1 = up to 10% smoothing)

    Returns:
        Smoothed labels where boundary regions get more smoothing
    """
    # Edge-dependent smoothing amount
    smooth_amount = edge_scores * smoothing_factor

    # Classical label smoothing:
    # Wood (1) smoothed toward leaf (0): 1 → 1-smooth_amount
    # Leaf (0) smoothed toward wood (1): 0 → smooth_amount
    smoothed_labels = labels * (1 - smooth_amount) + (1 - labels) * smooth_amount

    return smoothed_labels