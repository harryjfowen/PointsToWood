import torch
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
import torch_scatter


def pointcutmix_insert(leaf_pos, leaf_refl, leaf_label, wood_pos, wood_refl, wood_label, insert_scale=1.0, max_points=None, carve_margin=0.02):
    """
    Insert wood-dominant cloud inside leaf-dominant cloud in the same 3D frame.
    Genuine interlacing: leaf points define the container; wood points are transformed
    to lie inside that region (same coordinate system).

    Args:
        leaf_*: Leaf-dominant voxel (container).
        wood_*: Wood-dominant voxel (insert).
        insert_scale: Scale wood to this fraction of leaf extent (1.0 = same extent).
                      None or <= 0 = no scaling, only translate (center wood at leaf center).
        max_points: If set, subsample mixed cloud to this size (keeps ratio roughly).
        carve_margin: If > 0, voxel carve: remove leaf points in voxels that contain wood.
                      carve_margin is the voxel size (m). Only carves wood-occupied voxels;
                      handles complex branches (Y-shaped, curved) without over-removing.

    Returns:
        mixed_pos, mixed_refl, mixed_label (all in leaf's coordinate frame).
    """
    device = leaf_pos.device
    leaf_center = leaf_pos.mean(dim=0)
    leaf_extent = (leaf_pos - leaf_center).abs().max()
    if leaf_extent < 1e-8:
        leaf_extent = 1.0

    wood_center = wood_pos.mean(dim=0)
    wood_extent = (wood_pos - wood_center).abs().max()
    if wood_extent < 1e-8:
        wood_extent = 1.0

    # Place wood in leaf frame: translate to leaf center; optionally scale to fit
    if insert_scale is not None and insert_scale > 0:
        wood_pos_in_leaf = (wood_pos - wood_center) * (leaf_extent * insert_scale / wood_extent) + leaf_center
    else:
        wood_pos_in_leaf = (wood_pos - wood_center) + leaf_center

    # Carve: remove leaf points in voxels that contain wood (branch displaces foliage)
    if carve_margin is not None and carve_margin > 0:
        all_pos = torch.cat([leaf_pos, wood_pos_in_leaf], dim=0)
        voxel_ids = voxel_grid(all_pos, carve_margin, batch=None)
        voxel_ids, _ = consecutive_cluster(voxel_ids)
        n_leaf = leaf_pos.size(0)
        leaf_voxels = voxel_ids[:n_leaf]
        wood_voxels = voxel_ids[n_leaf:]
        wood_voxel_ids = wood_voxels.unique()
        leaf_keep = ~torch.isin(leaf_voxels, wood_voxel_ids)
        leaf_pos = leaf_pos[leaf_keep]
        leaf_refl = leaf_refl[leaf_keep]
        leaf_label = leaf_label[leaf_keep]
        if leaf_pos.size(0) == 0:
            return wood_pos_in_leaf, wood_refl, wood_label.float()

    mixed_pos = torch.cat([leaf_pos, wood_pos_in_leaf], dim=0)
    mixed_refl = torch.cat([leaf_refl, wood_refl], dim=0)
    mixed_label = torch.cat([leaf_label.float(), wood_label.float()], dim=0)

    if max_points is not None and mixed_pos.size(0) > max_points:
        perm = torch.randperm(mixed_pos.size(0), device=device)
        idx = perm[:max_points]
        mixed_pos = mixed_pos[idx]
        mixed_refl = mixed_refl[idx]
        mixed_label = mixed_label[idx]

    return mixed_pos, mixed_refl, mixed_label


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
        center_idx = torch.randint(0, N, (1,), device=device).item()
        distances = torch.norm(pos1 - pos1[center_idx], dim=1)
        _, nearest_indices = torch.topk(distances, n_keep, largest=False)
        mask[nearest_indices] = True

    else:  # method == 'random'
        # PointCutMix-R: Random selection (device for correct indexing on GPU)
        random_indices = torch.randperm(N, device=device)[:n_keep]
        mask[random_indices] = True

    # Ensure both point clouds have same size (pad/subsample if needed)
    if len(pos2) != N:
        if len(pos2) > N:
            subset_indices = torch.randperm(len(pos2), device=device)[:N]
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


def apply_pointcutmix_batch(batch_pos, batch_reflectance, batch_label, prob=0.5, beta=1.0, method='insert', insert_scale=1.0, max_points=None, carve_margin=0.02):
    """
    Apply PointCutMix to a batch by randomly pairing voxels.

    Args:
        batch_*: List of voxels in the batch
        prob: Probability of applying PointCutMix to each voxel
        beta: Beta distribution parameter (for method 'spatial' / 'random')
        method: 'insert' = wood inside leaf in same 3D frame (genuine interlacing);
                'spatial' / 'random' = original two-blob mix
        insert_scale: For 'insert', scale wood to this fraction of leaf extent (default 1.0)
        max_points: For 'insert', cap mixed sample size (default None = keep all)

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

            if method == 'insert':
                # Insert wood-dominant inside leaf-dominant (same 3D frame)
                leaf_is_i = not current_is_wood  # current i is wood -> j is leaf
                if leaf_is_i:
                    leaf_pos, leaf_refl, leaf_label = batch_pos[i], batch_reflectance[i], batch_label[i]
                    wood_pos, wood_refl, wood_label = batch_pos[j], batch_reflectance[j], batch_label[j]
                else:
                    leaf_pos, leaf_refl, leaf_label = batch_pos[j], batch_reflectance[j], batch_label[j]
                    wood_pos, wood_refl, wood_label = batch_pos[i], batch_reflectance[i], batch_label[i]
                mixed_pos, mixed_refl, mixed_label = pointcutmix_insert(
                    leaf_pos, leaf_refl, leaf_label,
                    wood_pos, wood_refl, wood_label,
                    insert_scale=insert_scale, max_points=max_points, carve_margin=carve_margin
                )
            else:
                # Original: two-blob mix (spatial or random)
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


def recompute_edge_scores(pos, labels, batch=None, voxel_size=0.25):
    """
    Recompute edge scores after PointCutMix based on new mixed labels.

    Args:
        pos: Mixed point positions [N, 3]
        labels: Mixed labels (can be soft) [N]
        batch: Optional batch vector [N] (required when pos/labels are batched)
        voxel_size: Size for voxel grid clustering

    Returns:
        edge_scores: New edge scores reflecting the mixed boundaries [N]
    """
    if batch is not None:
        # Per-sample voxelization so we don't mix points from different graphs
        out_list = []
        for b in batch.unique(sorted=True):
            mask = batch == b
            pos_b, labels_b = pos[mask], labels[mask]
            edge_b = _recompute_edge_scores_single(pos_b, labels_b, voxel_size)
            out_list.append(edge_b)
        return torch.cat(out_list, dim=0)
    return _recompute_edge_scores_single(pos, labels, voxel_size)


def _recompute_edge_scores_single(pos, labels, voxel_size=0.25):
    """Edge scores for a single point cloud (no batch)."""
    cluster = voxel_grid(pos, size=voxel_size, batch=None)
    wood_points = (labels > 0.5).float()
    pos_sum = torch_scatter.scatter_add(wood_points, cluster, dim=0)
    count = torch_scatter.scatter_add(torch.ones_like(labels), cluster, dim=0)
    pos_prop = pos_sum / (count + 1e-6)
    edge_scores = ((pos_prop[cluster] > 0) & (pos_prop[cluster] < 1)).float()
    return edge_scores

