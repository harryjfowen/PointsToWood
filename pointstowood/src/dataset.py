import os
import glob
import random
import re
import torch
import numpy as np
from abc import ABC
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Sampler
from torch_geometric.nn import voxel_grid
import torch_scatter
from src.augmentation import augmentations
from src.pointcutmix import pointcutmix_insert, recompute_edge_scores


def _parse_group(key: str) -> str:
    """Extract biome group from voxel filename prefix.
    e.g. 'fin01_lw_pl_3_voxel_5.pt' -> 'fin'
         'wood01_voxel_0.pt'         -> 'wood'
         'spa24_lw_voxel_2.pt'       -> 'spa'
    """
    token = os.path.basename(key).split('_voxel_')[0].split('_')[0]
    return re.sub(r'\d+$', '', token).lower() or 'unknown'

def sor_filter(pos, reflectance=None, y=None, edge_scores=None, k=16, std_threshold=1.0):
    try:
        from sklearn.neighbors import KDTree
    except ImportError:
        print("Warning: sklearn not available, skipping denoising")
        return pos, reflectance, y, edge_scores
    
    pos_np = pos.cpu().numpy()
    tree = KDTree(pos_np)
    distances, _ = tree.query(pos_np, k=k)
    mean_distances = np.mean(distances, axis=1)
    mean = np.mean(mean_distances)
    std = np.std(mean_distances)
    threshold = mean + std_threshold * std
    mask = mean_distances < threshold
    
    pos_filtered = pos[mask]
    reflectance_filtered = reflectance[mask] if reflectance is not None else None
    y_filtered = y[mask] if y is not None else None
    edge_scores_filtered = edge_scores[mask] if edge_scores is not None else None
    
    return pos_filtered, reflectance_filtered, y_filtered, edge_scores_filtered


def _load_voxel_file(path, weights_only=True):
    """Load a voxel .pt file; support both dict {'point_cloud', 'grid_size'} and legacy tensor. Returns (point_cloud_tensor, grid_size or None)."""
    data = torch.load(path, map_location='cpu', weights_only=weights_only)
    if isinstance(data, dict) and 'point_cloud' in data:
        return data['point_cloud'], data.get('grid_size')
    return data, None


def _point_count_from_pc(pc):
    """Get number of points from loaded point cloud. Handles (N,C) and edge cases. Points are rows (dim 0)."""
    if pc is None:
        return 0
    if hasattr(pc, 'shape'):
        sh = pc.shape
        if len(sh) >= 2:
            return int(sh[0])  # (N, C) -> N points
        if len(sh) == 1:
            # Flattened (N*C,): assume 5 cols, return N (conservative)
            n = int(sh[0])
            if n % 5 == 0:
                return n // 5
            return n  # fallback: treat as points
        return 0
    return 0


class TrainingDataset(Dataset, ABC):
    def __init__(self, voxels, augmentation, mode, max_pts, device, denoise=False, denoise_k=16, denoise_std=1.0, pointcutmix=True, pointcutmix_prob=0.25, pointcutmix_refl_dropout=0.0, density_aug=False, density_aug_prob=0.5, density_aug_spacing_min=0.01, density_aug_spacing_max=0.04):
        if not voxels:
            raise ValueError("The 'voxels' parameter cannot be empty.")
        self.voxels = voxels
        self.keys = sorted(glob.glob(os.path.join(voxels, '*.pt')))
        self.device = device
        self.max_pts = max_pts
        self.reflectance_index = 3
        self.label_index = 4
        self.augmentation = augmentation
        self.mode = mode
        self.labels = []
        self.voxel_size = 0.25

        self.denoise = denoise
        self.denoise_k = denoise_k
        self.denoise_std = denoise_std
        self.pointcutmix = pointcutmix
        self.pointcutmix_prob = pointcutmix_prob
        self.pointcutmix_refl_dropout = pointcutmix_refl_dropout
        self.density_aug = density_aug
        self.density_aug_prob = density_aug_prob
        self.density_aug_spacing_min = density_aug_spacing_min
        self.density_aug_spacing_max = density_aug_spacing_max

        if self.denoise:
            print(f"Denoising enabled with k={denoise_k}, std_threshold={denoise_std}")
        self._eligible_leaf = []
        self._eligible_wood = []
        self.grid_sizes = []  # per real index; used for resolution-aware batching (None = legacy file)
        for idx, key in enumerate(self.keys):
            point_cloud, grid_size = _load_voxel_file(key)
            self.grid_sizes.append(grid_size)
            y = point_cloud[:, self.label_index]
            sample_label = 1 if (y > 0.50).sum() > len(y) / 2 else 0
            self.labels.append(sample_label)
            if self.pointcutmix and self.mode == 'train':
                n = point_cloud.shape[0]
                cap = self.max_pts if self.max_pts > 0 else 50000
                half = cap // 2
                if sample_label == 0 and n <= half:
                    self._eligible_leaf.append(idx)
                elif sample_label == 1 and n <= half:
                    self._eligible_wood.append(idx)

        # Group indices for GroupDRO — parsed from filename prefix
        self.groups = [_parse_group(k) for k in self.keys]
        self.group_list = sorted(set(self.groups))
        _g2i = {g: i for i, g in enumerate(self.group_list)}
        self.group_indices = [_g2i[g] for g in self.groups]

        if self.pointcutmix and self.mode == 'train':
            self._indices_by_label = {
                0: [i for i, l in enumerate(self.labels) if l == 0],
                1: [i for i, l in enumerate(self.labels) if l == 1]
            }
            # Extra dataset slots for mixed samples (appended, not replacing). More batches/samples per epoch.
            self._num_mix_slots = int(round(pointcutmix_prob * len(self.keys)))
            pct = int(round(pointcutmix_prob * 100))
            refl_str = f", refl_dropout={pointcutmix_refl_dropout}" if pointcutmix_refl_dropout > 0 else ""
            print(f"PointCutMix: ~{pct}% mixes (append){refl_str}; {len(self._eligible_leaf)} leaf / {len(self._eligible_wood)} wood eligible; +{self._num_mix_slots} mix slots/epoch")
        else:
            self._num_mix_slots = 0

    def __len__(self):
        return len(self.keys) + getattr(self, '_num_mix_slots', 0)

    def _load_one_no_aug(self, index):
        """Load one voxel: denoise, subsample only. No augmentation. Returns (pos, reflectance, y)."""
        point_cloud, _ = _load_voxel_file(self.keys[index])
        pos = torch.as_tensor(point_cloud[:, :3], dtype=torch.float).requires_grad_(False)
        reflectance = torch.as_tensor(point_cloud[:, self.reflectance_index], dtype=torch.float)
        y = torch.as_tensor(point_cloud[:, self.label_index], dtype=torch.float)
        if self.denoise:
            pos, reflectance, y, _ = sor_filter(pos, reflectance, y, None, self.denoise_k, self.denoise_std)
        if self.max_pts > 0 and len(pos) > self.max_pts:
            indices = torch.randperm(len(pos))[:self.max_pts]
            pos = pos[indices]
            reflectance = reflectance[indices]
            y = y[indices]
        return pos, reflectance, y

    def __getitem__(self, index):
        n_real = len(self.keys)
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

        # PointCutMix (append): indices [n_real, len(self)) are mix-only slots; return mixed sample.
        if self.mode == 'train' and self.pointcutmix and index >= n_real and self._eligible_leaf and self._eligible_wood:
            li = random.choice(self._eligible_leaf)
            wi = random.choice(self._eligible_wood)
            leaf_pos, leaf_refl, leaf_label = self._load_one_no_aug(li)
            wood_pos, wood_refl, wood_label = self._load_one_no_aug(wi)
            max_pts_mix = self.max_pts if self.max_pts > 0 else None  # 0 = no subsampling
            pos, reflectance, y = pointcutmix_insert(
                leaf_pos, leaf_refl, leaf_label,
                wood_pos, wood_refl, wood_label,
                insert_scale=1.0, max_points=max_pts_mix, carve_margin=0.02
            )
            local_shift = torch.mean(pos[:, :3], axis=0).requires_grad_(False)
            pos = pos - local_shift
            scaling_factor = torch.sqrt((pos ** 2).sum(dim=1)).max()
            if scaling_factor < 1e-8:
                scaling_factor = torch.tensor(1.0, dtype=pos.dtype)
            edge_scores = recompute_edge_scores(pos, y, batch=None, voxel_size=self.voxel_size)
            out_mix = Data(pos=pos, reflectance=reflectance, y=y, sf=scaling_factor, edge_scores=edge_scores)
            out_mix.group_idx = torch.tensor([0], dtype=torch.long)  # placeholder: mix samples not group-attributed
            return out_mix

        # Standard path: load one voxel (real sample only; mixes come from extra indices above)
        point_cloud, grid_size = _load_voxel_file(self.keys[index])
        pos = torch.as_tensor(point_cloud[:, :3], dtype=torch.float).requires_grad_(False)
        reflectance = torch.as_tensor(point_cloud[:, self.reflectance_index], dtype=torch.float)
        y = torch.as_tensor(point_cloud[:, self.label_index], dtype=torch.float)
        
        if self.mode == 'train' and point_cloud.shape[-1] > 5:
            edge_scores_precomputed = torch.as_tensor(point_cloud[:, 5], dtype=torch.float)
        else:
            edge_scores_precomputed = None
        
        if self.denoise:
            pos, reflectance, y, edge_scores_precomputed = sor_filter(pos, reflectance, y, edge_scores_precomputed, self.denoise_k, self.denoise_std)
        
            if edge_scores_precomputed is not None and edge_scores_precomputed.size(0) != pos.size(0):
                edge_scores_precomputed = None
        
        if self.max_pts > 0 and len(pos) > self.max_pts:
            indices = torch.randperm(len(pos))[:self.max_pts]
            pos = pos[indices]
            reflectance = reflectance[indices]
            y = y[indices]
            if edge_scores_precomputed is not None:
                edge_scores_precomputed = edge_scores_precomputed[indices]
        
        if self.augmentation:
            pos, reflectance, y = augmentations(pos, reflectance, y, self.mode)

        local_shift = torch.mean(pos[:, :3], axis=0).requires_grad_(False)
        pos = pos - local_shift
        scaling_factor = torch.sqrt((pos ** 2).sum(dim=1)).max().clamp(min=1e-8)

        if torch.any(torch.isnan(reflectance)):
            print('nans in relfectance')

        if edge_scores_precomputed is not None:
            edge_scores = edge_scores_precomputed
        else:
            cluster = voxel_grid(pos, size=self.voxel_size, batch=None)
            pos_sum = torch_scatter.scatter_add((y == 1).float(), cluster, dim=0)
            count = torch_scatter.scatter_add(torch.ones_like(y), cluster, dim=0)
            pos_prop = pos_sum / (count + 1e-6)
            edge_scores = ((pos_prop[cluster] > 0) & (pos_prop[cluster] < 1)).float()

        out = Data(
            pos=pos,
            reflectance=reflectance,
            y=y,
            sf=scaling_factor,
            edge_scores=edge_scores
        )
        if grid_size is not None:
            out.grid_size = grid_size
        out.group_idx = torch.tensor([self.group_indices[index]], dtype=torch.long)
        return out

class TestingDataset(Dataset, ABC):
    def __init__(self, voxels, max_pts, device, in_memory=False, denoise=False, denoise_k=16, denoise_std=1.0, file_pattern=None):
        if not voxels:
            raise ValueError("The 'voxels' parameter cannot be empty.")
        self.voxels = voxels
        # Use file_pattern if provided, otherwise load all .pt files
        if file_pattern:
            self.keys = sorted(glob.glob(os.path.join(voxels, file_pattern)))
        else:
            self.keys = sorted(glob.glob(os.path.join(voxels, '*.pt')))
        self.device = device
        self.max_pts = max_pts
        self.reflectance_index = 3
        
        self.denoise = denoise
        self.denoise_k = denoise_k
        self.denoise_std = denoise_std
        if self.denoise:
            print(f"Denoising enabled with k={denoise_k}, std_threshold={denoise_std}")

    def __len__(self):
        return len(self.keys)  

    def __getitem__(self, index):
        point_cloud, grid_size = _load_voxel_file(self.keys[index])
        pos = torch.as_tensor(point_cloud[:, :3], dtype=torch.float).requires_grad_(False)
        reflectance = torch.as_tensor(point_cloud[:, self.reflectance_index], dtype=torch.float)

        if self.denoise:
            pos, reflectance, _, _ = sor_filter(pos, reflectance, k=self.denoise_k, std_threshold=self.denoise_std)
        
        if self.max_pts > 0 and len(pos) > self.max_pts:
            indices = torch.randperm(len(pos))[:self.max_pts]
            pos = pos[indices]
            reflectance = reflectance[indices]
        
        local_shift = torch.mean(pos[:, :3], axis=0).requires_grad_(False)
        pos = pos - local_shift
        scaling_factor = torch.sqrt((pos ** 2).sum(dim=1)).max().clamp(min=1e-8)

        nan_mask = torch.isnan(pos).any(dim=1) | torch.isnan(reflectance)
        pos = pos[~nan_mask]
        reflectance = reflectance[~nan_mask]

        if nan_mask.any(): 
            print(f"Encountered NaN values in sample at index {index}")
        
        data = Data(pos=pos, reflectance=reflectance, local_shift=local_shift, sf=scaling_factor)
        if grid_size is not None:
            data.grid_size = grid_size
        return data

class BalanceClassSampler(Sampler):
    def __init__(self, labels, mode="downsampling"):
        super().__init__(labels)
        labels = np.array(labels)
        samples_per_class = {label: (labels == label).sum() for label in set(labels)}
        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(mode, str):
            assert mode in ["downsampling", "upsampling"]

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = (
                mode if isinstance(mode, int) else max(samples_per_class.values())
            )
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))

    def __iter__(self):
        indices = []
        for key in sorted(self.lbl2idx):
            replace_flag = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_flag
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.length


def _fixed_batch_collate(samples):
    """Collate for fixed batch_size path: build batch and set voxel_size when all samples share same grid_size."""
    batch = Batch.from_data_list(samples)
    grid_sizes = [getattr(s, 'grid_size', None) for s in samples]
    if grid_sizes and all(g is not None and g == grid_sizes[0] for g in grid_sizes):
        batch.voxel_size = torch.tensor([float(grid_sizes[0])], dtype=torch.float32)
    return batch


class BalancedPointBudgetSampler(Sampler):
    """
    Point-budget-aware batching for training.
    packing_mode: "ffd" = First-Fit Decreasing, "bfd" = Best-Fit Decreasing (tighter, flatter),
                  "balanced" = class-balanced best-fit (round-robin wood/leaf/mix),
                  "balanced_bfd" = class-aware BFD + utilization smoothing.
    """
    def __init__(self, dataset, labels, target_points_per_batch=50000, min_points_per_batch=16000, mode="downsampling", packing_mode="bfd", verbose=True):
        self.dataset = dataset
        self.labels = np.array(labels)
        self.target_points = target_points_per_batch
        self.min_points = min_points_per_batch  # Avoid tiny batches; allow overflow, collate will downsample
        self.mode = mode
        self.packing_mode = packing_mode
        self.verbose = verbose

        # Point counts for all indices (real + mix slots when PointCutMix append is on)
        # max_pts <= 0 or >= 100M means "no subsampling" (use full voxels); otherwise cap for packing
        NO_LIMIT = 100_000_000
        self._use_raw_counts = dataset.max_pts <= 0 or dataset.max_pts >= NO_LIMIT
        effective_max = 999999999 if self._use_raw_counts else min(dataset.max_pts, max(self.target_points, 50000))
        n_real = len(dataset.keys)
        n_total = len(dataset)
        self.point_counts = []
        self._raw_point_counts = []  # for diagnostic
        for idx in range(n_real):
            try:
                pc, _ = _load_voxel_file(dataset.keys[idx], weights_only=True)
                raw_count = _point_count_from_pc(pc)
                self._raw_point_counts.append(raw_count)
                count = min(raw_count, effective_max)
                self.point_counts.append(count)
                if self.verbose and idx == 0 and pc is not None and hasattr(pc, 'shape'):
                    print(f"First voxel shape: {pc.shape} -> {raw_count} points (effective cap {effective_max:,})")
            except Exception:
                self._raw_point_counts.append(0)
                self.point_counts.append(effective_max // 2)  # Fallback
        for idx in range(n_real, n_total):
            self.point_counts.append(effective_max)  # Mix slots

        # Create class-balanced index pools (real indices only)
        self.lbl2idx = {
            label: np.arange(len(labels))[self.labels == label].tolist()
            for label in set(labels)
        }

        # Determine samples per class
        samples_per_class = {label: len(idxs) for label, idxs in self.lbl2idx.items()}
        if mode == "upsampling":
            self.samples_per_class = max(samples_per_class.values())
        else:
            self.samples_per_class = min(samples_per_class.values())

        self.n_real = n_real
        self.n_total = n_total
        if packing_mode == "bfd":
            self.batches = self._create_bfd_batches()
        elif packing_mode == "ffd":
            self.batches = self._create_ffd_batches()
        elif packing_mode == "balanced":
            self.batches = self._create_balanced_batches()
        elif packing_mode == "balanced_bfd":
            self.batches = self._create_balanced_bfd_batches()
        else:
            raise ValueError(f"Unknown packing_mode: {packing_mode}")

    def _create_bfd_batches(self):
        """Best-Fit Decreasing: sort by size descending, place in bin with smallest remainder that fits."""
        batches = []
        grid_sizes = getattr(self.dataset, 'grid_sizes', None)
        n_real = self.n_real
        n_total = self.n_total

        if grid_sizes is not None and len(grid_sizes) >= n_real:
            res2idx = {}
            for idx in range(n_real):
                res = grid_sizes[idx]
                res2idx.setdefault(res, []).append(idx)
            if n_total > n_real:
                res2idx.setdefault(None, []).extend(range(n_real, n_total))
        else:
            res2idx = {None: list(range(n_total))}

        for res, indices in res2idx.items():
            if not indices:
                continue
            items = sorted([(i, self.point_counts[i]) for i in indices], key=lambda x: -x[1])
            res_batches = []
            batch_pts = []

            for idx, pts in items:
                best_b = -1
                best_remainder = self.target_points + 1
                for b in range(len(res_batches)):
                    remainder = self.target_points - (batch_pts[b] + pts)
                    if 0 <= remainder < best_remainder:
                        best_remainder = remainder
                        best_b = b
                if best_b >= 0:
                    res_batches[best_b].append(idx)
                    batch_pts[best_b] += pts
                else:
                    res_batches.append([idx])
                    batch_pts.append(pts)

            self._merge_small_batches(res_batches, batch_pts)
            batches.extend(res_batches)

        self._merge_small_batches(batches, None)  # final pass, recompute pts

        if self.verbose:
            self._print_batch_stats(batches)
        return batches

    def _merge_small_batches(self, batches, batch_pts_opt, min_util=0.5, max_merge_ratio=1.05):
        """Merge small batches only when result stays near target (avoids oversized + undersized combo)."""
        target = self.target_points
        max_merge = target * max_merge_ratio

        i = 1
        while i < len(batches):
            bp_i = sum(self.point_counts[j] for j in batches[i]) if batch_pts_opt is None else batch_pts_opt[i]
            if bp_i < target * min_util:
                best_j = -1
                best_diff = target + 1  # prefer merge that gets closest to target
                for j in range(i):
                    bp_j = sum(self.point_counts[k] for k in batches[j]) if batch_pts_opt is None else batch_pts_opt[j]
                    total = bp_j + bp_i
                    if total <= max_merge:
                        diff = abs(total - target)
                        if diff < best_diff:
                            best_diff = diff
                            best_j = j
                if best_j >= 0:
                    batches[best_j].extend(batches[i])
                    if batch_pts_opt is not None:
                        batch_pts_opt[best_j] += bp_i
                        batch_pts_opt.pop(i)
                    batches.pop(i)
                    continue
            i += 1

        # First batch tiny: merge into batch that gets closest to target
        while len(batches) > 1:
            bp0 = sum(self.point_counts[j] for j in batches[0])
            if bp0 >= target * min_util:
                break
            best_j = -1
            best_diff = target + 1
            for j in range(1, len(batches)):
                bp_j = sum(self.point_counts[k] for k in batches[j])
                total = bp_j + bp0
                if total <= max_merge:
                    diff = abs(total - target)
                    if diff < best_diff:
                        best_diff = diff
                        best_j = j
            if best_j >= 0:
                batches[best_j].extend(batches[0])
                if batch_pts_opt is not None and len(batch_pts_opt) > best_j:
                    batch_pts_opt[best_j] += bp0
                    batch_pts_opt.pop(0)
                batches.pop(0)
            else:
                break

    def _create_ffd_batches(self):
        """First-Fit Decreasing: sort by size descending, pack into first batch that fits. Resolution-aware."""
        batches = []
        grid_sizes = getattr(self.dataset, 'grid_sizes', None)
        n_real = self.n_real
        n_total = self.n_total

        if grid_sizes is not None and len(grid_sizes) >= n_real:
            res2idx = {}
            for idx in range(n_real):
                res = grid_sizes[idx]
                res2idx.setdefault(res, []).append(idx)
            if n_total > n_real:
                res2idx.setdefault(None, []).extend(range(n_real, n_total))
        else:
            res2idx = {None: list(range(n_total))}

        for res, indices in res2idx.items():
            if not indices:
                continue
            # (idx, pts) sorted descending
            items = sorted([(i, self.point_counts[i]) for i in indices], key=lambda x: -x[1])
            res_batches = []
            batch_pts = []  # current points per batch

            for idx, pts in items:
                placed = False
                for b in range(len(res_batches)):
                    if batch_pts[b] + pts <= self.target_points:
                        res_batches[b].append(idx)
                        batch_pts[b] += pts
                        placed = True
                        break
                if not placed:
                    res_batches.append([idx])
                    batch_pts.append(pts)

            self._merge_small_batches(res_batches, batch_pts)
            batches.extend(res_batches)

        self._merge_small_batches(batches, None)
        if self.verbose:
            self._print_batch_stats(batches)
        return batches

    def _print_batch_stats(self, batches):
        """Print point-count and batch utilization stats."""
        if not self.verbose:
            return
        raw_counts = getattr(self, '_raw_point_counts', [])
        max_raw = max(raw_counts) if raw_counts else 0
        max_effective = max(self.point_counts) if self.point_counts else 0
        use_raw = getattr(self, '_use_raw_counts', False)
        effective_max = 999999999 if use_raw else min(self.dataset.max_pts, max(self.target_points, 50000))
        print(f"Point counts: max raw (on-disk)={max_raw:,}, max effective (for packing)={max_effective:,}, cap={effective_max:,} (dataset.max_pts={self.dataset.max_pts:,})")
        batch_sizes = [len(b) for b in batches]
        batch_points = [sum(self.point_counts[i] for i in b) for b in batches]
        utilization = [bp / self.target_points for bp in batch_points]
        over_target = [bp for bp in batch_points if bp > self.target_points * 2]
        if over_target:
            print(f"WARNING: Some batches exceed 2x target (max {max(batch_points):,}); collate downsamples to {self.target_points:,}")
        mode_str = {
            "ffd": "FFD",
            "bfd": "BFD",
            "balanced": "BALANCED",
            "balanced_bfd": "BALANCED-BFD",
        }.get(self.packing_mode, "custom")
        print(f"Point-budget ({mode_str}): {len(batches)} batches (resolution-aware), {min(batch_sizes)}–{max(batch_sizes)} samples/batch (avg {np.mean(batch_sizes):.1f}), "
              f"{min(batch_points):,.0f}–{max(batch_points):,.0f} pts pre-collate (collate caps to {self.target_points:,}), "
              f"utilization {min(utilization):.0%}–{max(utilization):.0%} (avg {np.mean(utilization):.0%})")

    def _label_of(self, idx):
        if idx < len(self.labels):
            return int(self.labels[idx])
        return -1  # mix slot

    def _balanced_indices_for_resolution(self, indices):
        """Sample class-balanced real indices for this resolution bucket; append mix slots as-is."""
        real = [i for i in indices if i < self.n_real]
        mix = [i for i in indices if i >= self.n_real]
        if not real:
            return mix

        lbl2idx = {}
        for idx in real:
            lbl = int(self.labels[idx])
            lbl2idx.setdefault(lbl, []).append(idx)

        if not lbl2idx:
            return real + mix

        if self.mode == "upsampling":
            n_per = max(len(v) for v in lbl2idx.values())
        else:
            n_per = min(len(v) for v in lbl2idx.values())

        selected = []
        for lbl, idxs in lbl2idx.items():
            chosen = np.random.choice(idxs, n_per, replace=len(idxs) < n_per).tolist()
            selected.extend(chosen)
        selected.extend(mix)
        return selected

    def _smooth_batch_extremes(self, batches, min_donor_util=0.55, low_util=0.65, max_iters=4):
        """Local rebalancing: move a single sample from donor to low-util bins when it improves both."""
        if len(batches) <= 1:
            return
        target = self.target_points

        for _ in range(max_iters):
            batch_pts = [sum(self.point_counts[i] for i in b) for b in batches]
            low_bins = [i for i, p in enumerate(batch_pts) if p < low_util * target]
            if not low_bins:
                break

            moved_any = False
            for low_i in low_bins:
                low_p = batch_pts[low_i]
                if low_p >= low_util * target:
                    continue

                best_move = None
                best_improve = 0.0
                for donor_i, donor in enumerate(batches):
                    if donor_i == low_i or len(donor) <= 1:
                        continue
                    donor_p = batch_pts[donor_i]
                    if donor_p <= min_donor_util * target:
                        continue

                    base = abs(target - low_p) + abs(target - donor_p)
                    for s_idx, sample_idx in enumerate(donor):
                        sp = self.point_counts[sample_idx]
                        if low_p + sp > target:
                            continue
                        new_donor_p = donor_p - sp
                        if new_donor_p < min_donor_util * target:
                            continue
                        new_cost = abs(target - (low_p + sp)) + abs(target - new_donor_p)
                        improve = base - new_cost
                        if improve > best_improve:
                            best_improve = improve
                            best_move = (donor_i, s_idx, sample_idx, sp)

                if best_move is not None:
                    donor_i, s_idx, sample_idx, sp = best_move
                    batches[donor_i].pop(s_idx)
                    batches[low_i].append(sample_idx)
                    batch_pts[donor_i] -= sp
                    batch_pts[low_i] += sp
                    moved_any = True

            if not moved_any:
                break

    def _create_balanced_bfd_batches(self):
        """
        Hybrid mode: class-balanced sample selection + class-aware BFD placement + local smoothing.
        Goal: keep utilization tight without large high/low extremes.
        """
        batches = []
        grid_sizes = getattr(self.dataset, 'grid_sizes', None)
        n_real = self.n_real
        n_total = self.n_total

        if grid_sizes is not None and len(grid_sizes) >= n_real:
            res2idx = {}
            for idx in range(n_real):
                res = grid_sizes[idx]
                res2idx.setdefault(res, []).append(idx)
            if n_total > n_real:
                # Mix slots do not have a native resolution tag.
                res2idx.setdefault(None, []).extend(range(n_real, n_total))
        else:
            res2idx = {None: list(range(n_total))}

        for _, indices in res2idx.items():
            if not indices:
                continue

            selected = self._balanced_indices_for_resolution(indices)
            if not selected:
                continue

            items = sorted(selected, key=lambda i: self.point_counts[i], reverse=True)
            res_batches = []
            batch_pts = []
            batch_lbl_hist = []

            for idx in items:
                pts = self.point_counts[idx]
                lbl = self._label_of(idx)

                # Oversized single sample: unavoidable, keep isolated.
                if pts >= self.target_points:
                    res_batches.append([idx])
                    batch_pts.append(pts)
                    batch_lbl_hist.append({lbl: 1})
                    continue

                best_b = -1
                best_score = None
                for b in range(len(res_batches)):
                    new_pts = batch_pts[b] + pts
                    if new_pts > self.target_points:
                        continue
                    rem = self.target_points - new_pts
                    lbl_count = batch_lbl_hist[b].get(lbl, 0)
                    # Prefer bins that fit tightly and improve class diversity.
                    score = rem + (0.15 * self.target_points * lbl_count) - (0.08 * self.target_points if lbl_count == 0 else 0.0)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_b = b

                if best_b >= 0:
                    res_batches[best_b].append(idx)
                    batch_pts[best_b] += pts
                    batch_lbl_hist[best_b][lbl] = batch_lbl_hist[best_b].get(lbl, 0) + 1
                else:
                    res_batches.append([idx])
                    batch_pts.append(pts)
                    batch_lbl_hist.append({lbl: 1})

            # Keep this mode strict-cap: no intentional >100% merges.
            self._smooth_batch_extremes(res_batches, min_donor_util=0.55, low_util=0.65, max_iters=4)
            self._merge_small_batches(res_batches, None, min_util=0.45, max_merge_ratio=1.0)
            batches.extend(res_batches)

        if self.verbose:
            self._print_batch_stats(batches)
        return batches

    def _create_balanced_batches(self):
        """Create batches that are both class-balanced and point-budget aware.
        Uses best-fit style packing: when adding to a batch, pick the largest sample that fits,
        alternating class to maintain balance. Much better utilization than sequential packing.
        """
        batches = []
        grid_sizes = getattr(self.dataset, 'grid_sizes', None)
        n_real = self.n_real
        mix_indices = list(range(n_real, self.n_total)) if self.n_total > n_real else []

        if grid_sizes is not None and len(grid_sizes) >= n_real:
            res2lbl2idx = {}
            for idx in range(n_real):
                res = grid_sizes[idx]
                label = self.labels[idx]
                res2lbl2idx.setdefault(res, {}).setdefault(label, []).append(idx)
        else:
            res2lbl2idx = {None: {label: np.where(self.labels == label)[0].tolist() for label in set(self.labels)}}

        for res, lbl2idx in res2lbl2idx.items():
            if not any(lbl2idx.values()):
                continue
            samples_per_class = min(len(idxs) for idxs in lbl2idx.values()) if self.mode == "downsampling" else max(len(idxs) for idxs in lbl2idx.values())
            class_pools = {
                label: np.random.choice(idxs, samples_per_class, replace=len(idxs) < samples_per_class).tolist()
                for label, idxs in lbl2idx.items()
            }
            # Per-class queues: (idx, point_count) sorted descending (best-fit picks largest that fits)
            labels_sorted = sorted(class_pools.keys())
            queues = {
                label: sorted([(i, self.point_counts[i]) for i in class_pools[label]], key=lambda x: -x[1])
                for label in labels_sorted
            }
            if res is None and mix_indices:
                queues[-1] = sorted([(i, self.point_counts[i]) for i in mix_indices], key=lambda x: -x[1])
                labels_sorted = labels_sorted + [-1]

            current_batch = []
            current_points = 0
            next_class_idx = 0
            res_batches = []

            # Hard cap: never pack more than target; single samples > target unavoidable (collate downsamples)
            max_batch_points = self.target_points

            while any(queues.values()):
                # Round-robin: pick next class that has samples (maintains balance)
                for _ in range(len(labels_sorted)):
                    label = labels_sorted[next_class_idx % len(labels_sorted)]
                    next_class_idx += 1
                    if queues.get(label):
                        break
                else:
                    label = next(lbl for lbl, q in queues.items() if q)
                queue = queues[label]

                remaining = max_batch_points - current_points
                # Best-fit: largest that fits, or if none fit then emit-first then overflow
                best_idx = -1
                best_pts = -1
                overflow_idx = -1
                for i, (idx, pts) in enumerate(queue):
                    if pts <= remaining:
                        if pts > best_pts:
                            best_pts, best_idx = pts, i
                    elif overflow_idx < 0:
                        overflow_idx = i

                # Nothing fits (overflow): small samples join current; large ones get their own batch
                if best_idx < 0 and overflow_idx >= 0:
                    idx, pts = queue.pop(overflow_idx)
                    # Small overflow: add to current to avoid single-4K batches (cap: target + min_points)
                    can_join = (current_batch and pts <= self.min_points
                                and current_points + pts <= max_batch_points + self.min_points)
                    if can_join:
                        current_batch.append(idx)
                        current_points += pts
                    else:
                        if current_batch:
                            res_batches.append(current_batch)
                            current_batch = []
                            current_points = 0
                        current_batch.append(idx)
                        current_points += pts
                elif best_idx >= 0:
                    idx, pts = queue.pop(best_idx)
                    # Strict cap: emit before adding if would exceed (except small samples to pack together)
                    would_exceed = current_points + pts > max_batch_points
                    small_ok = pts <= self.min_points and current_points + pts <= max_batch_points + self.min_points
                    if would_exceed and not small_ok:
                        if current_batch:
                            res_batches.append(current_batch)
                            current_batch = []
                            current_points = 0
                    current_batch.append(idx)
                    current_points += pts
                else:
                    break

                # Emit when at/over target or queues empty
                if current_points >= self.target_points or not any(queues.values()):
                    if current_batch:
                        # Greedy fill: before emitting, add any small sample that fits (avoids orphan batches)
                        slack = max_batch_points + self.min_points - current_points
                        filled = True
                        while filled and slack > 0:
                            filled = False
                            for qlabel, q in queues.items():
                                if not q:
                                    continue
                                # Find largest small sample that fits
                                for i, (qidx, qpts) in enumerate(q):
                                    if qpts <= slack and qpts <= self.min_points:
                                        q.pop(i)
                                        current_batch.append(qidx)
                                        current_points += qpts
                                        slack = max_batch_points + self.min_points - current_points
                                        filled = True
                                        break
                                if filled:
                                    break
                        if current_points >= self.min_points or not any(queues.values()):
                            res_batches.append(current_batch)
                            current_batch = []
                            current_points = 0

            if current_batch:
                res_batches.append(current_batch)

            # Merge small batches (< 50% util) into neighbor; holistic approach to avoid 3% orphans
            min_util = 0.5
            i = 1
            while i < len(res_batches):
                bp = sum(self.point_counts[j] for j in res_batches[i])
                if bp < self.target_points * min_util:
                    res_batches[i - 1].extend(res_batches[i])
                    res_batches.pop(i)
                else:
                    i += 1
            while len(res_batches) > 1 and sum(self.point_counts[j] for j in res_batches[0]) < self.target_points * min_util:
                res_batches[1].extend(res_batches[0])
                res_batches.pop(0)
            batches.extend(res_batches)

        # Final pass: merge standalone tiny batches (< 30%) into previous (even across resolutions)
        min_standalone = 0.3
        i = 1
        while i < len(batches):
            bp = sum(self.point_counts[j] for j in batches[i])
            if bp < self.target_points * min_standalone:
                batches[i - 1].extend(batches[i])
                batches.pop(i)
            else:
                i += 1

        if self.verbose:
            self._print_batch_stats(batches)
        return batches

    def __iter__(self):
        # Shuffle batches each epoch
        np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


class PointBudgetSampler(Sampler):
    """
    GPU memory-aware point budget sampler that automatically determines optimal batch sizes
    based on available GPU memory and runtime profiling.
    """
    def __init__(self, dataset, target_points_per_batch=None, memory_fraction=0.7, verbose=True):
        self.dataset = dataset
        self.memory_fraction = memory_fraction
        self.verbose = verbose
        self.target_points = target_points_per_batch or self._estimate_optimal_budget()
        self.batches = self._create_batches()

    def _estimate_optimal_budget(self):
        """Calculate optimal point budget as multiple of max_pts that fits in GPU memory"""
        import torch
        import gc

        if not torch.cuda.is_available():
            if self.verbose:
                print("CUDA not available, using 1x max_pts")
            return min(self.dataset.max_pts, 50000)

        device = torch.cuda.current_device()

        # Clear cache and get baseline memory usage
        torch.cuda.empty_cache()
        gc.collect()
        baseline_memory = torch.cuda.memory_allocated(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        available_memory = (total_memory - baseline_memory) * self.memory_fraction

        # Inference: ~400 bytes/point (model activations, no gradients). Conservative for OOM safety.
        bytes_per_point = 400

        # Cap effective max_pts for budget calc: huge values (e.g. 9999999) would produce oversized batches
        effective_max = min(self.dataset.max_pts, 50000)

        points_that_fit = int(available_memory / bytes_per_point)
        multiplier = max(1, points_that_fit // effective_max)
        multiplier = min(multiplier, 4)

        optimal_budget = multiplier * effective_max
        # Hard cap: never exceed 2M points per batch
        optimal_budget = min(optimal_budget, 2_000_000)

        if self.verbose:
            print(f"Point-budget (inference):")
            print(f"  Device: {torch.cuda.get_device_name(device)}")
            print(f"  Total GPU Memory: {total_memory / 1e9:.1f} GB")
            print(f"  Available for batching: {available_memory / 1e9:.1f} GB ({self.memory_fraction*100:.0f}%)")
            print(f"  Max points per sample: {self.dataset.max_pts:,} (effective: {effective_max:,})")
            print(f"  Target batch size: {optimal_budget:,} points")

        return optimal_budget


    def _get_sample_info(self, idx):
        """Load sample once; return (point_count, grid_size) for resolution-aware batching."""
        basename = os.path.basename(self.dataset.keys[idx])
        if not basename.startswith('voxel_'):
            try:
                name = basename.replace('.pt', '')
                parts = name.split('_')
                for p in parts:
                    if p.isdigit():
                        count = min(int(p), self.dataset.max_pts)
                        if count <= 0:
                            return 1, None
                        return count, None
            except Exception:
                pass
        try:
            point_cloud, grid_size = _load_voxel_file(self.dataset.keys[idx], weights_only=True)
            count = point_cloud.shape[0]
            if count <= 0:
                return 1, grid_size
            return min(count, self.dataset.max_pts), grid_size
        except Exception as e:
            if self.verbose:
                print(f"ERROR loading sample {idx}: {e}")
            return min(1024, self.dataset.max_pts), None

    def _get_point_count(self, idx):
        """Get point count for a dataset sample (reuse existing logic)"""
        count, _ = self._get_sample_info(idx)
        return count

    def _create_batches(self):
        """Create optimally packed batches; partition by grid_size so 4m and 2m never mix."""
        if self.verbose:
            print("Analyzing point cloud sizes for optimal adaptive batching (resolution-aware)...")

        sample_info = []
        for idx in range(len(self.dataset)):
            point_count, grid_size = self._get_sample_info(idx)
            sample_info.append((idx, point_count, grid_size))

        if self.verbose:
            point_counts = [x[1] for x in sample_info]
            print(f"Point count distribution: min={min(point_counts)}, max={max(point_counts)}, avg={np.mean(point_counts):.1f}")

            # More detailed analysis
            zero_count = sum(1 for pc in point_counts if pc <= 0)
            small_count = sum(1 for pc in point_counts if 0 < pc <= 1000)
            medium_count = sum(1 for pc in point_counts if 1000 < pc <= 5000)
            large_count = sum(1 for pc in point_counts if pc > 5000)

            print(f"Sample size breakdown:")
            if zero_count > 0:
                print(f"  Zero/invalid: {zero_count} samples")
            print(f"  Small (1-1000): {small_count} samples")
            print(f"  Medium (1001-5000): {medium_count} samples")
            print(f"  Large (>5000): {large_count} samples")

            # Check if adaptive batching makes sense for this data
            total_points = sum(point_counts)
            optimal_batch_count = total_points // self.target_points
            if optimal_batch_count < 10:
                print(f"WARNING: Total points ({total_points:,}) could fit in {optimal_batch_count} batches")
                print(f"         Consider using fixed batching with smaller max-pts instead")

        # Partition by grid_size so each batch has a single resolution (2m and 4m never mixed)
        res2samples = {}
        for idx, pc, gs in sample_info:
            res2samples.setdefault(gs, []).append((idx, pc))

        batches = []
        for res, si in res2samples.items():
            batches.extend(self._optimal_bin_packing(si))

        if self.verbose and len(res2samples) > 1:
            print(f"Resolution-aware: {len(res2samples)} grid sizes (no mixing within a batch)")

        if self.verbose:
            print(f"Created {len(batches)} adaptive batches with target {self.target_points} points each")

            # Analyze batch efficiency
            batch_points = [sum(self._get_point_count(idx) for idx in batch) for batch in batches]
            batch_sizes = [len(batch) for batch in batches]
            utilization = [bp / self.target_points for bp in batch_points]

            print(f"Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, avg={np.mean(batch_sizes):.1f}")
            print(f"Points per batch: min={min(batch_points):,}, max={max(batch_points):,}, avg={np.mean(batch_points):,.0f}")
            print(f"Memory utilization: min={min(utilization):.1%}, max={max(utilization):.1%}, avg={np.mean(utilization):.1%}")

            # Check for any batch exceeding the limit
            over_limit = [bp for bp in batch_points if bp > self.target_points]
            if over_limit:
                print(f"WARNING: {len(over_limit)} batches exceed target ({max(over_limit):,} > {self.target_points:,})")

        return batches

    def _optimal_bin_packing(self, sample_info):
        """Simple best-fit bin packing to get as close as possible to target_points"""
        # Sort samples by size (largest first for better packing)
        samples = sorted(sample_info, key=lambda x: x[1], reverse=True)
        available_samples = samples.copy()
        batches = []

        while available_samples:
            batch = []
            current_points = 0

            # Start with the first available sample
            idx, point_count = available_samples.pop(0)
            batch.append(idx)
            current_points += point_count

            # Greedily add samples that get us closest to target without exceeding it
            improved = True
            while improved and available_samples:
                improved = False
                remaining_budget = self.target_points - current_points

                if remaining_budget <= 0:
                    break

                # Find the sample that gets us closest to target without exceeding
                best_idx = -1
                best_fit_points = 0

                for i, (sample_idx, sample_points) in enumerate(available_samples):
                    if sample_points <= remaining_budget and sample_points > best_fit_points:
                        best_fit_points = sample_points
                        best_idx = i

                # Add the best fitting sample if found
                if best_idx >= 0:
                    sample_idx, sample_points = available_samples.pop(best_idx)
                    batch.append(sample_idx)
                    current_points += sample_points
                    improved = True

            batches.append(batch)

        return batches

    def __iter__(self):
        import random
        import torch
        random.shuffle(self.batches)

        for i, batch in enumerate(self.batches):
            if self.verbose and torch.cuda.is_available() and i < 3:  # Log first 3 batches
                batch_points = sum(self._get_point_count(idx) for idx in batch)
                current_memory = torch.cuda.memory_allocated() / 1e6
                print(f"    Batch {i+1}: {len(batch)} samples, {batch_points:,} points, GPU: {current_memory:.1f} MB")
            yield batch

    def __len__(self):
        return len(self.batches)


def _downsample_data_to_n(data, n, device=None):
    """Subsample a single Data to n points (random). Preserves pos, reflectance, y, edge_scores; sf unchanged."""
    if data.pos.size(0) <= n:
        return data
    dev = device if device is not None else data.pos.device
    idx = torch.randperm(data.pos.size(0), device=dev)[:n]
    out = data.clone()
    out.pos = data.pos[idx]
    if hasattr(data, 'reflectance') and data.reflectance is not None:
        out.reflectance = data.reflectance[idx]
    if hasattr(data, 'y') and data.y is not None:
        out.y = data.y[idx]
    if hasattr(data, 'edge_scores') and data.edge_scores is not None:
        out.edge_scores = data.edge_scores[idx]
    return out


def _downsample_batch_to_point_budget(batch, max_points):
    """Subsample a PyG Batch so total points <= max_points. Preserves batch structure. Used by collate only."""
    n = batch.pos.size(0)
    if n <= max_points:
        return batch
    device = batch.pos.device
    indices = torch.randperm(n, device=device)[:max_points]
    batch = batch.clone()
    batch.pos = batch.pos[indices]
    batch.batch = batch.batch[indices]
    if hasattr(batch, 'reflectance') and batch.reflectance is not None:
        batch.reflectance = batch.reflectance[indices]
    if hasattr(batch, 'y') and batch.y is not None:
        batch.y = batch.y[indices]
    if hasattr(batch, 'edge_scores') and batch.edge_scores is not None:
        batch.edge_scores = batch.edge_scores[indices]
    if hasattr(batch, 'x') and batch.x is not None:
        batch.x = batch.x[indices]
    old_batch = batch.batch
    unique_old = old_batch.unique(sorted=True)
    new_batch = torch.empty_like(old_batch)
    for new_idx, old_idx in enumerate(unique_old):
        new_batch[old_batch == old_idx] = new_idx
    batch.batch = new_batch
    if hasattr(batch, 'sf') and batch.sf is not None:
        if batch.sf.numel() == old_batch.max().item() + 1:
            batch.sf = batch.sf[unique_old]
        elif batch.sf.numel() == n:
            batch.sf = batch.sf[indices]
    if hasattr(batch, 'local_shift') and batch.local_shift is not None:
        # local_shift is stored per-sample as flattened [B*3]; preserve surviving samples.
        n_shift = batch.local_shift.numel() // 3
        if n_shift > 0:
            shift = batch.local_shift.view(n_shift, 3)
            valid_ids = unique_old[unique_old < n_shift]
            batch.local_shift = shift[valid_ids].flatten()
    return batch


def point_budget_collate(samples, target_points_per_batch):
    """
    Collate that caps each sample to the remaining batch budget in order.
    So the first sample gets up to target_points_per_batch points; the second gets up to
    (target - points_used_by_first); etc. Any sample (e.g. PointCutMix) that would exceed
    its slot gets random points removed to fit. Final batch is defensively capped so total <= target_points_per_batch.
    """
    if not samples:
        return Batch()
    capped = []
    points_so_far = 0
    for s in samples:
        remaining = target_points_per_batch - points_so_far
        if remaining <= 0:
            continue
        cap = min(s.pos.size(0), remaining)
        s_cap = _downsample_data_to_n(s, cap)
        capped.append(s_cap)
        points_so_far += s_cap.pos.size(0)
    if not capped:
        # Edge case: no room left; keep one sample capped to budget
        single = _downsample_data_to_n(samples[0], min(samples[0].pos.size(0), target_points_per_batch))
        batch = Batch.from_data_list([single])
    else:
        batch = Batch.from_data_list(capped)
    # Defensive: never return over budget (handles any worker/order edge case)
    if batch.pos.size(0) > target_points_per_batch:
        batch = _downsample_batch_to_point_budget(batch, target_points_per_batch)
    # Resolution-aware: set batch voxel_size when all samples share the same grid_size (single-resolution batch)
    in_batch = capped if capped else [samples[0]]
    grid_sizes = [getattr(s, 'grid_size', None) for s in in_batch]
    if grid_sizes and all(g is not None and g == grid_sizes[0] for g in grid_sizes):
        batch.voxel_size = torch.tensor([float(grid_sizes[0])], dtype=torch.float32)
    return batch


def create_train_loader(args, device):
    density_aug = getattr(args, 'density_aug', False)
    density_spacing = getattr(args, 'density_aug_spacing', [0.01, 0.04])
    smin, smax = density_spacing[0], density_spacing[1]
    train_dataset = TrainingDataset(
        voxels=args.trfile,
        augmentation=args.augmentation,
        mode='train',
        device=device,
        max_pts=args.max_pts,
        denoise=getattr(args, 'denoise', False),
        denoise_k=getattr(args, 'denoise_k', 16),
        denoise_std=getattr(args, 'denoise_std', 1.0),
        pointcutmix=getattr(args, 'pointcutmix', False),
        pointcutmix_prob=getattr(args, 'pointcutmix_prob', 0.25),
        pointcutmix_refl_dropout=getattr(args, 'pointcutmix_refl_dropout', 0.0),
        density_aug=density_aug,
        density_aug_prob=getattr(args, 'density_aug_prob', 0.5),
        density_aug_spacing_min=smin,
        density_aug_spacing_max=smax,
    )
    if density_aug:
        print(f"Density aug: random grid downsampling {smin*100:.1f}-{smax*100:.1f} cm (prob={getattr(args, 'density_aug_prob', 0.5)}, per-batch)")

    # Use point-budget-aware batching if specified, otherwise fixed batch size
    max_points_per_batch = getattr(args, 'max_points_per_batch', 0)

    if max_points_per_batch > 0:
        # Point-budget-aware balanced sampling (prevents OOM)
        min_points_per_batch = getattr(args, 'min_points_per_batch', 16000)
        train_sampler = BalancedPointBudgetSampler(
            dataset=train_dataset,
            labels=train_dataset.labels,
            target_points_per_batch=max_points_per_batch,
            min_points_per_batch=min_points_per_batch,
            mode=args.balance_mode,
            packing_mode=getattr(args, 'packing_mode', 'bfd'),
            verbose=getattr(args, 'verbose', False)
        )

        def _train_collate(batch):
            return point_budget_collate(batch, max_points_per_batch)

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=min(8, os.cpu_count() or 4),
            pin_memory=True,
            collate_fn=_train_collate
        )
        if getattr(train_dataset, 'pointcutmix', False):
            total_slots = sum(len(b) for b in train_sampler.batches)
            mix_slots = getattr(train_dataset, '_num_mix_slots', 0)
            print(f"PointCutMix: +{mix_slots} mix samples/epoch (append); {total_slots} total samples → {len(train_sampler.batches)} batches (point-budget)")
    else:
        # Original fixed batch size sampling
        base_sampler = BalanceClassSampler(
            labels=train_dataset.labels,
            mode=args.balance_mode,
        )
        mix_slots = getattr(train_dataset, '_num_mix_slots', 0)
        n_real = len(train_dataset.keys)
        if mix_slots > 0:
            # Append mix indices so DataLoader requests them → more samples/batches per epoch
            class _SamplerWithMix(Sampler):
                def __init__(self, base, n_real, mix_slots):
                    self.base = base
                    self.n_real = n_real
                    self.mix_slots = mix_slots

                def __iter__(self):
                    indices = list(iter(self.base))
                    indices.extend(range(self.n_real, self.n_real + self.mix_slots))
                    np.random.shuffle(indices)
                    return iter(indices)

                def __len__(self):
                    return len(self.base) + self.mix_slots

            train_sampler = _SamplerWithMix(base_sampler, n_real, mix_slots)
        else:
            train_sampler = base_sampler

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers=min(8, os.cpu_count() or 4),
            pin_memory=True,
            collate_fn=_fixed_batch_collate
        )
        if getattr(train_dataset, 'pointcutmix', False):
            total_slots = len(train_sampler)
            print(f"PointCutMix: +{mix_slots} mix samples/epoch (append); {total_slots} total samples → {len(train_loader)} batches")

    return train_loader, train_dataset

def create_test_loader(args, device):
    test_dataset = TrainingDataset(
        voxels=args.tefile,
        augmentation=args.augmentation,
        mode='test',
        device=device,
        max_pts=args.max_pts,
        denoise=getattr(args, 'denoise', False),
        denoise_k=getattr(args, 'denoise_k', 16),
        denoise_std=getattr(args, 'denoise_std', 1.0)
    )

    # Use point-budget-aware batching if specified
    max_points_per_batch = getattr(args, 'max_points_per_batch', 0)

    if max_points_per_batch > 0:
        test_sampler = BalancedPointBudgetSampler(
            dataset=test_dataset,
            labels=test_dataset.labels,
            target_points_per_batch=max_points_per_batch,
            min_points_per_batch=getattr(args, 'min_points_per_batch', 16000),
            mode=args.balance_mode,
            packing_mode=getattr(args, 'packing_mode', 'bfd'),
            verbose=False  # Less verbose for validation
        )

        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            num_workers=min(8, os.cpu_count() or 4),
            pin_memory=True,
            collate_fn=_fixed_batch_collate
        )
    else:
        test_sampler = BalanceClassSampler(
            labels=test_dataset.labels,
            mode=args.balance_mode,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            drop_last=True,
            num_workers=min(8, os.cpu_count() or 4),
            pin_memory=True,
            collate_fn=_fixed_batch_collate
        )

    return test_loader, test_dataset

def create_inference_loader(args, device):
    test_dataset = TestingDataset(
        voxels=args.vxfile,
        device=device,
        max_pts=args.max_pts,
        denoise=getattr(args, 'denoise', False),
        denoise_k=getattr(args, 'denoise_k', 16),
        denoise_std=getattr(args, 'denoise_std', 1.0),
        file_pattern=getattr(args, 'eval_file_pattern', None)
    )
    
    use_perspectives = hasattr(args, 'boost_perspective') and args.boost_perspective
    
    if use_perspectives:
        if args.verbose:
            print("Using multi-perspective inference with 7 different views of each point cloud")
        from src.perspectives import MultiPerspectiveDataset
        test_dataset = MultiPerspectiveDataset(test_dataset)
    
    from torch.utils.data import DataLoader
    import os

    # Conservative worker count to avoid "too many open files" errors
    # Use at most 4 workers to prevent file handle exhaustion
    cpu_count = os.cpu_count() if os.cpu_count() else 4
    num_workers = min(4, max(1, cpu_count - 2))

    # Choose batching strategy based on batch_size parameter
    use_adaptive = getattr(args, 'batch_size', 0) == 0

    if use_adaptive:
        # Match trainer behavior: explicit point budget by default.
        max_points_per_batch = getattr(args, 'max_points_per_batch', 50000)
        target_points = max_points_per_batch if max_points_per_batch > 0 else None
        point_sampler = PointBudgetSampler(
            test_dataset,
            target_points_per_batch=target_points,
            memory_fraction=getattr(args, 'memory_fraction', 0.7),
            verbose=True
        )
        target_pts = point_sampler.target_points

        def inference_collate(batch):
            from torch_geometric.data import Batch
            b = Batch.from_data_list(batch)
            if b.pos.size(0) > target_pts:
                b = _downsample_batch_to_point_budget(b, target_pts)
            # Single-resolution batch: set voxel_size so model uses correct scale (2m vs 4m not mixed)
            grid_sizes = [getattr(s, 'grid_size', None) for s in batch]
            if grid_sizes and all(g is not None and g == grid_sizes[0] for g in grid_sizes):
                b.voxel_size = torch.tensor([float(grid_sizes[0])], dtype=torch.float32)
            return b

        test_loader = DataLoader(
            test_dataset,
            batch_sampler=point_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=False,
            collate_fn=inference_collate
        )
    else:
        # Traditional fixed batch size
        if args.verbose:
            print(f"Using fixed batching: {args.batch_size} samples per batch")

        def geometric_collate(batch):
            from torch_geometric.data import Batch
            b = Batch.from_data_list(batch)
            grid_sizes = [getattr(s, 'grid_size', None) for s in batch]
            if grid_sizes and all(g is not None and g == grid_sizes[0] for g in grid_sizes):
                b.voxel_size = torch.tensor([float(grid_sizes[0])], dtype=torch.float32)
            return b

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=False,
            collate_fn=geometric_collate
        )

    return test_loader, test_dataset 
