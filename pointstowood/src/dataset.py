import os
import glob
import torch
import numpy as np
from abc import ABC
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Sampler
from torch_geometric.nn import voxel_grid
import torch_scatter
from src.augmentation import augmentations

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

class TrainingDataset(Dataset, ABC):
    def __init__(self, voxels, augmentation, mode, max_pts, device, denoise=False, denoise_k=16, denoise_std=1.0,
                 edge_label_smoothing=False, smoothing_factor=0.1):
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
        self.edge_label_smoothing = edge_label_smoothing
        self.smoothing_factor = smoothing_factor

        if self.denoise:
            print(f"Denoising enabled with k={denoise_k}, std_threshold={denoise_std}")
        if self.edge_label_smoothing:
            print(f"Edge-aware label smoothing enabled with factor={smoothing_factor}")
        
        for key in self.keys:
            point_cloud = torch.load(key, weights_only=True)
            y = point_cloud[:, self.label_index]
            sample_label = 1 if (y > 0.50).sum() > len(y) / 2 else 0
            self.labels.append(sample_label)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if index >= len(self.keys):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.keys)}")

        point_cloud = torch.load(self.keys[index], weights_only=True)
        pos = torch.as_tensor(point_cloud[:, :3], dtype=torch.float).requires_grad_(False)
        reflectance = torch.as_tensor(point_cloud[:, self.reflectance_index], dtype=torch.float)
        y = torch.as_tensor(point_cloud[:, self.label_index], dtype=torch.float)
        
        if self.mode == 'train' and point_cloud.shape[-1] > 5:
            edge_scores_precomputed = torch.as_tensor(point_cloud[:, 5], dtype=torch.float)
        else:
            edge_scores_precomputed = None
        
        if self.denoise:
            pos, reflectance, y, edge_scores_precomputed = sor_filter(pos, reflectance, y, edge_scores_precomputed, self.denoise_k, self.denoise_std)
        
        if len(pos) > self.max_pts:
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
        scaling_factor = torch.sqrt((pos ** 2).sum(dim=1)).max()

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

        # Note: Edge-aware label smoothing now applied in trainer after final edge scores

        return Data(
            pos=pos,
            reflectance=reflectance,
            y=y,
            sf=scaling_factor,
            edge_scores=edge_scores
        )

class TestingDataset(Dataset, ABC):
    def __init__(self, voxels, max_pts, device, in_memory=False, denoise=False, denoise_k=16, denoise_std=1.0):
        if not voxels:
            raise ValueError("The 'voxels' parameter cannot be empty.")
        self.voxels = voxels
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
        point_cloud = torch.load(self.keys[index])
        pos = torch.as_tensor(point_cloud[:, :3], dtype=torch.float).requires_grad_(False)
        reflectance = torch.as_tensor(point_cloud[:, self.reflectance_index], dtype=torch.float)

        if self.denoise:
            pos, reflectance, _, _ = sor_filter(pos, reflectance, k=self.denoise_k, std_threshold=self.denoise_std)
        
        if len(pos) > self.max_pts:
            indices = torch.randperm(len(pos))[:self.max_pts]
            pos = pos[indices]
            reflectance = reflectance[indices]
        
        local_shift = torch.mean(pos[:, :3], axis=0).requires_grad_(False)
        pos = pos - local_shift
        scaling_factor = torch.sqrt((pos ** 2).sum(dim=1)).max()

        nan_mask = torch.isnan(pos).any(dim=1) | torch.isnan(reflectance)
        pos = pos[~nan_mask]
        reflectance = reflectance[~nan_mask]

        if nan_mask.any(): 
            print(f"Encountered NaN values in sample at index {index}")
        
        data = Data(pos=pos, reflectance=reflectance, local_shift=local_shift, sf=scaling_factor)
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
            return self.dataset.max_pts

        device = torch.cuda.current_device()

        # Clear cache and get baseline memory usage
        torch.cuda.empty_cache()
        gc.collect()
        baseline_memory = torch.cuda.memory_allocated(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        available_memory = (total_memory - baseline_memory) * self.memory_fraction

        # Conservative estimate: assume each point uses ~100 bytes (includes model overhead)
        # This accounts for: coordinates, features, intermediate activations, gradients, etc.
        bytes_per_point = 100

        # Calculate how many max_pts worth of points we can fit
        points_that_fit = int(available_memory / bytes_per_point)
        multiplier = max(1, points_that_fit // self.dataset.max_pts)

        # Cap at reasonable maximum (8x max_pts = 131K for 16K max_pts)
        multiplier = min(multiplier, 8)

        optimal_budget = multiplier * self.dataset.max_pts

        if self.verbose:
            print(f"GPU Memory Analysis:")
            print(f"  Device: {torch.cuda.get_device_name(device)}")
            print(f"  Total GPU Memory: {total_memory / 1e9:.1f} GB")
            print(f"  Available for batching: {available_memory / 1e9:.1f} GB ({self.memory_fraction*100:.0f}%)")
            print(f"  Max points per sample: {self.dataset.max_pts:,}")
            print(f"  Memory multiplier: {multiplier}x")
            print(f"  Target batch size: {optimal_budget:,} points ({optimal_budget / self.dataset.max_pts:.1f}x max_pts)")

        return optimal_budget


    def _get_point_count(self, idx):
        """Get point count for a dataset sample (reuse existing logic)"""
        # Skip filename parsing for voxel files - filenames like "voxel_0.pt" don't contain point counts
        basename = os.path.basename(self.dataset.keys[idx])
        if basename.startswith('voxel_'):
            # For voxel files, always load the actual file to get point count
            pass
        else:
            # For other files, try filename parsing first
            try:
                name = basename.replace('.pt', '')
                parts = name.split('_')
                for p in parts:
                    if p.isdigit():
                        count = min(int(p), self.dataset.max_pts)
                        if count <= 0:
                            if self.verbose:
                                print(f"WARNING: Sample {basename} has {count} points from filename")
                            return 1  # Minimum viable count
                        return count
            except Exception:
                pass
        try:
            point_cloud = torch.load(self.dataset.keys[idx], map_location='cpu')
            count = point_cloud.shape[0]
            del point_cloud
            if count <= 0:
                if self.verbose:
                    print(f"WARNING: Sample {self.dataset.keys[idx]} has {count} points from file")
                return 1  # Minimum viable count
            return min(count, self.dataset.max_pts)
        except Exception as e:
            if self.verbose:
                print(f"ERROR loading sample {idx}: {e}")
            return min(1024, self.dataset.max_pts)  # Safe fallback

    def _create_batches(self):
        """Create optimally packed batches using intelligent bin packing"""
        if self.verbose:
            print("Analyzing point cloud sizes for optimal adaptive batching...")

        sample_info = []
        for idx in range(len(self.dataset)):
            point_count = self._get_point_count(idx)
            sample_info.append((idx, point_count))

        if self.verbose:
            point_counts = [pc for _, pc in sample_info]
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

        # Use intelligent bin packing algorithm
        batches = self._optimal_bin_packing(sample_info)

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

def create_train_loader(args, device):
    train_dataset = TrainingDataset(
        voxels=args.trfile,
        augmentation=args.augmentation,
        mode='train',
        device=device,
        max_pts=args.max_pts,
        denoise=getattr(args, 'denoise', False),
        denoise_k=getattr(args, 'denoise_k', 16),
        denoise_std=getattr(args, 'denoise_std', 1.0),
        edge_label_smoothing=getattr(args, 'edge_label_smoothing', False),
        smoothing_factor=getattr(args, 'smoothing_factor', 0.1)
    )
    
    train_sampler = BalanceClassSampler(
        labels=train_dataset.labels,
        mode=args.balance_mode,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=32,
        pin_memory=True
    )
    
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

    test_sampler = BalanceClassSampler(
        labels=test_dataset.labels,
        mode=args.balance_mode,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        drop_last=True, 
        num_workers=32, 
        pin_memory=True
    )
    
    return test_loader, test_dataset

def create_inference_loader(args, device):
    test_dataset = TestingDataset(
        voxels=args.vxfile, 
        device=device, 
        max_pts=args.max_pts,
        denoise=getattr(args, 'denoise', False),
        denoise_k=getattr(args, 'denoise_k', 16),
        denoise_std=getattr(args, 'denoise_std', 1.0)
    )
    
    use_perspectives = hasattr(args, 'boost_perspective') and args.boost_perspective
    
    if use_perspectives:
        if args.verbose:
            print("Using multi-perspective inference with 7 different views of each point cloud")
        from src.perspectives import MultiPerspectiveDataset
        test_dataset = MultiPerspectiveDataset(test_dataset)
    
    target_points = args.max_pts

    from torch.utils.data import DataLoader
    import os

    # Conservative worker count to avoid "too many open files" errors
    # Use at most 4 workers to prevent file handle exhaustion
    cpu_count = os.cpu_count() if os.cpu_count() else 4
    num_workers = min(4, max(1, cpu_count - 2))

    # Custom collate function using torch_geometric's batching
    def geometric_collate(batch):
        """Use torch_geometric's built-in batching for Data objects"""
        from torch_geometric.data import Batch
        return Batch.from_data_list(batch)

    # Choose batching strategy based on batch_size parameter
    use_adaptive = getattr(args, 'batch_size', 0) == 0

    if use_adaptive:
        if args.verbose:
            print("Using adaptive GPU memory-aware point budget sampler")
            print(f"  Memory fraction: {getattr(args, 'memory_fraction', 0.7)*100:.0f}%")
            print(f"  DataLoader workers: {num_workers} (limited to 4 max, from {cpu_count} CPU threads)")
        point_sampler = PointBudgetSampler(
            test_dataset,
            target_points_per_batch=None,  # Let it auto-determine
            memory_fraction=getattr(args, 'memory_fraction', 0.7),
            verbose=args.verbose
        )

        test_loader = DataLoader(
            test_dataset,
            batch_sampler=point_sampler,
            shuffle=False,
            num_workers=num_workers,  # Auto-detected optimal workers
            pin_memory=True,  # Faster GPU transfer
            prefetch_factor=4,  # Prefetch multiple batches
            persistent_workers=False,  # Disable to prevent file handle buildup
            collate_fn=geometric_collate
        )
    else:
        # Traditional batching - use DataLoader's built-in batching
        if args.verbose:
            print(f"Using traditional fixed batching with {args.batch_size} samples per batch")

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,  # Auto-detected optimal workers
            pin_memory=True,  # Faster GPU transfer
            prefetch_factor=4,  # Prefetch multiple batches
            persistent_workers=False,  # Disable to prevent file handle buildup
            collate_fn=geometric_collate
        )

    return test_loader, test_dataset 