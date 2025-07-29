import os
import glob
import torch
import numpy as np
from abc import ABC
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Sampler
from torch_geometric.nn import knn, voxel_grid
import torch_scatter
from src.augmentation import augmentations

def sor_filter(pos, reflectance=None, y=None, k=16, std_threshold=1.0):
    try:
        from sklearn.neighbors import KDTree
    except ImportError:
        print("Warning: sklearn not available, skipping denoising")
        return pos, reflectance, y
    
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
    
    return pos_filtered, reflectance_filtered, y_filtered

class TrainingDataset(Dataset, ABC):
    def __init__(self, voxels, augmentation, mode, max_pts, device, denoise=False, denoise_k=16, denoise_std=1.0):
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
        self.voxel_size = 0.1 
        
        self.denoise = denoise
        self.denoise_k = denoise_k
        self.denoise_std = denoise_std
        if self.denoise:
            print(f"Denoising enabled with k={denoise_k}, std_threshold={denoise_std}")
        
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
        
        if self.denoise:
            pos, reflectance, y = sor_filter(pos, reflectance, y, self.denoise_k, self.denoise_std)
        
        if len(pos) > self.max_pts:
            indices = torch.randperm(len(pos))[:self.max_pts]
            pos = pos[indices]
            reflectance = reflectance[indices]
            y = y[indices]
        
        if self.augmentation:
            pos, reflectance, y = augmentations(pos, reflectance, y, self.mode)

        local_shift = torch.mean(pos[:, :3], axis=0).requires_grad_(False)
        pos = pos - local_shift
        scaling_factor = torch.sqrt((pos ** 2).sum(dim=1)).max()

        if torch.any(torch.isnan(reflectance)):
            print('nans in relfectance')

        cluster = voxel_grid(pos, size=self.voxel_size, batch=None)        
        pos_sum = torch_scatter.scatter_add((y == 1).float(), cluster, dim=0)
        count = torch_scatter.scatter_add(torch.ones_like(y), cluster, dim=0)
        pos_prop = pos_sum / (count + 1e-6)        
        edge_scores = ((pos_prop[cluster] > 0) & (pos_prop[cluster] < 1)).float()
        
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
            pos, reflectance = sor_filter(pos, reflectance, k=self.denoise_k, std_threshold=self.denoise_std)
        
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

#Credit to catalyst sampler where we got the code from: https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
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
    def __init__(self, dataset, target_points_per_batch=65536):
        self.dataset = dataset
        self.target_points = target_points_per_batch
        self.batches = self._create_batches()
    
    def _get_point_count(self, idx):
        try:
            point_cloud = torch.load(self.dataset.keys[idx], map_location='cpu')
            count = point_cloud.shape[0]
            del point_cloud
            return count
        except:
            return 8192
    
    def _create_batches(self):
        print("Analyzing point cloud sizes for optimal batching...")
        
        sample_info = []
        for idx in range(len(self.dataset)):
            point_count = self._get_point_count(idx)
            sample_info.append((idx, point_count))
        
        sample_info.sort(key=lambda x: x[1])
        
        batches = []
        current_batch = []
        current_points = 0
        
        for idx, point_count in sample_info:
            if current_points + point_count > self.target_points and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_points = 0
            
            current_batch.append(idx)
            current_points += point_count
            
            if len(current_batch) >= 8:
                batches.append(current_batch)
                current_batch = []
                current_points = 0
        
        if current_batch:
            batches.append(current_batch)
        
        print(f"Created {len(batches)} batches with target {self.target_points} points each")
        
        batch_sizes = [len(batch) for batch in batches]
        print(f"Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, avg={np.mean(batch_sizes):.1f}")
        
        return batches
    
    def __iter__(self):
        import random
        random.shuffle(self.batches)
        for batch in self.batches:
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
        denoise_std=getattr(args, 'denoise_std', 1.0)
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
    
    target_points = (args.max_pts // 2) * args.batch_size
    if args.verbose:
        print(f"Using point-budget batching with target {target_points} points per batch")
        print(f"  (based on {args.max_pts//2} avg points × {args.batch_size} batch size)")
    
    point_sampler = PointBudgetSampler(test_dataset, target_points_per_batch=target_points)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_sampler=point_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return test_loader, test_dataset 