from src.dataset import create_inference_loader
import os
import sys
import pandas as pd
import numpy as np
from pykdtree.kdtree import KDTree
from tqdm.auto import tqdm
import torch
from src.io import save_file
from collections import OrderedDict
from numba import jit, prange
import psutil
import gc
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter_max, scatter_add

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.setrecursionlimit(10 ** 8)        
        
# Memory tracking removed for clean output

def _load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    raw_state = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    adjusted_state_dict = OrderedDict()
    for key, value in raw_state.items():
        if key.startswith('module.'):
            key = key[7:]
        adjusted_state_dict[key] = value
    return checkpoint, adjusted_state_dict


def _infer_model_config(model_name, checkpoint, state_dict):
    name = (model_name or "").lower()
    distill_cfg = checkpoint.get('distill_config', {}) if isinstance(checkpoint, dict) else {}

    has_full_depth = any(k.startswith('sa2_module.residual_blocks.7.') for k in state_dict.keys())
    has_light_depth = any(k.startswith('sa2_module.residual_blocks.1.') for k in state_dict.keys())
    if has_full_depth:
        model_family = 'full'
    elif has_light_depth:
        model_family = 'light'
    else:
        model_family = 'full' if ('eu' in name or 'global' in name) else 'light'

    seg0 = state_dict.get('seg_head.0.weight', None)
    if isinstance(seg0, torch.Tensor) and seg0.ndim == 3 and seg0.shape[1] % 3 == 0:
        c_base = int(seg0.shape[1] // 3)
    else:
        c_base = int(distill_cfg.get('student_c', 16 if model_family == 'light' else 128))

    k_tensor = state_dict.get('sa1_module.conv.kernel_points', None)
    if isinstance(k_tensor, torch.Tensor):
        num_kernel_points = int(k_tensor.shape[0])
    else:
        num_kernel_points = int(distill_cfg.get('student_kernels', 8 if model_family == 'light' else 16))

    has_kernel_dirs = any(k.endswith('conv.kernel_dirs') for k in state_dict.keys())
    learnable_kernels = not has_kernel_dirs
    if 'student_learnable_kernels' in distill_cfg and model_family == 'light':
        learnable_kernels = bool(distill_cfg['student_learnable_kernels'])

    spatial_mix_lite = any('spatial_mix_scale' in k for k in state_dict.keys())

    if 'dualnorm_lite' in distill_cfg:
        dualnorm_lite = bool(distill_cfg['dualnorm_lite'])
    else:
        # Heuristic: when DualNorm-lite is active in training, alpha/beta logits usually move from init.
        dualnorm_init_logit = float(torch.logit(torch.tensor(0.25)).item())
        dn_keys = [k for k in state_dict.keys() if k.endswith('dualnorm_alpha_logit') or k.endswith('dualnorm_beta_logit')]
        moved = False
        for k in dn_keys:
            t = state_dict.get(k, None)
            if isinstance(t, torch.Tensor) and t.numel() == 1:
                if abs(float(t.detach().cpu().item()) - dualnorm_init_logit) > 1e-3:
                    moved = True
                    break
        dualnorm_lite = moved

    # Block counts per SA stage — read from checkpoint if present, else infer from state dict
    sa1_blocks = int(distill_cfg.get('sa1_blocks', 1) if model_family == 'light' else 4)
    sa3_blocks = int(distill_cfg.get('sa3_blocks', 1) if model_family == 'light' else 2)
    # SA2 blocks: count how many residual_blocks exist in the checkpoint
    sa2_count = sum(1 for k in state_dict if k.startswith('sa2_module.residual_blocks.') and k.endswith('.layer_scale'))
    if sa2_count > 0:
        sa2_blocks = sa2_count
    else:
        sa2_blocks = int(distill_cfg.get('sa2_blocks', 2) if model_family == 'light' else 6)

    return {
        'model_family': model_family,
        'c_base': c_base,
        'num_kernel_points': num_kernel_points,
        'learnable_kernels': learnable_kernels,
        'spatial_mix_lite': spatial_mix_lite,
        'dualnorm_lite': dualnorm_lite,
        'sa1_blocks': sa1_blocks,
        'sa2_blocks': sa2_blocks,
        'sa3_blocks': sa3_blocks,
    }


def load_model(model, state_dict):
    model_state = model.state_dict()
    compatible = OrderedDict()
    skipped_shape = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append(key)
            continue
        compatible[key] = value

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    coverage = len(compatible) / max(1, len(model_state))
    print(f"Checkpoint load coverage: {len(compatible)}/{len(model_state)} tensors ({coverage * 100:.1f}%)")
    if skipped_shape:
        print(f"Skipped {len(skipped_shape)} tensors due to shape mismatch.")
    if missing:
        print(f"Missing tensors after load: {len(missing)}")
    if unexpected:
        print(f"Unexpected tensors in checkpoint: {len(unexpected)}")
    return model
    
class KnnCloudClassifier:
    """Aggregate KNN predictions into a final label per original point.

    Modes (mutually exclusive, priority top→bottom):
      1. *max_probability*  – choose the most confident prediction in the
         neighborhood (furthest from 0.5).
      2. *any_wood* (value ≠ 1) – if *any* probability ≥ any_wood, label as wood
         irrespective of aggregate probability.
      3. Default – take the median probability and compare against *is_wood*.
    """

    def __init__(self, is_wood: float, any_wood: float, max_probability: bool = False):
        self.is_wood = is_wood
        self.any_wood = any_wood
        self.max_probability = max_probability

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _labels_median_threshold(nbr_classification, labels, is_wood):
        """Median probability compared with *is_wood* threshold."""
        num_neighborhoods = labels.shape[0]
        for i in prange(num_neighborhoods):
            median_prob = np.median(nbr_classification[i, :, -1])
            labels[i, 1] = median_prob
            labels[i, 0] = 1 if median_prob >= is_wood else 0
        return labels

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _labels_any_wood(nbr_classification, labels, any_wood):
        """Label wood if *any* probability ≥ any_wood, else leaf."""
        num_neighborhoods = labels.shape[0]
        for i in prange(num_neighborhoods):
            probs = nbr_classification[i, :, -1]
            labels[i, 1] = np.median(probs)
            labels[i, 0] = 1 if np.any(probs >= any_wood) else 0
        return labels

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _labels_argmax(nbr_classification, labels):
        num_neighborhoods = labels.shape[0]
        for i in prange(num_neighborhoods):
            probs = nbr_classification[i, :, -1]
            conf_idx = np.argmax(np.abs(probs - 0.5))
            conf_prob = probs[conf_idx]
            labels[i, 1] = conf_prob
            labels[i, 0] = nbr_classification[i, conf_idx, -2]
        return labels

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _labels_hysteresis(nbr_classification, labels, t_low=0.40, t_high=0.70, m_of_k=4):
        num_neighborhoods = labels.shape[0]
        for i in prange(num_neighborhoods):
            probs = nbr_classification[i, :, -1]
            median_prob = np.median(probs)
            labels[i, 1] = median_prob
            if median_prob >= t_high:
                labels[i, 0] = 1
            elif median_prob <= t_low:
                labels[i, 0] = 0
            else:
                count_high = 0
                for p in probs:
                    if p >= t_high:
                        count_high += 1
                labels[i, 0] = 1 if count_high >= m_of_k else 0
        return labels

    def collect_predictions(self, classification, original):
        original = original.drop(columns=[c for c in original.columns if c in ['prediction', 'pwood', 'pleaf']])


        kd_tree = KDTree(classification[:, :3])
        _, indices = kd_tree.query(original.values[:, :3], k=16)

        labels = np.zeros((original.shape[0], 2))

        if hasattr(self, 'use_hysteresis') and self.use_hysteresis:
            labels = self._labels_hysteresis(classification[indices], labels)
        elif self.max_probability:
            labels = self._labels_argmax(classification[indices], labels)
        elif self.any_wood != 1:
            labels = self._labels_any_wood(classification[indices], labels, self.any_wood)
        else:
            labels = self._labels_median_threshold(classification[indices], labels, self.is_wood)

        original.loc[:, ['prediction', 'pwood']] = labels
        return original


class GridCloudClassifier:
    """Aggregate per-voxel. When max_probability=True: argmax |p-0.5| per voxel (matches trainer eval)."""
    def __init__(self, is_wood: float, any_wood: float, grid_size: float, max_probability: bool = False):
        self.is_wood = is_wood
        self.any_wood = any_wood
        self.grid_size = grid_size
        self.max_probability = max_probability

    def collect_predictions(self, classified_pc: np.ndarray, original: pd.DataFrame) -> pd.DataFrame:
        original = original.drop(columns=[c for c in original.columns if c in ['prediction', 'pwood', 'pleaf']])

        orig_pos = torch.as_tensor(original[['x','y','z']].values, dtype=torch.float, device='cpu')
        class_pos = torch.as_tensor(classified_pc[:, :3], dtype=torch.float, device='cpu')
        class_prob = torch.as_tensor(classified_pc[:, -1], dtype=torch.float, device='cpu')
        class_prob = torch.nan_to_num(class_prob, nan=0.0)

        combined_pos = torch.cat([orig_pos, class_pos], dim=0)
        cluster = voxel_grid(combined_pos, self.grid_size)
        cluster, _ = consecutive_cluster(cluster)

        n_orig = orig_pos.shape[0]
        n_clusters = int(cluster.max().item()) + 1
        class_cluster = cluster[n_orig:]
        orig_cluster = cluster[:n_orig]

        if self.max_probability:
            # Label from argmax |p-0.5| per voxel; pwood from median (reflects genuine uncertainty)
            conf = torch.abs(class_prob - 0.5)
            _, argmax_idx = scatter_max(conf, class_cluster, dim=0, dim_size=n_clusters)
            winning_prob = class_prob[argmax_idx.clamp(0, len(class_prob) - 1)]
            has_class = scatter_add(torch.ones_like(class_cluster, dtype=torch.float), class_cluster, dim=0, dim_size=n_clusters) > 0
            voxel_label = torch.zeros(n_clusters, dtype=torch.int64)
            voxel_prob = torch.zeros(n_clusters, dtype=torch.float)
            voxel_label[has_class] = (winning_prob[has_class] >= 0.5).long()
            # Median probability per voxel for pwood (sort-based, avoids per-cluster loop)
            class_prob_np = class_prob.numpy()
            class_cluster_np = class_cluster.numpy()
            sort_idx = np.argsort(class_cluster_np, kind='stable')
            sorted_clusters = class_cluster_np[sort_idx]
            sorted_probs = class_prob_np[sort_idx]
            boundaries = np.flatnonzero(np.diff(sorted_clusters)) + 1
            groups = np.split(sorted_probs, boundaries)
            unique_clusters = sorted_clusters[np.concatenate(([0], boundaries))]
            median_prob_np = np.zeros(n_clusters, dtype=np.float32)
            for cid, grp in zip(unique_clusters, groups):
                median_prob_np[cid] = np.median(grp)
            voxel_prob[has_class] = torch.from_numpy(median_prob_np)[has_class].clamp(0.0, 1.0)
        else:
            neg_inf = torch.full((n_orig,), float('-inf'), device='cpu')
            prob_for_max = torch.cat([neg_inf, class_prob], dim=0)
            max_prob, _ = scatter_max(prob_for_max, cluster, dim=0)
            voxel_label = (max_prob >= self.any_wood).to(torch.int64)
            voxel_prob = torch.clamp(max_prob, 0.0, 1.0)

        point_labels = voxel_label[orig_cluster].numpy()
        point_probs = voxel_prob[orig_cluster].numpy()

        original.loc[:, ['prediction', 'pwood']] = np.stack([point_labels, point_probs], axis=1)
        return original

def SemanticSegmentation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = os.path.join(args.wdir, 'model', args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'No model found at {model_path}')

    checkpoint, checkpoint_state = _load_checkpoint(model_path, device)

    dualnorm_override = getattr(args, 'dualnorm_lite', None)
    inferred = _infer_model_config(args.model, checkpoint, checkpoint_state)
    dualnorm_lite = inferred['dualnorm_lite'] if dualnorm_override is None else bool(dualnorm_override)

    if inferred['model_family'] == 'full':
        from src.model import NetFull as Net
        model = Net(
            num_classes=1,
            C=inferred['c_base'],
            num_kernel_points=inferred['num_kernel_points'],
            learnable_kernels=inferred['learnable_kernels'],
            drop_path_rate=0.0,
            dualnorm_lite=dualnorm_lite,
            spatial_mix_lite=inferred['spatial_mix_lite'],
        ).to(device)
        model_label = "EU/Global"
    else:
        from src.model import NetLight as Net
        model = Net(
            num_classes=1,
            C=inferred['c_base'],
            num_kernel_points=inferred['num_kernel_points'],
            learnable_kernels=inferred['learnable_kernels'],
            drop_path_rate=0.0,
            dualnorm_lite=dualnorm_lite,
            spatial_mix_lite=inferred['spatial_mix_lite'],
            sa1_blocks=inferred['sa1_blocks'],
            sa2_blocks=inferred['sa2_blocks'],
            sa3_blocks=inferred['sa3_blocks'],
        ).to(device)
        model_label = "Biome"

    print(
        f"Loading {model_label} model | C={inferred['c_base']} | K={inferred['num_kernel_points']} | "
        f"learnable_kernels={inferred['learnable_kernels']} | spatial_mix_lite={inferred['spatial_mix_lite']} | dualnorm_lite={dualnorm_lite}"
    )

    try:
        load_model(model, checkpoint_state)
    except KeyError:
        raise Exception(f'No model loaded at {os.path.join(args.wdir,"model",args.model)}')

    test_loader, test_dataset = create_inference_loader(args, device)

    model.eval()
    output_list = []
    default_grid_size = args.grid_size[0] if isinstance(args.grid_size, (list, tuple)) else args.grid_size

    with tqdm(total=len(test_loader), colour='white', ascii="▒█", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', desc="Inference") as pbar:
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device, non_blocking=True)
            # Use batch voxel_size when collate set it (single-resolution batch); else fallback to args
            if getattr(data, 'voxel_size', None) is None:
                data.voxel_size = torch.tensor([float(default_grid_size)], dtype=torch.float32, device=data.pos.device)
            elif data.voxel_size.device != data.pos.device:
                data.voxel_size = data.voxel_size.to(data.pos.device)

            with torch.no_grad(), torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                outputs = model(data)
                outputs = torch.nan_to_num(outputs)
                probs = torch.sigmoid(outputs).float()

            # Move to CPU once
            batch_ids = data.batch.cpu()
            pos = data.pos.cpu().numpy()
            probs_np = probs.cpu().numpy()
            preds_np = (probs_np >= args.is_wood).astype(np.int64)
            local_shift = data.local_shift.cpu().numpy()

            # Split by sub-batch using index boundaries (avoids repeated boolean masking)
            batch_counts = torch.bincount(batch_ids)
            splits = torch.cumsum(batch_counts, dim=0).numpy()
            starts = np.concatenate(([0], splits[:-1]))

            for b, (s, e) in enumerate(zip(starts, splits)):
                shift = local_shift[3 * b : 3 * b + 3]
                outputb = np.column_stack((pos[s:e] + shift, preds_np[s:e], probs_np[s:e]))
                output_list.append(outputb)

            del data, outputs, probs

            if torch.cuda.is_available() and (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

            pbar.update(1)

    classified_pc = np.vstack(output_list)

    # Force garbage collection to reduce VMS bloat
    del output_list
    gc.collect()

    
    if args.verbose: print("Spatially aggregating prediction probabilites and labels...")

    # Default: argmax |p-0.5| per voxel. If --any-wood passed: label wood when any point in voxel >= any_wood.
    grid_size = getattr(args, 'collect_grid_size', 0.04) or 0.04
    use_any_wood = getattr(args, 'any_wood', None) is not None
    grid_classifier = GridCloudClassifier(
        is_wood=args.is_wood,
        any_wood=args.any_wood if use_any_wood else 0.5,
        grid_size=grid_size,
        max_probability=not use_any_wood,
    )
    args.pc = grid_classifier.collect_predictions(classified_pc, args.pc)

    headers = list(dict.fromkeys(args.headers + ['prediction', 'pwood']))
    save_file(args.odir, args.pc.copy(), additional_fields=headers, verbose=False)

    return args
