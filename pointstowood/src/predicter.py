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


class StreamingGridAggregator:
    """Stream predictions directly to aggregation without vstack intermediate.

    Accumulates voxel-level statistics batch-by-batch, avoiding large intermediate arrays.
    """
    def __init__(self, grid_size: float, max_probability: bool = False, any_wood: float = 0.5):
        self.grid_size = grid_size
        self.max_probability = max_probability
        self.any_wood = any_wood
        # Store per-voxel statistics: {voxel_id: {'probs': [], 'preds': []}}
        self.voxel_data = {}
        self.voxel_coords = {}  # Track voxel centers for later remapping

    def add_batch(self, pos: np.ndarray, probs: np.ndarray, preds: np.ndarray,
                  orig_pos: np.ndarray = None):
        """Add a batch of predictions, aggregate to voxels immediately.

        Args:
            pos: Predicted voxel positions [N, 3]
            probs: Probabilities [N]
            preds: Binary predictions [N]
            orig_pos: Original point positions [M, 3] for voxel assignment.
                     If None, use pos for voxel grid.
        """
        pos_torch = torch.as_tensor(pos, dtype=torch.float32)
        probs_torch = torch.as_tensor(probs, dtype=torch.float32)
        preds_torch = torch.as_tensor(preds, dtype=torch.int64)

        # Assign predicted points to voxels
        cluster_pred = voxel_grid(pos_torch, self.grid_size)

        # If we have original points, also assign them and merge voxel spaces
        if orig_pos is not None:
            orig_torch = torch.as_tensor(orig_pos, dtype=torch.float32)
            combined = torch.cat([orig_torch, pos_torch], dim=0)
            cluster_combined = voxel_grid(combined, self.grid_size)
            cluster_combined, _ = consecutive_cluster(cluster_combined)

            cluster_pred = cluster_combined[len(orig_pos):]
            self.orig_cluster = cluster_combined[:len(orig_pos)]

        # Accumulate per-voxel data
        for vid in range(int(cluster_pred.max().item()) + 1):
            mask = cluster_pred == vid
            if mask.any():
                if vid not in self.voxel_data:
                    self.voxel_data[vid] = {'probs': [], 'preds': []}
                self.voxel_data[vid]['probs'].append(probs_torch[mask].numpy())
                self.voxel_data[vid]['preds'].append(preds_torch[mask].numpy())

    def finalize(self, original: pd.DataFrame) -> pd.DataFrame:
        """Compute final voxel labels and map back to original points."""
        original = original.drop(columns=[c for c in original.columns if c in ['prediction', 'pwood', 'pleaf']])

        n_voxels = len(self.voxel_data)
        if n_voxels == 0:
            original.loc[:, ['prediction', 'pwood']] = np.column_stack([
                np.zeros(len(original), dtype=np.int64),
                np.full(len(original), 0.5, dtype=np.float32)
            ])
            return original

        voxel_labels = np.zeros(n_voxels, dtype=np.int64)
        voxel_probs = np.zeros(n_voxels, dtype=np.float32)

        for vid, data in self.voxel_data.items():
            probs = np.concatenate(data['probs'])
            preds = np.concatenate(data['preds'])

            if self.max_probability:
                conf = np.abs(probs - 0.5)
                best_idx = np.argmax(conf)
                voxel_labels[vid] = preds[best_idx]
                voxel_probs[vid] = probs[best_idx]
            else:
                voxel_probs[vid] = np.median(probs)
                voxel_labels[vid] = 1 if voxel_probs[vid] >= self.any_wood else 0

        # Map voxel predictions back to original points
        orig_labels = voxel_labels[self.orig_cluster.numpy()]
        orig_probs = voxel_probs[self.orig_cluster.numpy()]

        original.loc[:, ['prediction', 'pwood']] = np.column_stack([orig_labels, orig_probs])
        return original


class GridCloudClassifier:
    """Aggregate per-voxel. When max_probability=True: argmax |p-0.5| per voxel (matches trainer eval)."""
    def __init__(self, is_wood: float, any_wood: float, grid_size: float, max_probability: bool = False):
        self.is_wood = is_wood
        self.any_wood = any_wood
        self.grid_size = grid_size
        self.max_probability = max_probability

    def collect_predictions(self, classified_pc: np.ndarray, original: pd.DataFrame) -> pd.DataFrame:
        """Aggregate predictions to voxels. Avoids large intermediate vstack by working with numpy."""
        original = original.drop(columns=[c for c in original.columns if c in ['prediction', 'pwood', 'pleaf']])

        # Work entirely in numpy to avoid unnecessary torch conversions
        orig_pos_np = original[['x','y','z']].values.astype(np.float32)
        class_pos_np = classified_pc[:, :3].astype(np.float32)
        class_prob_np = np.clip(classified_pc[:, -1], 0.0, 1.0)
        class_pred_np = classified_pc[:, -2].astype(np.int64)

        # Assign both to voxel grid
        combined_pos = np.vstack([orig_pos_np, class_pos_np])
        combined_pos_torch = torch.as_tensor(combined_pos, dtype=torch.float32)
        cluster = voxel_grid(combined_pos_torch, self.grid_size)
        cluster, _ = consecutive_cluster(cluster)
        cluster_np = cluster.numpy()

        n_orig = len(orig_pos_np)
        n_clusters = int(cluster.max().item()) + 1
        class_cluster_np = cluster_np[n_orig:]
        orig_cluster_np = cluster_np[:n_orig]

        if self.max_probability:
            # Argmax |p-0.5| per voxel; pwood from median
            conf = np.abs(class_prob_np - 0.5)
            voxel_label = np.zeros(n_clusters, dtype=np.int64)
            voxel_prob = np.zeros(n_clusters, dtype=np.float32)

            # Sort-based grouping: compute argmax and median per voxel
            sort_idx = np.argsort(class_cluster_np, kind='stable')
            sorted_clusters = class_cluster_np[sort_idx]
            sorted_conf = conf[sort_idx]
            sorted_probs = class_prob_np[sort_idx]
            sorted_preds = class_pred_np[sort_idx]

            boundaries = np.flatnonzero(np.diff(sorted_clusters)) + 1
            conf_groups = np.split(sorted_conf, boundaries)
            prob_groups = np.split(sorted_probs, boundaries)
            pred_groups = np.split(sorted_preds, boundaries)
            unique_clusters = sorted_clusters[np.concatenate(([0], boundaries))]

            for cid, conf_grp, prob_grp, pred_grp in zip(unique_clusters, conf_groups, prob_groups, pred_groups):
                best_idx = np.argmax(conf_grp)
                voxel_label[int(cid)] = pred_grp[best_idx]
                voxel_prob[int(cid)] = np.median(prob_grp)
        else:
            # Max probability across cluster
            voxel_prob = np.full(n_clusters, -np.inf, dtype=np.float32)
            for i, cid in enumerate(class_cluster_np):
                voxel_prob[cid] = max(voxel_prob[cid], class_prob_np[i])
            voxel_prob = np.clip(voxel_prob, 0.0, 1.0)
            voxel_label = (voxel_prob >= self.any_wood).astype(np.int64)

        # Map back to original points
        point_labels = voxel_label[orig_cluster_np]
        point_probs = voxel_prob[orig_cluster_np]

        original.loc[:, ['prediction', 'pwood']] = np.column_stack([point_labels, point_probs])
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
    default_grid_size = args.grid_size[0] if isinstance(args.grid_size, (list, tuple)) else args.grid_size

    # Pre-allocate aggregation to avoid vstack
    if args.verbose: print("Running inference and streaming predictions to aggregation...")

    grid_size = getattr(args, 'collect_grid_size', 0.04) or 0.04
    use_any_wood = getattr(args, 'any_wood', None) is not None

    # Collect all predictions first (unavoidable for voxel remapping), but do minimal processing
    all_pos = []
    all_preds = []
    all_probs = []

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

            # Move to CPU once, extract only what's needed
            batch_ids = data.batch.cpu().numpy()
            pos = data.pos[:, :3].cpu().numpy()  # Only xyz, not reflectance
            probs_np = probs.cpu().numpy().astype(np.float32)
            preds_np = (probs_np >= args.is_wood).astype(np.int64)
            local_shift = data.local_shift.cpu().numpy()

            # Split by sub-batch and append directly (no column_stack intermediate)
            batch_counts = np.bincount(batch_ids, minlength=batch_ids.max() + 1)
            start = 0
            for b, count in enumerate(batch_counts):
                if count > 0:
                    end = start + count
                    shift = local_shift[3 * b : 3 * b + 3]
                    all_pos.append(pos[start:end] + shift)
                    all_preds.append(preds_np[start:end])
                    all_probs.append(probs_np[start:end])
                    start = end

            del data, outputs, probs

            if torch.cuda.is_available() and (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

            pbar.update(1)

    # Concatenate all predictions (unavoidable, but once at the end)
    classified_pos = np.concatenate(all_pos, dtype=np.float32)
    classified_preds = np.concatenate(all_preds, dtype=np.int64)
    classified_probs = np.concatenate(all_probs, dtype=np.float32)

    del all_pos, all_preds, all_probs
    gc.collect()

    if args.verbose: print("Spatially aggregating predictions to voxels...")

    # Single vstack-equivalent at aggregation time, working with numpy for speed
    grid_classifier = GridCloudClassifier(
        is_wood=args.is_wood,
        any_wood=args.any_wood if use_any_wood else 0.5,
        grid_size=grid_size,
        max_probability=not use_any_wood,
    )

    # Pass all predictions at once for efficient aggregation
    classified_pc = np.column_stack([classified_pos, classified_preds, classified_probs])
    args.pc = grid_classifier.collect_predictions(classified_pc, args.pc)

    headers = list(dict.fromkeys(args.headers + ['prediction', 'pwood']))
    save_file(args.odir, args.pc.copy(), additional_fields=headers, verbose=False)

    return args
