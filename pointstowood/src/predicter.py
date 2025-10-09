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
import resource
import psutil
import gc
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter_max

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.setrecursionlimit(10 ** 8)        
        
from collections import OrderedDict

# Memory tracking removed for clean output

def load_model(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    adjusted_state_dict = OrderedDict()
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('module.'):
            key = key[7:]
        adjusted_state_dict[key] = value
    model.load_state_dict(adjusted_state_dict, strict=False)
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

        indices_file = os.path.join('nbrs.npy')

        if os.path.exists(indices_file):
            indices = np.load(indices_file)
        else:
            kd_tree = KDTree(classification[:, :3])
            k = 16 if self.any_wood != 1 else 16
            _, indices = kd_tree.query(original.values[:, :3], k=k)

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
        neg_inf = torch.full((n_orig,), float('-inf'), device='cpu')
        prob_for_max = torch.cat([neg_inf, class_prob], dim=0)
        max_prob, _ = scatter_max(prob_for_max, cluster, dim=0)
        voxel_label = (max_prob >= self.any_wood).to(torch.int64)
        voxel_prob = torch.clamp(max_prob, 0.0, 1.0)

        orig_cluster = cluster[:n_orig]
        point_labels = voxel_label[orig_cluster].numpy()
        point_probs = voxel_prob[orig_cluster].numpy()

        original.loc[:, ['prediction', 'pwood']] = np.stack([point_labels, point_probs], axis=1)
        return original

def SemanticSegmentation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Auto-detect model type
    if 'eu' in args.model.lower():
        from src.model import NetFull as Net
        model = Net(num_classes=1, C=64, num_kernel_points=32, learnable_kernels=False).to(device)  # Full EU model
    else:
        from src.model import NetLight as Net
        model = Net(num_classes=1, C=16, num_kernel_points=8, learnable_kernels=True).to(device)  # Lightweight biome model

    print(f'Loading {"EU" if "eu" in args.model.lower() else "Biome"} model')

    try:
        load_model(os.path.join(args.wdir,'model',args.model), model, device)
    except KeyError:
        raise Exception(f'No model loaded at {os.path.join(args.wdir,"model",args.model)}')

    test_loader, test_dataset = create_inference_loader(args, device)

    model.eval()
    output_list = []

    with tqdm(total=len(test_loader), colour='white', ascii="▒█", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', desc="Inference") as pbar:
        for batch_idx, data in enumerate(test_loader):
            # Async GPU transfer for better overlap
            data = data.to(device, non_blocking=True)

            # Use no_grad() for inference to prevent gradient storage
            with torch.no_grad():
                outputs = model(data)
                outputs = torch.nan_to_num(outputs)

                probs = torch.sigmoid(outputs)
                preds = (probs >= args.is_wood).type(torch.int64).cpu()
                preds = np.expand_dims(preds, axis=1)

                batches = np.unique(data.batch.cpu())
                pos = data.pos.cpu().numpy()
                probs_2d = np.expand_dims(probs.cpu().numpy(), axis=1)
                output = np.concatenate((pos, preds, probs_2d), axis=1)

            for batch in batches:
                outputb = np.asarray(output[data.batch.cpu() == batch])
                outputb[:, :3] = outputb[:, :3] + np.asarray(data.local_shift.cpu())[3 * batch : 3 + (3 * batch)]
                output_list.append(outputb)

            # Explicit cleanup of batch tensors to reduce VMS fragmentation
            del data, outputs, probs, preds, pos, probs_2d, output

            # Clear GPU cache after every batch for maximum memory efficiency
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()  # Force CPU memory cleanup too

            pbar.update(1)

    classified_pc = np.vstack(output_list)

    # Force garbage collection to reduce VMS bloat
    del output_list
    gc.collect()

    
    if args.verbose: print("Spatially aggregating prediction probabilites and labels...")

    # If collection voxelization is requested, replace KNN aggregation
    if hasattr(args, 'collect_grid_size') and args.collect_grid_size and args.collect_grid_size > 0:
        grid_classifier = GridCloudClassifier(
            is_wood=args.is_wood,
            any_wood=args.any_wood,
            grid_size=args.collect_grid_size,
            max_probability=getattr(args, 'max_probability', False),
        )
        args.pc = grid_classifier.collect_predictions(classified_pc, args.pc)
    else:
        # Default: KNN-based aggregation
        classifier = KnnCloudClassifier(
            is_wood=args.is_wood,
            any_wood=args.any_wood,
            max_probability=getattr(args, 'max_probability', False),
        )
        # Set hysteresis if enabled
        if getattr(args, 'hysteresis', False):
            classifier.use_hysteresis = True

        args.pc = classifier.collect_predictions(classified_pc, args.pc)

    headers = list(dict.fromkeys(args.headers + ['prediction', 'pwood']))
    save_file(args.odir, args.pc.copy(), additional_fields=headers, verbose=False)

    return args