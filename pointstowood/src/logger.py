import numpy as np
import torch
import os
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, average_precision_score, matthews_corrcoef
from collections import OrderedDict


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]

    iou_wood = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    iou_leaf = TN / (TN + FP + FN) if (TN + FP + FN) > 0 else 0.0
    miou = (iou_wood + iou_leaf) / 2

    accuracy = balanced_accuracy_score(y_true, y_pred, sample_weight=None)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    Fbeta = fbeta_score(y_true, y_pred, beta=0.5, average='binary', zero_division=0)  # beta<1 prioritises precision
    precision_wood = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_wood = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    return accuracy, precision, recall, f1, miou, precision_wood, recall_wood, Fbeta, mcc, fpr


class MetricsTracker:
    """Tracks and accumulates metrics during training/testing.

    Args:
        full_metrics: If True, accumulate all predictions for exact AUPRC/mIoU (validation).
                      If False, only track fast threshold-based metrics (training).
    """

    def __init__(self, full_metrics: bool = False, edge_voxel_size: float = 0.25):
        self.full_metrics = full_metrics
        self.edge_voxel_size = edge_voxel_size  # Voxel size used for edge detection (0.25m default)
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self.loss = 0.0
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.miou = 0.0
        self.precision_wood = 0.0
        self.recall_wood = 0.0
        self.fbeta = 0.0
        self.mcc = 0.0
        self.fpr = 0.0
        self.brier_score = 0.0
        self.num_batches = 0
        # Only allocate for full metrics (validation)
        if self.full_metrics:
            self.all_probs = []
            self.all_labels = []
            self.all_edge_scores = []
            self.all_positions = []  # For spatial ASD computation

    def update(self, loss, outputs, targets, edge_scores=None, pos=None):
        """Update metrics with new batch results."""
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.50).int()
            y_true = (targets >= 0.5).cpu().numpy().astype(int)
            y_pred = preds.cpu().numpy().astype(int)

            acc, prec, rec, f1, miou, prw, rew, fbeta, mcc, fpr = calculate_metrics(y_true, y_pred)

            self.loss += loss.item() if hasattr(loss, 'item') else loss
            self.accuracy += acc
            self.precision += prec
            self.recall += rec
            self.f1 += f1
            self.miou += miou
            self.precision_wood += prw
            self.recall_wood += rew
            self.fbeta += fbeta
            self.mcc += mcc
            self.fpr += fpr
            brier = ((probs.cpu() - torch.from_numpy(y_true).float()) ** 2).mean().item()
            self.brier_score += brier
            self.num_batches += 1

            # Only accumulate for AUPRC/stratified-mIoU in validation mode
            if self.full_metrics:
                self.all_probs.append(probs.cpu())
                self.all_labels.append(torch.from_numpy(y_true))
                if edge_scores is not None:
                    self.all_edge_scores.append(edge_scores.cpu())
                if pos is not None:
                    self.all_positions.append(pos.cpu() if hasattr(pos, 'cpu') else pos)
    
    def update_with_smoothing(self, loss, outputs, targets, pos, batch, edge_scores=None):
        """Update metrics with KNN-smoothed predictions for testing."""
        with torch.no_grad():
            probs = torch.sigmoid(outputs).detach()

            from torch_geometric.nn import knn
            import torch_scatter
            row, col = knn(pos[:, :3], pos[:, :3], 8, batch, batch)
            idx = row.view(-1)
            conf = torch.abs(probs - 0.5) + 1e-3
            w_sum = torch_scatter.scatter_add(conf[col] * probs[col], idx, dim=0)
            w_cnt = torch_scatter.scatter_add(conf[col], idx, dim=0)
            smoothed = (w_sum / w_cnt).clamp(0, 1)

            preds = (smoothed >= 0.50).type(torch.int64).detach()
            y_true = (targets >= 0.5).cpu().numpy().astype(int)
            y_pred = preds.cpu().numpy().astype(int)

            acc, prec, rec, f1, miou, prw, rew, fbeta, mcc, fpr = calculate_metrics(y_true, y_pred)

            self.loss += loss.item() if hasattr(loss, 'item') else loss
            self.accuracy += acc
            self.precision += prec
            self.recall += rec
            self.f1 += f1
            self.miou += miou
            self.precision_wood += prw
            self.recall_wood += rew
            self.fbeta += fbeta
            self.mcc += mcc
            self.fpr += fpr
            brier = ((smoothed.cpu() - torch.from_numpy(y_true).float()) ** 2).mean().item()
            self.brier_score += brier
            self.num_batches += 1

            if self.full_metrics:
                self.all_probs.append(smoothed.cpu())
                self.all_labels.append(torch.from_numpy(y_true))
                if edge_scores is not None:
                    self.all_edge_scores.append(edge_scores.cpu())
                if pos is not None:
                    self.all_positions.append(pos.cpu())

    def _get_accumulated_data(self):
        """Get accumulated data as numpy arrays (validation only)."""
        if not self.full_metrics or len(self.all_probs) == 0:
            return None, None, None, None
        probs = torch.cat(self.all_probs).numpy()
        labels = torch.cat(self.all_labels).numpy()
        edges = torch.cat(self.all_edge_scores).numpy() if self.all_edge_scores else None
        positions = torch.cat(self.all_positions).numpy() if self.all_positions else None
        return probs, labels, edges, positions

    def get_auprc(self):
        """Calculate AUPRC from accumulated predictions."""
        probs, labels, _, _ = self._get_accumulated_data()
        if probs is None:
            return 0.0
        try:
            return average_precision_score(labels, probs)
        except ValueError:
            return 0.0

    def get_edge_auprc(self):
        """Calculate AUPRC for edge points only."""
        probs, labels, edges, _ = self._get_accumulated_data()
        if probs is None or edges is None:
            return 0.0, 0.0

        edge_mask = edges > 0.5
        non_edge_mask = ~edge_mask

        try:
            edge_auprc = average_precision_score(labels[edge_mask], probs[edge_mask]) if edge_mask.sum() > 0 else 0.0
        except ValueError:
            edge_auprc = 0.0

        try:
            non_edge_auprc = average_precision_score(labels[non_edge_mask], probs[non_edge_mask]) if non_edge_mask.sum() > 0 else 0.0
        except ValueError:
            non_edge_auprc = 0.0

        return edge_auprc, non_edge_auprc

    def get_edge_fbeta(self, beta=0.5):
        """Calculate Fbeta for edge points only (beta<1 prioritises precision)."""
        probs, labels, edges, _ = self._get_accumulated_data()
        if probs is None or edges is None:
            return 0.0
        edge_mask = edges > 0.5
        if edge_mask.sum() == 0:
            return 0.0
        preds = (probs[edge_mask] >= 0.5).astype(int)
        try:
            return fbeta_score(labels[edge_mask], preds, beta=beta, zero_division=0)
        except ValueError:
            return 0.0

    def get_edge_asd(self, threshold=0.5):
        """
        Average Surface Distance: spatial quality of ALL misclassifications.
        
        Measures average 3D distance from every misclassified point (anywhere, not
        just edges) to the nearest correctly-classified point of the SAME true class.
        This catches both:
        - Boundary errors (misclassified points near wood/leaf transitions)
        - Isolated hallucinations (leaf predicted as wood far from any real wood)
        
        Isolated FPs far from any correct wood will have large distances, pulling ASD up.
        Boundary errors near correct points will have small distances.
        
        Normalized by the edge voxel size (0.25m by default).
        
        Returns normalized distance [0, 1] where:
        - 0.0 = perfect (no misclassifications)
        - low  = errors are near boundaries (small spatial error)
        - high = errors are far from correct examples (hallucinations)
        - 1.0  = clipped maximum
        
        Lower is better.
        """
        probs, labels, edges, positions = self._get_accumulated_data()
        if probs is None:
            return 1.0
        
        preds = (probs >= 0.5).astype(int)
        errors = preds != labels
        correct = ~errors
        
        if errors.sum() == 0:
            return 0.0  # Perfect
        if correct.sum() == 0:
            return 1.0  # Everything wrong
        
        if positions is not None and positions.shape[0] == len(preds):
            try:
                from scipy.spatial import cKDTree
                
                error_positions = positions[errors, :3]
                error_true_labels = labels[errors]
                
                all_distances = np.full(errors.sum(), fill_value=float('inf'))
                
                # For each class: distance from misclassified points to nearest
                # correctly-classified point of the SAME true class.
                # E.g., leaf FP (true=leaf, pred=wood) → distance to nearest correct leaf.
                for cls in [0, 1]:
                    cls_errors = error_true_labels == cls
                    cls_correct = correct & (labels == cls)
                    if cls_errors.sum() == 0 or cls_correct.sum() == 0:
                        continue
                    tree = cKDTree(positions[cls_correct, :3])
                    dists, _ = tree.query(error_positions[cls_errors], k=1)
                    all_distances[cls_errors] = dists
                
                # Replace any inf (no correct point of same class) with max normalization
                all_distances[~np.isfinite(all_distances)] = self.edge_voxel_size
                
                avg_distance = np.mean(all_distances)
                normalized_distance = np.clip(avg_distance / self.edge_voxel_size, 0.0, 1.0)
                return float(normalized_distance)
                
            except ImportError:
                pass
        
        # Fallback: simple error rate
        return float(errors.sum() / len(preds))

    def get_edge_coherency(self, beta=0.5):
        """
        Combined metric: Edge classification quality ⊗ Boundary spatial crispness.
        
        Uses harmonic mean for balanced combination:
        edge_coherency = 2 / (1/fbeta_edge + 1/(1-asd_edge))
        
        High score requires BOTH:
        - High fbeta_edge (correct classification at boundaries)
        - Low asd_edge (small spatial distance of errors from correct boundary)
        
        ASD is true Average Surface Distance (3D Euclidean distance in metres)
        from misclassified edge points to nearest correct edge point, normalized
        by the edge voxel size (0.25m). Harmonic mean penalizes imbalance.
        
        Range: [0, 1] where 1 = perfect (crisp + spatially accurate boundaries).
        Only evaluated on edge points (transition zones), ignores pure regions.
        
        This metric captures what matters for production forestry: boundaries
        should be in the right place (low ASD) AND correctly classified (high Fbeta).
        """
        fbeta_edge = self.get_edge_fbeta(beta=beta)
        asd_edge = self.get_edge_asd()
        
        # Convert ASD (error rate) to accuracy: 1 - asd
        boundary_accuracy = 1.0 - asd_edge
        
        # Handle edge cases
        if fbeta_edge <= 0 or boundary_accuracy <= 0:
            return 0.0
        
        # Harmonic mean: 2 / (1/a + 1/b)
        coherency = 2.0 / ((1.0 / fbeta_edge) + (1.0 / boundary_accuracy))
        return coherency

    def get_stratified_miou(self):
        """Calculate stratified mIoU: separate IoU for pure vs mixed (edge) regions."""
        probs, labels, edges, _ = self._get_accumulated_data()
        if probs is None or edges is None:
            return 0.0, {}

        preds = (probs >= 0.5).astype(int)

        edge_mask = edges > 0.5
        pure_mask = ~edge_mask

        def compute_iou(y_true, y_pred, pos_label):
            """Compute IoU for a specific class."""
            if len(y_true) == 0:
                return 0.0
            tp = ((y_pred == pos_label) & (y_true == pos_label)).sum()
            fp = ((y_pred == pos_label) & (y_true != pos_label)).sum()
            fn = ((y_pred != pos_label) & (y_true == pos_label)).sum()
            return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        # Pure regions (isolated structures - easy)
        iou_wood_pure = compute_iou(labels[pure_mask], preds[pure_mask], 1) if pure_mask.sum() > 0 else 0.0
        iou_leaf_pure = compute_iou(labels[pure_mask], preds[pure_mask], 0) if pure_mask.sum() > 0 else 0.0

        # Mixed/edge regions (interlaced structures - hard)
        iou_wood_edge = compute_iou(labels[edge_mask], preds[edge_mask], 1) if edge_mask.sum() > 0 else 0.0
        iou_leaf_edge = compute_iou(labels[edge_mask], preds[edge_mask], 0) if edge_mask.sum() > 0 else 0.0

        # Stratified mIoU: equal weight to all 4 components
        components = [iou_wood_pure, iou_leaf_pure, iou_wood_edge, iou_leaf_edge]
        valid_components = [c for c in components if c > 0]
        stratified_miou = np.mean(valid_components) if valid_components else 0.0

        breakdown = {
            'iou_wood_pure': iou_wood_pure,
            'iou_leaf_pure': iou_leaf_pure,
            'iou_wood_edge': iou_wood_edge,
            'iou_leaf_edge': iou_leaf_edge,
            'miou_pure': (iou_wood_pure + iou_leaf_pure) / 2 if (iou_wood_pure + iou_leaf_pure) > 0 else 0.0,
            'miou_edge': (iou_wood_edge + iou_leaf_edge) / 2 if (iou_wood_edge + iou_leaf_edge) > 0 else 0.0,
        }

        return stratified_miou, breakdown

    def get_stratified_mcc(self):
        """Calculate MCC separately for pure and edge regions."""
        probs, labels, edges, _ = self._get_accumulated_data()
        if probs is None or edges is None:
            return 0.0, 0.0

        preds = (probs >= 0.5).astype(int)
        edge_mask = edges > 0.5
        pure_mask = ~edge_mask

        def _mcc(y_true, y_pred):
            if len(y_true) == 0 or len(set(y_true)) < 2:
                return 0.0
            try:
                from sklearn.metrics import matthews_corrcoef
                return matthews_corrcoef(y_true, y_pred)
            except Exception:
                return 0.0

        mcc_pure = _mcc(labels[pure_mask], preds[pure_mask]) if pure_mask.sum() > 0 else 0.0
        mcc_edge = _mcc(labels[edge_mask], preds[edge_mask]) if edge_mask.sum() > 0 else 0.0
        return mcc_pure, mcc_edge

    def get_stratified_fpr(self):
        """Calculate stratified FPR: separate FPR for pure vs mixed (edge) regions.

        FPR = FP / (FP + TN) = false positives among actual negatives (leaves)

        Returns:
            fpr_pure: FPR in pure regions (should be very low)
            fpr_edge: FPR in edge/boundary regions (where FPs typically occur)
        """
        probs, labels, edges, _ = self._get_accumulated_data()
        if probs is None or edges is None:
            return 0.0, 0.0

        preds = (probs >= 0.5).astype(int)

        edge_mask = edges > 0.5
        pure_mask = ~edge_mask

        def compute_fpr(y_true, y_pred):
            """Compute FPR: FP / (FP + TN)"""
            # FP = predicted wood (1) but actually leaf (0)
            # TN = predicted leaf (0) and actually leaf (0)
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            return fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Pure regions
        fpr_pure = compute_fpr(labels[pure_mask], preds[pure_mask]) if pure_mask.sum() > 0 else 0.0

        # Edge/boundary regions
        fpr_edge = compute_fpr(labels[edge_mask], preds[edge_mask]) if edge_mask.sum() > 0 else 0.0

        return fpr_pure, fpr_edge

    def get_running_averages(self):
        """Get fast running averages (no expensive metrics). Use for progress bars."""
        if self.num_batches == 0:
            return {k: 0.0 for k in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'miou', 'precision_wood', 'recall_wood', 'fbeta', 'mcc', 'fpr', 'brier_score']}

        return {
            'loss': self.loss / self.num_batches,
            'accuracy': self.accuracy / self.num_batches,
            'precision': self.precision / self.num_batches,
            'recall': self.recall / self.num_batches,
            'f1': self.f1 / self.num_batches,
            'miou': self.miou / self.num_batches,
            'precision_wood': self.precision_wood / self.num_batches,
            'recall_wood': self.recall_wood / self.num_batches,
            'fbeta': self.fbeta / self.num_batches,
            'mcc': self.mcc / self.num_batches,
            'fpr': self.fpr / self.num_batches,
            'brier_score': self.brier_score / self.num_batches,
        }

    def get_averages(self):
        """Get all metrics including expensive ones (AUPRC, edge metrics). Call once at end of validation."""
        result = self.get_running_averages()
        if self.num_batches == 0:
            result.update({k: 0.0 for k in ['auprc', 'auprc_edge', 'auprc_non_edge', 'edge_ratio', 'stratified_miou', 'miou_pure', 'miou_edge', 'fbeta_edge', 'edge_asd', 'edge_coherency', 'fpr_pure', 'fpr_edge']})
            return result

        # Full metrics only computed in validation mode
        if self.full_metrics:
            # Recompute core metrics from the full accumulated prediction set.
            # Macro-averaging per-batch values (get_running_averages) distorts MCC,
            # precision, recall and fbeta whenever batches have unequal class balance.
            # Computing from all predictions at once gives the true global values.
            all_probs = torch.cat(self.all_probs)
            all_labels = torch.cat(self.all_labels)
            all_preds = (all_probs >= 0.5).int().numpy()
            y_true_all = all_labels.numpy().astype(int)
            acc_g, prec_g, rec_g, f1_g, miou_g, prw_g, rew_g, fbeta_g, mcc_g, fpr_g = calculate_metrics(y_true_all, all_preds)
            result.update({
                'accuracy': acc_g,
                'precision': prec_g,
                'recall': rec_g,
                'f1': f1_g,
                'miou': miou_g,
                'precision_wood': prw_g,
                'recall_wood': rew_g,
                'fbeta': fbeta_g,
                'mcc': mcc_g,
                'fpr': fpr_g,
            })
            edge_auprc, non_edge_auprc = self.get_edge_auprc()
            edge_ratio = edge_auprc / non_edge_auprc if non_edge_auprc > 0 else 0.0
            stratified_miou, miou_breakdown = self.get_stratified_miou()
            fpr_pure, fpr_edge = self.get_stratified_fpr()
            mcc_pure, mcc_edge = self.get_stratified_mcc()

            # Global Brier Score — proper scoring rule sensitive to calibration.
            # Directly exposes probability collapse (e.g. pwood=0.84 for true leaf points).
            brier_global = ((all_probs - all_labels.float()) ** 2).mean().item()
            result['brier_score'] = brier_global

            result.update({
                'auprc': self.get_auprc(),
                'auprc_edge': edge_auprc,
                'auprc_non_edge': non_edge_auprc,
                'edge_ratio': edge_ratio,
                'stratified_miou': stratified_miou,
                'miou_pure': miou_breakdown.get('miou_pure', 0.0),
                'miou_edge': miou_breakdown.get('miou_edge', 0.0),
                'fbeta_edge': self.get_edge_fbeta(),
                'edge_asd': self.get_edge_asd(),
                'edge_coherency': self.get_edge_coherency(),
                'fpr_pure': fpr_pure,
                'fpr_edge': fpr_edge,
                'mcc_pure': mcc_pure,
                'mcc_edge': mcc_edge,
            })
        else:
            # Placeholder zeros for training (not computed)
            result.update({
                'auprc': 0.0,
                'auprc_edge': 0.0,
                'auprc_non_edge': 0.0,
                'edge_ratio': 0.0,
                'stratified_miou': 0.0,
                'miou_pure': 0.0,
                'miou_edge': 0.0,
                'fbeta_edge': 0.0,
                'edge_asd': 1.0,  # Worst case placeholder
                'edge_coherency': 0.0,
                'fpr_pure': 0.0,
                'fpr_edge': 0.0,
                'mcc_pure': 0.0,
                'mcc_edge': 0.0,
                'brier_score': 0.0,
            })

        return result


class ModelManager:
    """Handles model saving, loading, and checkpoint management."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def load_model(self, path):
        """Load model from checkpoint (tolerates missing/unexpected keys)."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        adjusted_state_dict = OrderedDict()
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('module.'):
                key = key[7:]
            adjusted_state_dict[key] = value
        missing, unexpected = self.model.load_state_dict(adjusted_state_dict, strict=False)
        if missing:
            print(f"  New params (using init): {missing}")
        if unexpected:
            print(f"  Ignored old params: {unexpected}")
        return self.model

    def checkpoint_payload(self):
        """Serialize weights plus effective model config for exact inference reconstruction."""
        payload = {'model_state_dict': self.model.state_dict()}
        cfg = {}
        for key in (
            'model_family',
            'c_base',
            'k_neighbors',
            'stage_kernel_points',
            'num_kernel_points',
            'learnable_kernels',
            'spatial_mix_lite',
            'flash_dim',
            'memory_efficient_conv',
            'sparse_max',
            'sa1_blocks',
            'sa2_blocks',
            'sa3_blocks',
            'compressed_head',
            'compressed_head_dim',
        ):
            if hasattr(self.model, key):
                value = getattr(self.model, key)
                if value is not None:
                    if key == 'stage_kernel_points':
                        value = list(value)
                    cfg[key] = value
        if cfg:
            payload['model_config'] = cfg
        return payload

    def save_checkpoints(self, args, epoch):
        """Save checkpoint at specified epoch."""
        checkpoint_folder = os.path.join(args.wdir, 'checkpoints')
        if not os.path.isdir(checkpoint_folder):
            os.mkdir(checkpoint_folder)
        file = checkpoint_folder + '/' + f'epoch_{epoch}.pth'
        torch.save(self.checkpoint_payload(), file)
        return True

    def save_best_model(self, stat, best_stat, save_path):
        """Save model if current stat is better than best (higher is better)."""
        if stat > best_stat:
            best_stat = stat
            torch.save(self.checkpoint_payload(), save_path)
            print(f'Saving {save_path}')
        return best_stat

    def save_best_model_lower(self, stat, best_stat, save_path):
        """Save model if current stat is better than best (lower is better, e.g. FPR)."""
        if stat < best_stat:
            best_stat = stat
            torch.save(self.checkpoint_payload(), save_path)
            print(f'Saving {save_path}')
        return best_stat


class HistoryLogger:
    """Handles training history logging and saving."""
    
    def __init__(self, args):
        self.args = args
        self.history = None
    
    def log_epoch(self, epoch, lr, train_metrics, test_metrics=None):
        """Log epoch results to history."""
        epoch_results = np.array([[
            epoch, lr,
            train_metrics['loss'],
            train_metrics['accuracy'],
            train_metrics['f1'],
            train_metrics['precision'],
            train_metrics['recall']
        ]])

        if test_metrics:
            epoch_results = np.append(epoch_results, [[
                test_metrics['accuracy'],
                test_metrics['f1'],
                test_metrics['precision'],
                test_metrics['recall']
            ]], axis=1)
        
        if self.history is None:
            self.history = epoch_results
        else:
            self.history = np.vstack((self.history, epoch_results))
        
        self.save_history()
    
    def save_history(self):
        """Save training history to file."""
        try:
            history_path = os.path.join(
                self.args.wdir, 'model', 
                os.path.splitext(self.args.model)[0] + "_history.csv"
            )
            np.savetxt(history_path, self.history)
        except OSError:
            backup_path = os.path.join(
                self.args.wdir, 'model', 
                os.path.splitext(self.args.model)[0] + "_history_backup.csv"
            )
            np.savetxt(backup_path, self.history)
    
    def get_best_epoch(self, metric_idx=3):
        """Get epoch with best metric (default: accuracy)."""
        if self.history is None or len(self.history) == 0:
            return 0, 0.0
        best_idx = np.argmax(self.history[:, metric_idx])
        return int(self.history[best_idx, 0]), self.history[best_idx, metric_idx]


class WandbLogger:
    """Handles Weights & Biases logging."""
    
    def __init__(self, args):
        self.args = args
        self.wandb = None
        if args.wandb:
            import wandb
            self.wandb = wandb
            kernel_cfg = getattr(args, "num_kernel_points", None)
            self.wandb.init(
                project="PointsToWood", 
                config={
                    "architecture": "pointnet++",
                    "region": getattr(args, "region", "unknown"),
                    "grid_size": getattr(args, "grid_size", None),
                    "num_kernel_points": kernel_cfg,
                    "stage_kernel_points": kernel_cfg if isinstance(kernel_cfg, (list, tuple)) else None,
                    "epochs": args.num_epochs,
                }
            )
            self._define_metrics()

    def _define_metrics(self):
        """Set explicit x-axes and summary reducers so W&B charts behave sensibly."""
        if not self.wandb:
            return
        try:
            self.wandb.define_metric("epoch")
            self.wandb.define_metric("train_step")

            # Per-batch / per-update training traces
            self.wandb.define_metric("train/*", step_metric="train_step")

            # Epoch-level summaries and diagnostics
            self.wandb.define_metric("epoch/*", step_metric="epoch")
            self.wandb.define_metric("model/*", step_metric="epoch")
            self.wandb.define_metric("val/*", step_metric="epoch")
            self.wandb.define_metric("eval_*/*", step_metric="epoch")
            self.wandb.define_metric("group_dro/*", step_metric="epoch")
            self.wandb.define_metric("adaptive_sampling/*", step_metric="epoch")

            # Useful summaries for quick run comparison
            self.wandb.define_metric("epoch/train_loss", summary="min")
            self.wandb.define_metric("val/h4_mcc", summary="max")
            self.wandb.define_metric("val/harmonic_mcc", summary="max")
            self.wandb.define_metric("val/mcc_edge_mean", summary="max")
            self.wandb.define_metric("val/brier", summary="min")
        except Exception:
            pass

    def log_train_step(self, train_step, lr, running_metrics, batch_loss=None, ema_loss=None, extra_metrics=None):
        """Log within-epoch training traces so W&B lines update smoothly."""
        if not self.wandb:
            return

        log_dict = {
            "train_step": int(train_step),
            "train/lr": float(lr),
            "train/running_loss": float(running_metrics.get("loss", 0.0)),
            "train/balanced_acc": float(running_metrics.get("accuracy", 0.0)),
            "train/fbeta": float(running_metrics.get("fbeta", 0.0)),
            "train/precision": float(running_metrics.get("precision", 0.0)),
            "train/recall": float(running_metrics.get("recall", 0.0)),
        }
        if batch_loss is not None:
            log_dict["train/batch_loss"] = float(batch_loss)
        if ema_loss is not None:
            log_dict["train/ema_loss"] = float(ema_loss)
        if extra_metrics:
            for key, value in extra_metrics.items():
                if value is None:
                    continue
                log_dict[key] = float(value)
        self.wandb.log(log_dict)
    
    def log_epoch(self, epoch, lr, train_metrics, test_metrics=None, model_metrics=None):
        """Log epoch metrics to wandb.

        Groups:
          epoch/   — holistic per-epoch summaries
          train/   — within-epoch training traces
          model/   — flashlight gate, kernel routing
          val/     — h4_mcc, mcc split (global + edge), FPR, calibration
        """
        if not self.wandb:
            return

        log_dict = {
            "epoch": epoch,
            "epoch/lr": float(lr),
            "epoch/train_loss": float(train_metrics["loss"]),
            "epoch/train_balanced_acc": float(train_metrics["accuracy"]),
            "epoch/train_fbeta": float(train_metrics.get("fbeta", 0.0)),
        }

        if model_metrics:
            for key, value in model_metrics.items():
                if value is None:
                    continue
                log_dict[f"model/{key}"] = float(value)

        if test_metrics:
            mcc_with = float(test_metrics.get("mcc_with_refl", 0))
            mcc_no = float(test_metrics.get("mcc_no_refl", 0))
            mcc_edge_with = float(test_metrics.get("mcc_edge_with_refl", 0))
            mcc_edge_no = float(test_metrics.get("mcc_edge_no_refl", 0))
            mcc_pure_with = float(test_metrics.get("mcc_pure_with_refl", 0))
            mcc_pure_no = float(test_metrics.get("mcc_pure_no_refl", 0))
            fpr_with = float(test_metrics.get("fpr_with_refl", 0))
            fpr_no = float(test_metrics.get("fpr_no_refl", 0))
            fpr_edge_with = float(test_metrics.get("fpr_edge_with_refl", 0))
            fpr_edge_no = float(test_metrics.get("fpr_edge_no_refl", 0))
            refl_gain_edge = mcc_edge_with - mcc_edge_no
            log_dict.update({
                "val/h4_mcc":               float(test_metrics.get("h4_mcc", 0)),
                "val/harmonic_mcc":         float(test_metrics.get("harmonic_mcc", 0)),
                "val/mcc_with_refl":        mcc_with,
                "val/mcc_no_refl":          mcc_no,
                "val/refl_gain":            mcc_with - mcc_no,
                "val/mcc_pure_with_refl":   mcc_pure_with,
                "val/mcc_pure_no_refl":     mcc_pure_no,
                "val/mcc_pure_mean":        0.5 * (mcc_pure_with + mcc_pure_no),
                "val/mcc_edge_with_refl":   mcc_edge_with,
                "val/mcc_edge_no_refl":     mcc_edge_no,
                "val/mcc_edge_mean":        0.5 * (mcc_edge_with + mcc_edge_no),
                "val/refl_gain_edge":       refl_gain_edge,
                "val/fpr_mean":             0.5 * (fpr_with + fpr_no),
                "val/fpr_edge_mean":        0.5 * (fpr_edge_with + fpr_edge_no),
                "val/brier":                float(test_metrics.get("mean_brier", test_metrics.get("brier_score", 0))),
                "val/refl_dominance":        float(test_metrics.get("refl_dominance", 0)),
                "val/refl_dominance_edge":   float(test_metrics.get("refl_dominance_edge", 0)),
                "val/refl_dominance_pure":   float(test_metrics.get("refl_dominance_pure", 0)),
                "val/refl_dominance_edge_wood": float(test_metrics.get("refl_dominance_edge_wood", 0)),
                "val/refl_dominance_edge_ratio": float(test_metrics.get("refl_dominance_edge_ratio", 0)),
            })

        self.wandb.log(log_dict)
