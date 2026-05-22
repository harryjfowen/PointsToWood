from src.dataset import create_train_loader, create_test_loader, create_inference_loader, TrainingDataset, _fixed_batch_collate, FOCUS_ONLY_PREFIXES
from src.logger import MetricsTracker, ModelManager, HistoryLogger, WandbLogger
from src.statistics import calculate_harmonic_metrics, print_validation_summary, update_test_metrics_with_harmonic
from src.AnisotropicConv import AnisotropicConv
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import os
import glob
import pandas as pd
from src.loss import FocalLoss, ReflectanceFPPenalty, SupConLoss
from torch.optim import AdamW
from torch_geometric.nn import voxel_grid
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter_max, scatter_add
from torch.utils.data import Subset
from src.io import save_file
import warnings
import copy
import math




class ValidationGroupSamplerTracker:
    """Convert validation group performance into conservative sampling multipliers."""

    def __init__(
        self,
        enabled: bool = False,
        warmup_epochs: int = 20,
        ramp_epochs: int = 30,
        ema: float = 0.8,
        alpha: float = 1.0,
        min_multiplier: float = 0.75,
        max_multiplier: float = 2.0,
        metric: str = "balanced_acc",
    ):
        self.enabled = bool(enabled)
        self.warmup_epochs = max(1, int(warmup_epochs))
        self.ramp_epochs = max(1, int(ramp_epochs))
        self.ema = min(0.99, max(0.0, float(ema)))
        self.alpha = max(0.0, float(alpha))
        self.min_multiplier = float(min_multiplier)
        self.max_multiplier = float(max_multiplier)
        self.metric = metric
        self.scores = {}
        self.multipliers = {}
        self.last_status = f"warmup 0/{self.warmup_epochs}" if self.enabled else "off"
        self.last_breakdown = self.last_status

    def update(self, epoch: int, group_metrics: dict):
        if not self.enabled:
            self.last_status = "off"
            self.last_breakdown = self.last_status
            return False
        if not group_metrics:
            self.last_status = "no group metrics"
            self.last_breakdown = self.last_status
            return False

        current = {}
        for group_name, metrics in group_metrics.items():
            value = metrics.get(self.metric, metrics.get("balanced_acc", None))
            if value is None or not np.isfinite(value):
                continue
            current[group_name] = float(value)
        if not current:
            self.last_status = "no valid group metrics"
            self.last_breakdown = self.last_status
            return False

        for group_name, value in current.items():
            if group_name not in self.scores:
                self.scores[group_name] = value
            else:
                self.scores[group_name] = self.ema * self.scores[group_name] + (1.0 - self.ema) * value

        if epoch < self.warmup_epochs:
            self.multipliers = {g: 1.0 for g in self.scores}
            self.last_status = f"warmup {epoch}/{self.warmup_epochs}"
            self.last_breakdown = self._breakdown(prefix=self.last_status)
            return False

        values = np.asarray(list(self.scores.values()), dtype=np.float64)
        target = float(np.median(values))
        denom = max(abs(target), 1e-6)
        raw = {}
        for group_name, score in self.scores.items():
            deficit = target - score
            multiplier = 1.0 + self.alpha * (deficit / denom)
            raw[group_name] = float(np.clip(multiplier, self.min_multiplier, self.max_multiplier))

        raw_mean = max(float(np.mean(list(raw.values()))), 1e-6)
        ramp_t = min(1.0, (epoch - self.warmup_epochs + 1) / self.ramp_epochs)
        self.multipliers = {
            group_name: float(np.clip(1.0 + ramp_t * ((value / raw_mean) - 1.0), self.min_multiplier, self.max_multiplier))
            for group_name, value in raw.items()
        }

        worst_group = min(self.scores, key=self.scores.get)
        mult_values = list(self.multipliers.values())
        self.last_status = (
            f"active {self.metric} worst={worst_group}:{self.scores[worst_group]:.3f} "
            f"ramp={ramp_t:.2f} mult∈[{min(mult_values):.3f},{max(mult_values):.3f}]"
        )
        self.last_breakdown = self._breakdown(prefix=f"active {self.metric}")
        return True

    def status_str(self):
        return self.last_status

    def breakdown_str(self):
        return self.last_breakdown

    def _breakdown(self, prefix: str):
        if not self.scores:
            return prefix
        parts = []
        for group_name, score in sorted(self.scores.items(), key=lambda item: item[1]):
            multiplier = self.multipliers.get(group_name, 1.0)
            parts.append(f"{group_name}:{score:.3f}@{multiplier:.2f}x")
        return f"{prefix} | " + " ".join(parts)


class VoxelDifficultyTracker:
    """Per-voxel boundary-weighted loss EMA for hard example mining.

    Each training step, records per-sample boundary-weighted BCE loss indexed
    by voxel_idx (the position in train_dataset.keys). Maintains an EMA so
    voxels that are consistently hard get progressively higher sampling weight.
    Warm-starts from edge_fraction prior so cold voxels aren't initialised at 0.

    normalised_weights() returns values centred on 1.0 via tanh(z/2) so the
    deviation from 1.0 is controlled by voxel_alpha in the weight composition.
    """
    def __init__(self, n_real: int, prior=None, alpha_ema: float = 0.9):
        self.n_real = int(n_real)
        self.alpha = float(alpha_ema)
        if prior is not None and len(prior) == self.n_real:
            self.ema = np.asarray(prior, dtype=np.float64).copy()
        else:
            self.ema = np.zeros(self.n_real, dtype=np.float64)
        self.n_seen = np.zeros(self.n_real, dtype=np.int32)

    def update(self, voxel_ids: np.ndarray, losses: np.ndarray):
        for vid, loss in zip(voxel_ids, losses):
            vid = int(vid)
            if vid < 0 or vid >= self.n_real:
                continue
            loss = float(loss)
            if self.n_seen[vid] == 0:
                self.ema[vid] = loss
            else:
                self.ema[vid] = self.alpha * self.ema[vid] + (1.0 - self.alpha) * loss
            self.n_seen[vid] += 1

    def normalised_weights(self) -> np.ndarray:
        x = self.ema.copy()
        if x.size == 0:
            return np.ones(self.n_real, dtype=np.float64)
        sd = x.std()
        if np.isfinite(sd) and sd > 1e-8:
            mu = x.mean()
            z = (x - mu) / (sd + 1e-8)
            return 1.0 + np.tanh(z / 2.0)
        return np.ones(self.n_real, dtype=np.float64)

    def status_str(self) -> str:
        seen = int((self.n_seen > 0).sum())
        if seen == 0:
            return f"seen=0/{self.n_real} (prior only)"
        top_idx = int(np.argmax(self.ema))
        return (f"seen={seen}/{self.n_real} "
                f"max_ema={self.ema[top_idx]:.3f} "
                f"mean_ema={self.ema[self.n_seen > 0].mean():.3f}")


def _build_train_sampler_weights(train_dataset, edge_alpha: float = 0.0,
                                  group_multipliers: dict = None,
                                  voxel_difficulty=None, voxel_alpha: float = 0.0,
                                  coverage_tracker=None, unseen_boost: float = 0.0):
    """Compose edge-difficulty, per-voxel loss EMA, and group multipliers into per-sample weights."""
    n_real = len(getattr(train_dataset, 'keys', []))
    if n_real <= 0:
        return None

    weights = np.ones(n_real, dtype=np.float64)

    edge_fracs = np.asarray(getattr(train_dataset, 'edge_fractions', np.zeros(n_real)), dtype=np.float64)
    if edge_alpha > 0 and edge_fracs.size >= n_real:
        weights *= 1.0 + float(edge_alpha) * edge_fracs[:n_real]

    if voxel_difficulty is not None and voxel_alpha > 0:
        vw = voxel_difficulty.normalised_weights()  # centred on 1.0
        weights *= 1.0 + float(voxel_alpha) * (vw - 1.0)

    if coverage_tracker is not None and unseen_boost > 1.0:
        n_seen = getattr(coverage_tracker, 'n_seen', None)
        if n_seen is not None and len(n_seen) >= n_real:
            unseen = np.asarray(n_seen[:n_real]) <= 0
            if unseen.any() and not unseen.all():
                weights[unseen] *= float(unseen_boost)

    if group_multipliers:
        group_list = getattr(train_dataset, 'group_list', [])
        group_indices = getattr(train_dataset, 'group_indices', [])
        if group_list and len(group_indices) >= n_real:
            for idx in range(n_real):
                group_name = group_list[group_indices[idx]]
                weights[idx] *= float(group_multipliers.get(group_name, 1.0))

    weights = np.clip(weights, 1e-8, 8.0)  # cap before normalisation to prevent extreme starvation
    weights *= n_real / weights.sum()  # normalise mean to 1 so effective dataset size is stable
    mix_slots = getattr(train_dataset, '_num_mix_slots', 0)
    if mix_slots > 0:
        mix_weights = np.full(mix_slots, float(weights.mean()), dtype=np.float64)
        weights = np.concatenate([weights, mix_weights])
    return weights


def _apply_train_sampler_weights(train_loader, train_dataset, edge_alpha: float = 0.0,
                                  group_multipliers: dict = None,
                                  voxel_difficulty=None, voxel_alpha: float = 0.0,
                                  coverage_tracker=None, unseen_boost: float = 0.0,
                                  allow_replacement: bool = True):
    sampler = getattr(train_loader, 'batch_sampler', None)
    if sampler is None or not hasattr(sampler, 'set_weights'):
        return None
    weights = _build_train_sampler_weights(
        train_dataset, edge_alpha=edge_alpha, group_multipliers=group_multipliers,
        voxel_difficulty=voxel_difficulty, voxel_alpha=voxel_alpha,
        coverage_tracker=coverage_tracker, unseen_boost=unseen_boost,
    )
    if weights is None:
        return None
    try:
        sampler.set_weights(weights, allow_replacement=allow_replacement)
    except TypeError:
        sampler.set_weights(weights)
    return weights


def _per_sample_boundary_weighted_bce(logits: torch.Tensor, labels: torch.Tensor,
                                      batch: torch.Tensor, edge_scores: torch.Tensor = None) -> torch.Tensor:
    """Per-sample BCE with a light fixed emphasis on boundary points.

    This tracker is diagnostic/sampling-only, not the training objective. It uses
    raw BCE and a simple `(1 + edge_scores)` weight so mixed-label boundary voxels
    contribute more strongly to the voxel difficulty EMA regardless of the main
    loss curriculum.
    """
    point_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
    if edge_scores is not None and edge_scores.numel() == point_loss.numel():
        point_weight = 1.0 + edge_scores.float()
        point_loss = point_loss * point_weight
    else:
        point_weight = torch.ones_like(point_loss)

    n_samples = int(batch.max().item()) + 1
    per_sample_sum = scatter_add(point_loss, batch, dim=0, dim_size=n_samples)
    per_sample_weight = scatter_add(point_weight, batch, dim=0, dim_size=n_samples).clamp(min=1.0)
    return per_sample_sum / per_sample_weight

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

seed = 141190
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore", category=UserWarning)
torch.autograd.set_detect_anomaly(False)


def downsample_batch_to_point_budget(data, max_points, device):
    """Randomly subsample points in a PyG Batch so total points ≤ max_points. Preserves batch structure (consecutive batch indices)."""
    n = data.pos.size(0)
    if n <= max_points:
        return data
    indices = torch.randperm(n, device=device)[:max_points]
    # Per-node attributes
    data = data.clone()
    data.pos = data.pos[indices]
    data.batch = data.batch[indices]
    if hasattr(data, 'reflectance') and data.reflectance is not None:
        data.reflectance = data.reflectance[indices]
    if hasattr(data, 'y') and data.y is not None:
        data.y = data.y[indices]
    if hasattr(data, 'edge_scores') and data.edge_scores is not None:
        data.edge_scores = data.edge_scores[indices]
    if hasattr(data, 'x') and data.x is not None:
        data.x = data.x[indices]
    # Renumber batch to consecutive 0, 1, 2, ...
    old_batch = data.batch
    unique_old = old_batch.unique(sorted=True)
    new_batch = torch.empty_like(old_batch)
    for new_idx, old_idx in enumerate(unique_old):
        new_batch[old_batch == old_idx] = new_idx
    data.batch = new_batch
    # sf: per-graph (one per sample) -> keep only graphs that still have points
    if hasattr(data, 'sf') and data.sf is not None:
        if data.sf.numel() == old_batch.max().item() + 1:
            data.sf = data.sf[unique_old]
        elif data.sf.numel() == n:
            data.sf = data.sf[indices]
    for attr in ('voxel_idx', 'group_idx', 'difficulty'):
        value = getattr(data, attr, None)
        if value is not None and hasattr(value, 'size') and value.size(0) == old_batch.max().item() + 1:
            setattr(data, attr, value[unique_old])
    return data


def _set_batch_voxel_size(data, args):
    """Set voxel_size on batch for voxel-size-aware distance in AnisotropicConv. Use batch value when set by collate (single-resolution batching), else args."""
    existing = getattr(data, 'voxel_size', None)
    if existing is not None:
        if isinstance(existing, torch.Tensor):
            data.voxel_size = existing.to(data.pos.device)
        else:
            data.voxel_size = torch.tensor([float(existing)], dtype=torch.float32, device=data.pos.device)
        return
    grid_size = args.grid_size[0] if isinstance(args.grid_size, (list, tuple)) else args.grid_size
    data.voxel_size = torch.tensor([float(grid_size)], dtype=torch.float32, device=data.pos.device)


def _resolve_amp_config(device, amp_dtype: str = "auto"):
    """Resolve AMP enablement/dtype. auto: prefer bf16 when supported, else fp16."""
    dev = device.type if isinstance(device, torch.device) else str(device)
    use_cuda = torch.cuda.is_available() and str(dev).startswith('cuda')
    if not use_cuda:
        return False, torch.float16, "off"

    choice = str(amp_dtype).lower().strip()
    bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

    if choice == "bf16":
        if bf16_supported:
            return True, torch.bfloat16, "bf16"
        print("[AMP] bf16 requested but unsupported on this GPU; falling back to fp16.")
        return True, torch.float16, "fp16"
    if choice == "fp16":
        return True, torch.float16, "fp16"

    # auto
    if bf16_supported:
        return True, torch.bfloat16, "bf16(auto)"
    return True, torch.float16, "fp16(auto)"


def run_validation_pass(model, test_loader, device, mode_name, augmentation_mode, args, track_groups=False):
    """Run a single validation pass with specified augmentation mode."""
    # Temporarily modify the dataset mode
    original_mode = test_loader.dataset.mode
    test_loader.dataset.mode = augmentation_mode

    # Edge voxel size for ASD normalization (hardcoded in dataset.py:51 as 0.25m)
    # This is the resolution used to detect mixed-label voxels (edge points)
    test_tracker = MetricsTracker(full_metrics=True, edge_voxel_size=0.25)

    group_list = getattr(test_loader.dataset, 'group_list', None)
    group_preds: dict = {}  # group_name -> {'y_true': [], 'y_pred': []}

    # Limit validation steps (0 = use all data)
    val_steps = getattr(args, 'val_steps', 0)
    total_steps = min(len(test_loader), val_steps) if val_steps > 0 else len(test_loader)

    with tqdm(total=total_steps, colour='cyan' if 'With' in mode_name else 'yellow',
              ascii="▒█", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # Break if we've done enough steps
                if val_steps > 0 and i >= val_steps:
                    break

                data = data.to(device)
                _set_batch_voxel_size(data, args)
                model_output = model(data)

                outputs = model_output

                test_tracker.update(torch.tensor(0.0), outputs, data.y,
                                   edge_scores=getattr(data, 'edge_scores', None),
                                   pos=getattr(data, 'pos', None))

                # Accumulate per-group predictions using batch + group_idx
                if track_groups and group_list is not None and hasattr(data, 'group_idx') and data.group_idx is not None:
                    probs = torch.sigmoid(outputs)
                    y_pred = (probs >= 0.5).int().cpu().numpy()
                    y_true = (data.y >= 0.5).int().cpu().numpy()
                    batch_idx = data.batch.cpu().numpy()
                    group_ids = data.group_idx.view(-1).cpu().numpy()
                    for sample_i, gid in enumerate(group_ids):
                        gname = group_list[gid]
                        mask = batch_idx == sample_i
                        if not mask.any():
                            continue
                        if gname not in group_preds:
                            group_preds[gname] = {'y_true': [], 'y_pred': []}
                        group_preds[gname]['y_true'].append(y_true[mask])
                        group_preds[gname]['y_pred'].append(y_pred[mask])

                # Use fast running averages for progress bar (no expensive metrics)
                curr_metrics = test_tracker.get_running_averages()
                tepoch.set_description(f"Val {mode_name}")
                tepoch.update()
                tepoch.set_postfix({
                    'BAc': np.around(curr_metrics['accuracy'], 3),
                    'Pr': np.around(curr_metrics['precision'], 3),
                    'Re': np.around(curr_metrics['recall'], 3),
                    'Fbeta': np.around(curr_metrics['fbeta'], 3),
                    'MCC': np.around(curr_metrics['mcc'], 3),
                    'FPR': np.around(curr_metrics['fpr'], 3),
                })
            tepoch.close()

    # Restore original mode
    test_loader.dataset.mode = original_mode

    averages = test_tracker.get_averages()
    if track_groups and group_preds:
        group_metrics = {}
        for gname in sorted(group_preds):
            yt = np.concatenate(group_preds[gname]['y_true'])
            yp = np.concatenate(group_preds[gname]['y_pred'])
            if yt.size:
                group_metrics[gname] = _binary_site_metrics(yt, yp)
        if group_metrics:
            averages['group_metrics'] = group_metrics

    return averages


def _binary_site_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Pooled binary metrics for one focused source prefix."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    wood_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    leaf_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    wood_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    wood_f1 = 2.0 * wood_precision * wood_recall / (wood_precision + wood_recall) if (wood_precision + wood_recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    return {
        'balanced_acc': 0.5 * (wood_recall + leaf_recall),
        'wood_precision': wood_precision,
        'wood_recall': wood_recall,
        'wood_f1': wood_f1,
        'leaf_recall': leaf_recall,
        'fpr': fpr,
        'mcc': mcc,
        'true_wood_frac': float((y_true == 1).mean()) if y_true.size else 0.0,
        'pred_wood_frac': float((y_pred == 1).mean()) if y_pred.size else 0.0,
    }


def print_focus_site_eval(model, test_dataset, device, args, epoch, prefixes=('deu08', 'fin04', 'fin04-hard')):
    """Print one compact focused eval line for named shard prefixes."""
    if test_dataset is None or not hasattr(test_dataset, 'keys'):
        return

    best_by_prefix = getattr(print_focus_site_eval, '_best_balanced_acc', {})
    original_mode = getattr(test_dataset, 'mode', None)
    test_dataset.mode = 'val_with_reflectance'
    parts = []

    try:
        for prefix in prefixes:
            indices = [i for i, key in enumerate(test_dataset.keys)
                       if os.path.basename(str(key)).split('_voxel_')[0] == prefix]
            if not indices:
                continue

            subset = Subset(test_dataset, indices)
            loader = DataLoader(
                subset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=_fixed_batch_collate,
            )

            y_true_parts = []
            y_pred_parts = []
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    _set_batch_voxel_size(data, args)
                    outputs = model(data)
                    probs = torch.sigmoid(outputs)
                    y_true_parts.append((data.y >= 0.5).int().detach().cpu().numpy())
                    y_pred_parts.append((probs >= 0.5).int().detach().cpu().numpy())

            if not y_true_parts:
                continue
            metrics = _binary_site_metrics(np.concatenate(y_true_parts), np.concatenate(y_pred_parts))
            prev_best = best_by_prefix.get(prefix, float('-inf'))
            is_best = metrics['balanced_acc'] >= prev_best
            if is_best:
                best_by_prefix[prefix] = metrics['balanced_acc']
            ba_marker = '↑' if is_best else '↓'
            wood_delta_pct = 100.0 * (metrics['pred_wood_frac'] - metrics['true_wood_frac'])
            parts.append(
                f"  {prefix}: BA[{ba_marker}]={metrics['balanced_acc']:.3f} "
                f"Wood P/R={metrics['wood_precision']:.3f}/{metrics['wood_recall']:.3f} "
                f"WoodΔ={wood_delta_pct:+.1f}pp"
            )
    finally:
        if original_mode is not None:
            test_dataset.mode = original_mode

    if parts:
        print_focus_site_eval._best_balanced_acc = best_by_prefix
        print(f"Focus Eval Summary E{epoch}:")
        for part in parts:
            print(part)


def compute_refl_dominance_diagnostic(model, test_loader, device, args):
    """One-batch sensitivity of logits to reflectance, split by boundary/wood regions."""
    result = {
        "all": float("nan"),
        "edge": float("nan"),
        "pure": float("nan"),
        "edge_wood": float("nan"),
        "edge_ratio": float("nan"),
    }

    def _masked_mean(values, mask):
        if mask is None or values.numel() == 0:
            return float("nan")
        mask = mask.to(device=values.device, dtype=torch.bool)
        if mask.numel() != values.numel() or not bool(mask.any().detach().cpu().item()):
            return float("nan")
        return float(values[mask].mean().detach().cpu().item())

    model.eval()
    orig_mode = test_loader.dataset.mode
    test_loader.dataset.mode = "val_with_reflectance"
    try:
        data = next(iter(test_loader))
        data = data.to(device)
        _set_batch_voxel_size(data, args)
        data.reflectance.requires_grad_(True)
        with torch.enable_grad():
            out = model(data)
            out.sum().backward()
        if data.reflectance.grad is not None:
            grad_abs = data.reflectance.grad.detach().abs()
            result["all"] = float(grad_abs.mean().detach().cpu().item())

            edge_scores = getattr(data, "edge_scores", None)
            if edge_scores is not None and edge_scores.numel() == grad_abs.numel():
                edge_mask = edge_scores > 0.5
                pure_mask = ~edge_mask
                wood_mask = data.y >= 0.5
                result["edge"] = _masked_mean(grad_abs, edge_mask)
                result["pure"] = _masked_mean(grad_abs, pure_mask)
                result["edge_wood"] = _masked_mean(grad_abs, edge_mask & wood_mask)
                if np.isfinite(result["edge"]) and np.isfinite(result["pure"]) and result["pure"] > 0:
                    result["edge_ratio"] = result["edge"] / result["pure"]
    except Exception:
        pass
    finally:
        test_loader.dataset.mode = orig_mode
        model.zero_grad(set_to_none=True)
    return result



def _eval_metrics(classified_with_refl, classified_no_refl, gt_xyz_label, collect_grid_size, any_wood_threshold=None):
    """Compute MCC and FPR for both refl conditions vs GT at collect grid resolution.

    Concatenates pred and GT points into one voxel_grid so cluster IDs are consistent,
    then compares predictions against majority-vote GT per voxel. By default the
    prediction is argmax-|p-0.5|; with any_wood_threshold it becomes max(pwood)
    >= threshold, matching deployment any-wood aggregation.

    Args:
        classified_with_refl: [N, 4] (x, y, z, prob)
        classified_no_refl:   [N, 4] (x, y, z, prob)
        gt_xyz_label:         [M, 4] (x, y, z, label)
        collect_grid_size:    float, metres

    Returns dict: mcc_with_refl, mcc_no_refl, refl_gain, refl_gain_edge,
                  fpr_with_refl, fpr_no_refl, mcc_edge_with_refl, mcc_edge_no_refl
    """
    from torch_scatter import scatter_add as _scatter_add

    def _agg_pred(pred_arr, gt_arr, grid_size):
        pred_pos = torch.as_tensor(pred_arr[:, :3], dtype=torch.float32)
        pred_prob = torch.as_tensor(pred_arr[:, 3], dtype=torch.float32)
        gt_pos = torch.as_tensor(gt_arr[:, :3], dtype=torch.float32)
        gt_lab = torch.as_tensor(gt_arr[:, 3], dtype=torch.float32)

        # Joint voxel grid so pred and GT share cluster IDs
        all_pos = torch.cat([pred_pos, gt_pos], dim=0)
        raw_clusters = voxel_grid(all_pos, grid_size)
        clusters, _ = consecutive_cluster(raw_clusters)
        n_clusters = int(clusters.max().item()) + 1

        pred_c = clusters[:len(pred_pos)]
        gt_c   = clusters[len(pred_pos):]

        if any_wood_threshold is not None:
            pred_max, _ = scatter_max(pred_prob, pred_c, dim=0, dim_size=n_clusters)
            pred_label = (pred_max >= float(any_wood_threshold)).long()
        else:
            # Pred: argmax |p-0.5| per voxel
            conf = torch.abs(pred_prob - 0.5)
            _, argmax_idx = scatter_max(conf, pred_c, dim=0, dim_size=n_clusters)
            pred_label = (pred_prob[argmax_idx.clamp(0, len(pred_prob) - 1)] >= 0.5).long()

        # GT: majority vote per voxel
        gt_sum   = _scatter_add(gt_lab, gt_c, dim=0, dim_size=n_clusters)
        gt_count = _scatter_add(torch.ones_like(gt_lab), gt_c, dim=0, dim_size=n_clusters)
        gt_label = (gt_sum / gt_count.clamp(min=1) >= 0.5).long()

        # Valid: voxels that have both pred and GT coverage
        pred_has = _scatter_add(torch.ones(len(pred_pos)), pred_c, dim=0, dim_size=n_clusters) > 0
        gt_has   = _scatter_add(torch.ones(len(gt_pos)),   gt_c,   dim=0, dim_size=n_clusters) > 0
        valid = pred_has & gt_has

        return {
            'pred': pred_label[valid].numpy(),
            'gt': gt_label[valid].numpy(),
            'gt_pos': gt_pos,
            'gt_lab': gt_lab,
            'gt_cluster': gt_c,
            'valid': valid,
            'n_clusters': n_clusters,
        }

    def _mcc_fpr(p, g):
        tp = int(((p == 1) & (g == 1)).sum())
        tn = int(((p == 0) & (g == 0)).sum())
        fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = float(tp * tn - fp * fn) / denom if denom > 0 else 0.0
        fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
        return mcc, fpr

    def _basic_stats(p, g):
        tp = int(((p == 1) & (g == 1)).sum())
        tn = int(((p == 0) & (g == 0)).sum())
        fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        wood_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        leaf_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        wood_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fbeta = 2.0 * wood_precision * wood_recall / (wood_precision + wood_recall) if (wood_precision + wood_recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
        return {
            'accuracy': (tp + tn) / max(1, tp + tn + fp + fn),
            'balanced_accuracy': 0.5 * (wood_recall + leaf_recall),
            'precision': wood_precision,
            'recall': wood_recall,
            'fbeta': fbeta,
            'fpr': fpr,
            'mcc': mcc,
        }

    def _gt_edge_mask(meta, edge_voxel_size=0.25):
        """GT-derived edge mask projected onto eval voxels.

        A collect-grid voxel is considered an edge voxel if any GT point inside it
        belongs to a mixed-label 25 cm neighborhood, matching the training-time
        boundary definition used in the loss.
        """
        from src.pointcutmix import recompute_edge_scores

        point_edge = recompute_edge_scores(meta['gt_pos'], meta['gt_lab'], batch=None, voxel_size=edge_voxel_size)
        point_edge = (point_edge > 0.5).float()
        edge_hits = _scatter_add(point_edge, meta['gt_cluster'], dim=0, dim_size=meta['n_clusters'])
        edge_cluster = edge_hits > 0
        return edge_cluster[meta['valid']].cpu().numpy()

    if gt_xyz_label is None or len(gt_xyz_label) == 0:
        return {}

    try:
        meta_w = _agg_pred(classified_with_refl, gt_xyz_label, collect_grid_size)
        meta_n = _agg_pred(classified_no_refl,   gt_xyz_label, collect_grid_size)

        pred_w, gt_w = meta_w['pred'], meta_w['gt']
        pred_n, gt_n = meta_n['pred'], meta_n['gt']

        stats_w = _basic_stats(pred_w, gt_w)
        stats_n = _basic_stats(pred_n, gt_n)
        mcc_w, fpr_w = _mcc_fpr(pred_w, gt_w)
        mcc_n, fpr_n = _mcc_fpr(pred_n, gt_n)

        # GT-derived edge voxels: fixed boundary subset shared by both modes.
        is_edge = _gt_edge_mask(meta_w)
        mcc_edge_w = _mcc_fpr(pred_w[is_edge], gt_w[is_edge])[0] if is_edge.sum() > 10 else float('nan')
        mcc_edge_n = _mcc_fpr(pred_n[is_edge], gt_n[is_edge])[0] if is_edge.sum() > 10 else float('nan')

        return {
            'accuracy':      stats_w['accuracy'],
            'balanced_accuracy': stats_w['balanced_accuracy'],
            'precision':     stats_w['precision'],
            'recall':        stats_w['recall'],
            'fbeta':         stats_w['fbeta'],
            'fpr':           stats_w['fpr'],
            'mcc':           stats_w['mcc'],
            'mcc_with_refl':      mcc_w,
            'mcc_no_refl':        mcc_n,
            'refl_gain':          mcc_w - mcc_n,
            'fpr_with_refl':      fpr_w,
            'fpr_no_refl':        fpr_n,
            'mcc_edge_with_refl': mcc_edge_w,
            'mcc_edge_no_refl':   mcc_edge_n,
            'refl_gain_edge':     (mcc_edge_w - mcc_edge_n) if (not np.isnan(mcc_edge_w) and not np.isnan(mcc_edge_n)) else float('nan'),
        }
    except Exception as e:
        print(f'[Eval metrics] Failed: {e}')
        return {}


def run_eval_visualization(model, args, device, epoch, save_outputs: bool = True):
    """Run inference on eval voxels and aggregate to the deployment grid.

    By default this saves PLY outputs for inspection. Pass save_outputs=False to
    keep the eval pass metrics-only.
    """
    import re

    eval_vxfile = getattr(args, 'eval_vxfile', None)
    if not eval_vxfile or not os.path.isdir(eval_vxfile):
        if epoch == 1 and getattr(args, 'eval', False):
            print('[Eval] No eval voxels found; put a .ply in data/eval and run with --preprocess --eval first.')
        return

    # Group voxel payloads by source prefix. Preprocessing writes a single shard.pt containing
    # entries whose `name` is `{source}_voxel_{counter}`; fall back to per-voxel .pt globs only
    # when running in low-memory (disk_individual) mode.
    source_entries = {}  # prefix -> list[dict entries] or list[file paths]
    shard_path = os.path.join(eval_vxfile, 'shard.pt')
    if os.path.isfile(shard_path):
        try:
            shard = torch.load(shard_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f'[Eval] Failed to load shard {shard_path}: {e}')
            return
        for entry in shard:
            name = entry.get('name', '') if isinstance(entry, dict) else ''
            m = re.match(r'(.+)_voxel_\d+$', name)
            prefix = m.group(1) if m else '_default_'
            source_entries.setdefault(prefix, []).append(entry)
    else:
        all_keys = glob.glob(os.path.join(eval_vxfile, '*.pt'))
        if not all_keys:
            if epoch == 1 and getattr(args, 'eval', False):
                print('[Eval] Eval voxel folder is empty; put a .ply in data/eval and run with --preprocess --eval first.')
            return
        for key in all_keys:
            basename = os.path.basename(key)
            m = re.match(r'(.+)_voxel_\d+\.pt$', basename)
            prefix = m.group(1) if m else '_default_'
            source_entries.setdefault(prefix, []).append(key)

    if not source_entries:
        return

    vis_dir = os.path.join(os.path.dirname(eval_vxfile), 'visualisations')
    if save_outputs:
        os.makedirs(vis_dir, exist_ok=True)
    n_saved = 0
    grid_size_model = getattr(args, 'eval_grid_size', None)
    if grid_size_model is None:
        grid_size_model = args.grid_size[0] if isinstance(args.grid_size, (list, tuple)) else args.grid_size
    collect_grid_size = getattr(args, 'eval_collect_grid_size', 0.04)

    any_wood_threshold = getattr(args, 'eval_any_wood', None)

    def aggregate_and_save(classified_arr, out_path, gt_arr=None):
        """classified_arr columns: [x, y, z, prob, reflectance]. Argmax |p-0.5| per voxel."""
        pos_t = torch.as_tensor(classified_arr[:, :3], dtype=torch.float32, device='cpu')
        prob_t = torch.as_tensor(classified_arr[:, 3], dtype=torch.float32, device='cpu')
        refl_t = torch.as_tensor(classified_arr[:, 4], dtype=torch.float32, device='cpu')

        if gt_arr is not None and len(gt_arr) > 0:
            gt_pos_t = torch.as_tensor(gt_arr[:, :3], dtype=torch.float32, device='cpu')
            gt_lab_t = torch.as_tensor(gt_arr[:, 3], dtype=torch.float32, device='cpu')
            all_pos = torch.cat([pos_t, gt_pos_t], dim=0)
            raw_cluster = voxel_grid(all_pos, collect_grid_size)
            cluster, _ = consecutive_cluster(raw_cluster)
            n_clusters = int(cluster.max().item()) + 1
            n_pred = len(pos_t)
            pred_cluster = cluster[:n_pred]
            gt_cluster_t = cluster[n_pred:]
        else:
            pred_cluster = voxel_grid(pos_t, collect_grid_size)
            pred_cluster, _ = consecutive_cluster(pred_cluster)
            n_clusters = int(pred_cluster.max().item()) + 1
            gt_cluster_t = None

        pred_cnt = scatter_add(torch.ones_like(prob_t), pred_cluster, dim=0, dim_size=n_clusters)
        valid_pred = pred_cnt > 0

        voxel_pred = torch.zeros(n_clusters, dtype=torch.long)
        voxel_prob = torch.zeros(n_clusters, dtype=prob_t.dtype)
        voxel_refl = torch.zeros(n_clusters, dtype=refl_t.dtype)
        if any_wood_threshold is not None:
            max_prob, max_idx = scatter_max(prob_t, pred_cluster, dim=0, dim_size=n_clusters)
            safe_idx = max_idx.clamp(0, max(len(prob_t) - 1, 0))
            voxel_prob[valid_pred] = max_prob[valid_pred]
            voxel_pred[valid_pred] = (voxel_prob[valid_pred] >= float(any_wood_threshold)).long()
            voxel_refl[valid_pred] = refl_t[safe_idx[valid_pred]]
        else:
            conf = torch.abs(prob_t - 0.5)
            _, argmax_idx = scatter_max(conf, pred_cluster, dim=0, dim_size=n_clusters)
            safe_argmax_idx = argmax_idx.clamp(0, max(len(prob_t) - 1, 0))
            voxel_pred[valid_pred] = (prob_t[safe_argmax_idx[valid_pred]] >= 0.5).long()
            voxel_prob[valid_pred] = prob_t[safe_argmax_idx[valid_pred]]
            voxel_refl[valid_pred] = refl_t[safe_argmax_idx[valid_pred]]

        point_preds = voxel_pred[pred_cluster].numpy()
        point_probs = voxel_prob[pred_cluster].numpy()
        winning_refl = voxel_refl[pred_cluster].numpy()
        pos_np = pos_t.numpy()

        out_dict = {
            'x': pos_np[:, 0],
            'y': pos_np[:, 1],
            'z': pos_np[:, 2],
            'reflectance': winning_refl,
            'prediction': point_preds,
            'pwood': point_probs,
        }
        fields = ['reflectance', 'prediction', 'pwood']

        if gt_cluster_t is not None:
            gt_sum = scatter_add(gt_lab_t, gt_cluster_t, dim=0, dim_size=n_clusters)
            gt_cnt = scatter_add(torch.ones_like(gt_lab_t), gt_cluster_t, dim=0, dim_size=n_clusters)
            valid_gt = gt_cnt > 0
            voxel_gt = torch.zeros(n_clusters, dtype=torch.long)
            voxel_gt[valid_gt] = (gt_sum[valid_gt] / gt_cnt[valid_gt] >= 0.5).long()
            out_dict['label'] = voxel_gt[pred_cluster].numpy()
            fields.append('label')

        if save_outputs:
            out_df = pd.DataFrame(out_dict)
            save_file(out_path, out_df, additional_fields=fields, verbose=False)

    combined = []  # (classified_arr, prefix) for side-by-side export

    for prefix, entries_or_keys in sorted(source_entries.items()):
        class EvalArgs:
            pass
        eval_args = EvalArgs()
        eval_args.batch_size = getattr(args, 'batch_size', 4)
        eval_args.max_pts = getattr(args, 'max_pts', 16384)
        eval_args.grid_size = [float(grid_size_model)]
        eval_args.max_points_per_batch = getattr(args, 'max_points_per_batch', 50000)
        eval_args.verbose = False
        eval_args.wdir = args.wdir
        eval_args.model = args.model
        eval_args.vxfile = eval_vxfile
        if entries_or_keys and isinstance(entries_or_keys[0], dict):
            eval_args.in_memory = True
            eval_args.inference_voxels = entries_or_keys
        else:
            eval_args.in_memory = False
            eval_args.eval_file_pattern = f'{prefix}_voxel_*.pt' if prefix != '_default_' else 'voxel_*.pt'

        eval_loader, _ = create_inference_loader(eval_args, device)
        model.eval()
        output_list = []  # (N, 5) per batch element: x, y, z, prob, reflectance

        with torch.no_grad():
            for data in eval_loader:
                data = data.to(device, non_blocking=True)
                data.voxel_size = torch.tensor([float(grid_size_model)], dtype=torch.float32, device=data.pos.device)
                batch_ids = data.batch.cpu()
                pos = data.pos.cpu()
                refl = data.reflectance.cpu() if hasattr(data, 'reflectance') else None
                local_shift = data.local_shift.cpu()

                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    outputs = model(data)
                    outputs = torch.nan_to_num(outputs)
                    probs = torch.sigmoid(outputs).float()
                probs_np = probs.cpu().numpy().ravel()
                batch_counts = torch.bincount(batch_ids)
                splits = torch.cumsum(batch_counts, dim=0).numpy()
                starts = np.concatenate(([0], splits[:-1]))
                for b, (s, e) in enumerate(zip(starts, splits)):
                    shift = local_shift[3 * b : 3 * b + 3]
                    pos_global = (pos[s:e] + shift).numpy()
                    refl_col = refl[s:e].numpy() if refl is not None else np.zeros(int(e - s), dtype=np.float32)
                    output_list.append(np.column_stack((pos_global, probs_np[s:e], refl_col)))
                del data, outputs, probs

        if not output_list:
            continue

        classified = np.vstack(output_list)
        del output_list

        # GT labels from the voxel payloads (column 4): inference loader strips them.
        payloads = []
        if isinstance(entries_or_keys[0], dict):
            payloads = [entry.get('point_cloud') if isinstance(entry, dict) else entry for entry in entries_or_keys]
        else:
            for key in entries_or_keys:
                try:
                    raw = torch.load(key, map_location='cpu', weights_only=True)
                except Exception:
                    raw = torch.load(key, map_location='cpu', weights_only=False)
                payloads.append(raw.get('point_cloud') if isinstance(raw, dict) else raw)
        gt_points = []
        for pc in payloads:
            if isinstance(pc, torch.Tensor):
                pc = pc.numpy()
            elif not isinstance(pc, np.ndarray):
                pc = np.array(pc)
            if hasattr(pc, 'shape') and pc.ndim >= 2 and pc.shape[1] >= 5:
                gt_points.append(pc[:, [0, 1, 2, 4]])
        gt_all = np.vstack(gt_points) if gt_points else None

        # No-refl pass is temporarily disabled; pass refl twice so _eval_metrics keeps working.
        eval_m = _eval_metrics(
            classified[:, :4],
            classified[:, :4],
            gt_all,
            collect_grid_size,
            any_wood_threshold=any_wood_threshold,
        )
        if eval_m:
            tag = prefix if prefix != '_default_' else 'eval'
            print(
                f'[Eval {tag}] E{epoch} | '
                f'BAc={eval_m.get("balanced_accuracy", 0.0):.4f} '
                f'Pr={eval_m.get("precision", 0.0):.4f} '
                f'Re={eval_m.get("recall", 0.0):.4f} '
                f'Fbeta={eval_m.get("fbeta", 0.0):.4f} '
                f'MCC={eval_m.get("mcc", eval_m.get("mcc_with_refl", 0.0)):.4f} '
                f'H4={eval_m.get("h4_mcc", 0.0):.4f}'
            )
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.log({f'eval_{tag}/mcc': eval_m.get('mcc', eval_m.get('mcc_with_refl', 0.0)), 'epoch': epoch})
            except Exception:
                pass

        if save_outputs:
            out_path = (
                os.path.join(vis_dir, 'eval_latest.ply')
                if prefix == '_default_'
                else os.path.join(vis_dir, f'{prefix}_eval.ply')
            )
            aggregate_and_save(classified, out_path, gt_arr=gt_all)
        combined.append((classified, prefix, gt_all))
        n_saved += 1

    if save_outputs and len(combined) >= 2:
        padding = 1.0
        shifted_parts = []
        shifted_gts = []
        x_offset = 0.0
        for arr, _pfx, gt in combined:
            arr_shifted = arr.copy()
            mins = arr_shifted[:, :3].min(axis=0)
            maxs = arr_shifted[:, :3].max(axis=0)
            width = float(maxs[0] - mins[0])
            arr_shifted[:, 0] = arr_shifted[:, 0] - mins[0] + x_offset
            arr_shifted[:, 1] = arr_shifted[:, 1] - mins[1]
            arr_shifted[:, 2] = arr_shifted[:, 2] - mins[2]
            shifted_parts.append(arr_shifted)
            if gt is not None and len(gt) > 0:
                gt_shifted = gt.copy()
                gt_shifted[:, 0] = gt_shifted[:, 0] - mins[0] + x_offset
                gt_shifted[:, 1] = gt_shifted[:, 1] - mins[1]
                gt_shifted[:, 2] = gt_shifted[:, 2] - mins[2]
                shifted_gts.append(gt_shifted)
            x_offset += width + padding
        combined_arr = np.vstack(shifted_parts)
        combined_gt = np.vstack(shifted_gts) if shifted_gts else None
        combined_path = os.path.join(vis_dir, 'all_eval.ply')
        aggregate_and_save(combined_arr, combined_path, gt_arr=combined_gt)
        print(f'[Eval] Combined side-by-side: {combined_path} ({len(combined)} sources)')

    if save_outputs:
        print(f'[Eval] Saved {n_saved} visualisation(s) → {vis_dir}')


class EMAModel:
    """Exponential Moving Average for model weights."""

    def __init__(self, model, decay=0.9):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}



def SemanticTraining(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    torch.autograd.set_detect_anomaly(False)

    drop_path_rate = getattr(args, 'drop_path_rate', 0.0)
    learnable_kernels = getattr(args, 'learnable_kernels', False)
    num_kernel_points = getattr(args, 'num_kernel_points', (16, 16, 8))
    spatial_mix_lite = getattr(args, 'spatial_mix_lite', False)
    memory_efficient_conv = bool(getattr(args, 'memory_efficient_conv', False))
    sparse_max = bool(getattr(args, 'sparse_max', True))
    compressed_head = bool(getattr(args, 'compressed_head', False))
    compressed_head_dim = int(getattr(args, 'compressed_head_dim', 64))
    k_neighbors = int(getattr(args, 'k_neighbors', 16))
    lr = getattr(args, 'lr', 1e-3)
    if 'eu' in args.model.lower() or 'global' in args.model.lower():
        from src.model import NetFull as Net
        model = Net(num_classes=1, C=128, num_kernel_points=num_kernel_points, learnable_kernels=learnable_kernels, drop_path_rate=drop_path_rate, spatial_mix_lite=spatial_mix_lite, memory_efficient_conv=memory_efficient_conv, sparse_max=sparse_max, compressed_head=compressed_head, compressed_head_dim=compressed_head_dim, k_neighbors=k_neighbors).to(device)
        weight_decay = 1e-2
    else:
        from src.model import NetLight as Net
        model = Net(num_classes=1, C=16, num_kernel_points=num_kernel_points, learnable_kernels=True, drop_path_rate=drop_path_rate, spatial_mix_lite=spatial_mix_lite, memory_efficient_conv=memory_efficient_conv, sparse_max=sparse_max, compressed_head=compressed_head, compressed_head_dim=compressed_head_dim, k_neighbors=k_neighbors).to(device)
        weight_decay = 1e-2
    if spatial_mix_lite:
        print("SpatialMix-lite enabled (SA2 residual blocks)")
    if memory_efficient_conv:
        print("Memory-efficient AnisotropicConv enabled (lower peak memory, slower)")
    if sparse_max:
        print("Sparsemax kernel routing enabled")
    print("Seg head: " + (f"compressed C3->{compressed_head_dim}->1" if compressed_head else "full-width FP residual MLP"))

    stage_kernel_points = getattr(model, 'stage_kernel_points', None)
    if stage_kernel_points is not None:
        kernel_label = "/".join(str(k) for k in stage_kernel_points)
        print(f"Kernel points per stage: SA1/SA2/SA3 = {kernel_label}")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*60}')
    print(f'MODEL SUMMARY - {"NetFull (EU)" if "eu" in args.model.lower() else "NetLight (Biome)"}')
    print(f'{"="*60}')

    component_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        component_params[name] = params

    for name, params in sorted(component_params.items(), key=lambda x: -x[1]):
        pct = 100 * params / total_params
        print(f'{name:<20} {params:>12,} params ({pct:>5.1f}%)')

    print(f'{"-"*60}')
    print(f'{"TOTAL":<20} {total_params:>12,} params')
    print(f'{"-"*60}')
    print(f'{"="*60}\n')

    train_loader, train_dataset = create_train_loader(args, device)

    # Write 10 PointCutMix sample PLYs for visualization when --pointcutmix is on
    if getattr(args, 'pointcutmix', False):
        n_mix = getattr(train_dataset, '_num_mix_slots', 0)
        if n_mix > 0:
            import pandas as pd
            debug_dir = os.path.join(os.getcwd(), 'pointcutmix')
            os.makedirs(debug_dir, exist_ok=True)
            n_real = len(train_dataset.keys)
            for i in range(min(10, n_mix)):
                data = train_dataset[n_real + i]
                df = pd.DataFrame({
                    'x': data.pos[:, 0].cpu().numpy(),
                    'y': data.pos[:, 1].cpu().numpy(),
                    'z': data.pos[:, 2].cpu().numpy(),
                    'label': data.y.cpu().numpy().astype(np.float64),
                })
                if data.reflectance is not None:
                    df['reflectance'] = data.reflectance.cpu().numpy().astype(np.float64)
                extra = ['label', 'reflectance'] if data.reflectance is not None else ['label']
                save_file(os.path.join(debug_dir, f'pointcutmix_{i:02d}.ply'), df, additional_fields=extra, verbose=False)
            print(f"PointCutMix: wrote 10 sample mixes to {debug_dir}/")

    test_dataset = None
    focus_dataset = None
    if args.test:
        test_loader, test_dataset = create_test_loader(args, device)
        eval_vxfile = getattr(args, 'eval_vxfile', None)
        if eval_vxfile and os.path.isdir(eval_vxfile):
            focus_dataset = TrainingDataset(
                voxels=eval_vxfile,
                augmentation=False,
                mode='test',
                max_pts=args.max_pts,
                device=device,
                denoise=getattr(args, 'denoise', False),
                denoise_k=getattr(args, 'denoise_k', 16),
                denoise_std=getattr(args, 'denoise_std', 1.0),
                pointcutmix=False,
            )
        else:
            # Build focus dataset from same test voxels but without FOCUS_ONLY exclusions,
            # so fin04-hard shards are available for focus eval but not in val/adaptive sampling.
            focus_dataset = TrainingDataset(
                voxels=args.tefile,
                augmentation=False,
                mode='test',
                max_pts=args.max_pts,
                device=device,
                denoise=getattr(args, 'denoise', False),
                denoise_k=getattr(args, 'denoise_k', 16),
                denoise_std=getattr(args, 'denoise_std', 1.0),
                pointcutmix=False,
            )

    # Cyclical Focal Loss: gamma cycles 0 → gamma_max → 0 over training
    # Early: gamma=0 (pure BCE, strong gradients for learning)
    # Mid: gamma=gamma_max (focus on hard examples)
    # End: gamma→0 (stabilize predictions)
    # gamma_max=0 → plain BCE; label_smoothing=0 → no smoothing
    gamma_max = getattr(args, 'gamma_max', 2.0)
    gamma_peak_pct = float(getattr(args, 'gamma_peak_pct', 0.33))
    gamma_peak_pct = min(0.95, max(0.05, gamma_peak_pct))
    label_smoothing = getattr(args, 'label_smoothing', 0.05)
    focal_alpha = getattr(args, 'focal_alpha', None)
    boundary_max = getattr(args, 'boundary_weight', 0.0)
    boundary_ramp_start = getattr(args, 'boundary_ramp_start', 0.1)
    difficulty_on = bool(getattr(args, 'difficulty_sampling', False)) and float(getattr(args, 'difficulty_alpha', 0.0)) > 0.0
    adaptive_sampling_on = bool(getattr(args, 'adaptive_group_sampling', False))
    adaptive_warmup = getattr(args, 'adaptive_sampling_warmup', None)
    if adaptive_warmup is None:
        adaptive_warmup = max(1, int(round(0.10 * args.num_epochs)))
    adaptive_ramp = getattr(args, 'adaptive_sampling_ramp', None)
    if adaptive_ramp is None:
        adaptive_ramp = max(1, int(round(0.10 * args.num_epochs)))
    adaptive_sampler = ValidationGroupSamplerTracker(
        enabled=adaptive_sampling_on,
        warmup_epochs=adaptive_warmup,
        ramp_epochs=adaptive_ramp,
        alpha=getattr(args, 'adaptive_sampling_alpha', 5.0),
        min_multiplier=getattr(args, 'adaptive_sampling_min', 0.3),
        max_multiplier=getattr(args, 'adaptive_sampling_max', 3.0),
        metric=getattr(args, 'adaptive_sampling_metric', 'balanced_acc'),
    )
    if adaptive_sampling_on:
        sampler = getattr(train_loader, 'batch_sampler', None)
        if not hasattr(sampler, 'set_weights'):
            adaptive_sampler.enabled = False
            adaptive_sampling_on = False
            print("Adaptive group sampling requested but current sampler cannot update weights; disabled.")
        elif test_dataset is None or len(getattr(test_dataset, 'group_list', [])) <= 1:
            adaptive_sampler.enabled = False
            adaptive_sampling_on = False
            groups_seen = getattr(test_dataset, 'group_list', [])
            print(
                "Adaptive group sampling requested but validation has no usable groups "
                f"({groups_seen}); disabled. Rerun --preprocess so shard names keep source prefixes."
            )
        else:
            print(
                "Adaptive group sampling ON | "
                f"metric={adaptive_sampler.metric} | warmup={adaptive_sampler.warmup_epochs} | "
                f"ramp={adaptive_sampler.ramp_epochs} | cap={adaptive_sampler.min_multiplier:.2f}-{adaptive_sampler.max_multiplier:.2f}x"
            )

    # Per-voxel difficulty tracking: boundary-weighted loss EMA per training voxel.
    # Warm-starts from edge_fraction prior; online loss updates identify which specific
    # voxels the model is currently failing on (not just which biome).
    per_voxel_on = bool(getattr(args, 'per_voxel_difficulty', False))
    per_voxel_alpha = float(getattr(args, 'per_voxel_alpha', 2.0))
    per_voxel_warmup = int(getattr(args, 'per_voxel_warmup_epochs', 3))
    voxel_ema_alpha = float(getattr(args, 'voxel_ema_alpha', 0.9))
    if per_voxel_on:
        sampler = getattr(train_loader, 'batch_sampler', None)
        if sampler is None or not hasattr(sampler, 'set_weights'):
            per_voxel_on = False
            print("Per-voxel difficulty requested but sampler has no set_weights; disabled.")
        else:
            prior = np.asarray(
                getattr(train_dataset, 'edge_fractions', np.zeros(len(train_dataset.keys))),
                dtype=np.float64,
            )
            voxel_tracker = VoxelDifficultyTracker(
                n_real=len(train_dataset.keys), prior=prior, alpha_ema=voxel_ema_alpha,
            )
            print(
                f"Per-voxel difficulty ON | alpha={per_voxel_alpha} | "
                f"warmup={per_voxel_warmup} | ema={voxel_ema_alpha} | "
                f"n_voxels={len(train_dataset.keys)}"
            )
    if not per_voxel_on:
        voxel_tracker = None

    criterion = FocalLoss(
        gamma_max=gamma_max,
        alpha=focal_alpha,
        label_smoothing=label_smoothing,
        cyclical=True,
        pct_peak=gamma_peak_pct,
        boundary_max=boundary_max,
        boundary_ramp_start=boundary_ramp_start,
    )
    boundary_str = f", boundary={boundary_max}x" if boundary_max > 0 else ""
    refl_fp_weight = getattr(args, 'refl_fp_penalty', 0.15)
    refl_fp_str = f" | FP_penalty={refl_fp_weight}" if refl_fp_weight > 0 else ""
    print(f"Loss: Focal(γ={gamma_max},peak={gamma_peak_pct:.2f},label_smooth={label_smoothing}){boundary_str}{refl_fp_str}")
    refl_fp_ramp = getattr(args, 'refl_fp_ramp', True)
    refl_fp_flat = getattr(args, 'refl_fp_flat', False)
    refl_fp_criterion = ReflectanceFPPenalty(margin=1.0, strength=2.0, weight=refl_fp_weight, ramp=refl_fp_ramp, flat=refl_fp_flat) if refl_fp_weight > 0 else None

    contrastive_weight = float(getattr(args, 'contrastive_weight', 0.1))
    if contrastive_weight > 0:
        contrastive_criterion = SupConLoss(
            n_anchors=512,
            start_temp=0.2,
            end_temp=0.07,
            weight=contrastive_weight,
            ramp_frac=0.15,
        )
        print(f"Contrastive loss: SupCon weight={contrastive_weight} | temp 0.20→0.07 | ramp first 15% of training")
    else:
        contrastive_criterion = None

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bn" in name or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr) 

    # Epoch-level OneCycleLR: total_steps is epochs, and step() is called once
    # after each epoch rather than inside the batch loop.
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=args.num_epochs,
        pct_start=0.10,
        anneal_strategy='cos',
        div_factor=25
    )
            
    manager = ModelManager(model, device)
    history_logger = HistoryLogger(args)
    wandb_logger = WandbLogger(args)

    if os.path.isfile(os.path.join(args.wdir,'model',args.model)):
        try:
            manager.load_model(os.path.join(args.wdir,'model',args.model))
            print("Model loaded")
        except (KeyError, RuntimeError, Exception) as e:
            print(f"Failed to load model: {e}. Creating new...")
            torch.save(model.state_dict(), os.path.join(args.wdir,'model',args.model))
    else:
        torch.save(model.state_dict(), os.path.join(args.wdir,'model',args.model))
        print("Model created")

    best_h4_mcc = 0.0
    early_stop_best = -float('inf')
    early_stop_bad_epochs = 0

    amp_enabled, amp_dtype, amp_name = _resolve_amp_config(device, getattr(args, 'amp_dtype', 'auto'))
    # Grad scaling is only needed for fp16; bf16 has fp32-like exponent range.
    scaler = torch.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

    # EMA Loss tracking
    ema_loss = None
    ema_alpha = 0.9

    # EMA Model tracking: updated every step; decay per step (0.999 ~1k steps, 0.99 ~100 steps)
    use_ema = getattr(args, 'ema', False)
    ema_decay = getattr(args, 'ema_decay', 0.999)
    if use_ema:
        ema_model = EMAModel(model, decay=ema_decay)
        ema_model.register()
    else:
        ema_model = None

    accumulation_steps = getattr(args, 'accumulation_steps', 4)
    ema_str = f" | EMA(decay={ema_decay})" if use_ema else ""
    print(f"AMP: {amp_name} | Grad accum: {accumulation_steps}{ema_str}")
    optimizer.zero_grad(set_to_none=True)
    accumulated_batches = 0  # Track actual accumulated batches
    global_train_step = 0
    wandb_train_log_interval = max(1, int(getattr(args, 'wandb_train_log_interval', 25)))

    # Track best metrics across all epochs
    best_metrics_by_epoch = {}

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        print(f"\n{'='*100}\nEPOCH {epoch}\n{'='*100}")

        criterion.set_epoch(epoch, args.num_epochs)
        if refl_fp_criterion is not None:
            refl_fp_criterion.set_epoch(epoch, args.num_epochs)
        if contrastive_criterion is not None:
            contrastive_criterion.set_epoch(epoch, args.num_epochs)

        # Sampling ramps: edge difficulty can scale in, and validation-group weights can
        # bias future epochs after a warmup without changing the loss.
        _alpha_cur = 0.0
        _sampler_weights = None
        _voxel_alpha_cur = per_voxel_alpha if (per_voxel_on and epoch > per_voxel_warmup) else 0.0
        if difficulty_on:
            _alpha_full = float(getattr(args, 'difficulty_alpha', 2.0))
            _ramp_pct = float(getattr(args, 'difficulty_ramp_pct', 0.5))
            _t = min(1.0, (epoch - 1) / max(1, _ramp_pct * args.num_epochs))
            _alpha_cur = _alpha_full * _t
        # Once voxel-EMA is active it already encodes edge_frac via its warm-start prior —
        # drop edge_alpha to avoid applying the same signal twice.
        _edge_alpha_cur = 0.0 if _voxel_alpha_cur > 0 else _alpha_cur
        _coverage_warmup = per_voxel_on and voxel_tracker is not None and epoch <= per_voxel_warmup
        _unseen_boost = float(getattr(args, 'unseen_voxel_boost', 8.0)) if _coverage_warmup else 0.0
        _allow_replacement = not (
            _coverage_warmup and not bool(getattr(args, 'coverage_warmup_replacement', False))
        )
        if difficulty_on or adaptive_sampling_on or per_voxel_on:
            _sampler_weights = _apply_train_sampler_weights(
                train_loader,
                train_dataset,
                edge_alpha=_edge_alpha_cur,
                group_multipliers=adaptive_sampler.multipliers if adaptive_sampling_on else None,
                voxel_difficulty=voxel_tracker if per_voxel_on else None,
                voxel_alpha=_voxel_alpha_cur,
                coverage_tracker=voxel_tracker if _coverage_warmup else None,
                unseen_boost=_unseen_boost,
                allow_replacement=_allow_replacement,
            )

        # Push per-voxel difficulty into dataset so augmentations() can suppress
        # compress/copy for consistently-hard voxels. Only after EMA warmup — during
        # warmup the tracker is just the edge_fraction prior, not meaningful loss signal.
        if per_voxel_on and voxel_tracker is not None and epoch > per_voxel_warmup:
            _diff_raw = voxel_tracker.normalised_weights()  # [0, 2] centred at 1.0
            train_dataset.difficulty_scores = np.clip(_diff_raw - 1.0, 0.0, 1.0)  # [0,1]; 0=avg/easy

        refl_str = f" | ReflFP: {refl_fp_criterion.ramp_factor:.3f}" if refl_fp_criterion is not None else ""
        boundary_str = f" | Boundary: {criterion.boundary_weight:.2f}x" if criterion.boundary_max > 0 else ""
        diff_str = f" | DiffA: {_alpha_cur:.2f}" if difficulty_on and _sampler_weights is not None else ""
        adapt_str = f" | AdaptS: {adaptive_sampler.status_str()}" if adaptive_sampling_on else ""
        voxel_str = f" | VoxDiff: {voxel_tracker.status_str()}" if per_voxel_on and voxel_tracker is not None else ""
        coverage_str = (
            f" | Cover: unseen_boost={_unseen_boost:.1f}, no_repl={not _allow_replacement}"
            if _coverage_warmup else ""
        )
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f} | Gamma: {criterion.current_gamma:.3f}{boundary_str}{refl_str}{diff_str}{adapt_str}{voxel_str}{coverage_str}")
        train_tracker = MetricsTracker(full_metrics=False)  # Fast metrics only for training
        clamp_hits = 0
        clamp_max_unclamped = 0.0
        ema_focal = None
        ema_con = None
        ema_refl_fp = None

        # Max points per batch to avoid OOM (adjust based on your GPU)
        max_points_per_batch = getattr(args, 'max_points_per_batch', 50000)
        verbose = getattr(args, 'verbose', False)

        # Limit steps per epoch (0 = use all data)
        # If difficulty sampling is active and no explicit value was given, default to 50% of
        # the loader so the weighted sampler has genuine selection pressure each epoch.
        epoch_steps = getattr(args, 'epoch_steps', 0)
        if epoch_steps == 0 and difficulty_on:
            epoch_steps = max(1, len(train_loader) // 2)
        total_steps = min(len(train_loader), epoch_steps) if epoch_steps > 0 else len(train_loader)

        with tqdm(total=total_steps, colour='white', ascii="░▒", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as tepoch:
            for i, data in enumerate(train_loader):
                # Break if we've done enough steps for this epoch
                if epoch_steps > 0 and i >= epoch_steps:
                    break
                # Safety net: downsample batch if over budget (collate should already cap when using point-budget sampler)
                if data.pos.shape[0] > max_points_per_batch:
                    data = downsample_batch_to_point_budget(data, max_points_per_batch, data.pos.device)

                data = data.to(device)

                # Batch-level density aug: one spacing per batch so whole batch has same resolution
                if getattr(args, 'density_aug', False):
                    from src.augmentation import random_density_downsample_batch
                    spacing = getattr(args, 'density_aug_spacing', [0.01, 0.04])
                    random_density_downsample_batch(
                        data,
                        spacing_min=spacing[0],
                        spacing_max=spacing[1],
                        prob=getattr(args, 'density_aug_prob', 0.5),
                        difficulty=getattr(data, 'difficulty', None),
                        hard_skip_threshold=getattr(args, 'density_aug_hard_threshold', 0.75),
                        difficulty_power=getattr(args, 'density_aug_difficulty_power', 2.0),
                    )

                # PointCutMix runs in the dataset (loader) like normal augmentation

                # Skip batch early if any input contains NaN/Inf to prevent cascade
                try:
                    inputs_ok = True
                    for attr in ("x", "pos", "edge_scores", "y", "reflectance", "sf"):
                        if hasattr(data, attr):
                            tensor = getattr(data, attr)
                            if tensor is not None and not torch.isfinite(tensor).all():
                                inputs_ok = False
                                break
                    if not inputs_ok:
                        print(f"[Warning] Non-finite inputs at step {i}, skipping batch")
                        # If we've accumulated enough, force an optimizer update cycle without step
                        if accumulated_batches >= accumulation_steps:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.zero_grad(set_to_none=True)
                            accumulated_batches = 0
                        continue
                except Exception:
                    pass
                
                _set_batch_voxel_size(data, args)

                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    model_output = model(data)

                    outputs = model_output
                    per_sample_tracking_loss = None

                    if not torch.isfinite(outputs).all():
                        print(f"[Warning] Non-finite model outputs at step {i}, skipping batch")
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_batches = 0
                        continue

                    focal_loss_val = criterion(outputs, data.y.float(), edge_scores=getattr(data, 'edge_scores', None))
                    loss = focal_loss_val
                    per_sample_tracking_loss = None

                    if per_voxel_on and voxel_tracker is not None:
                        if per_sample_tracking_loss is None:
                            with torch.no_grad():
                                per_sample_tracking_loss = _per_sample_boundary_weighted_bce(
                                    outputs.detach(),
                                    data.y.float(),
                                    data.batch,
                                    edge_scores=getattr(data, 'edge_scores', None),
                                )
                        voxel_ids = getattr(data, 'voxel_idx', None)
                        if voxel_ids is not None:
                            voxel_ids_np = voxel_ids.view(-1).detach().cpu().numpy()
                            voxel_tracker.update(
                                voxel_ids_np[:per_sample_tracking_loss.numel()],
                                per_sample_tracking_loss.detach().cpu().numpy(),
                            )

                    refl_fp_val = 0.0
                    if refl_fp_criterion is not None and getattr(data, 'reflectance', None) is not None:
                        _rfp = refl_fp_criterion(outputs, data.y.float(), data.reflectance)
                        loss = loss + _rfp
                        refl_fp_val = float(_rfp.detach().item())

                    con_val = 0.0
                    if contrastive_criterion is not None and hasattr(model, 'last_proj'):
                        con_loss = contrastive_criterion(
                            model.last_proj,
                            (data.y.float() >= 0.5),
                            edge_scores=getattr(data, 'edge_scores', None),
                        )
                        if torch.isfinite(con_loss):
                            loss = loss + con_loss
                            con_val = float(con_loss.detach().item())

                    # Track per-component EMAs for balance diagnostics
                    _fv = float(focal_loss_val.detach().item())
                    ema_focal  = _fv if ema_focal  is None else ema_alpha * ema_focal  + (1 - ema_alpha) * _fv
                    ema_con    = con_val if ema_con    is None else ema_alpha * ema_con    + (1 - ema_alpha) * con_val
                    ema_refl_fp= refl_fp_val if ema_refl_fp is None else ema_alpha * ema_refl_fp + (1 - ema_alpha) * refl_fp_val

                    # Clamp for stability. Track unclamped value so we can warn if
                    # stacked losses (focal + boundary + refl-FP) hit the ceiling.
                    unclamped_loss_value = float(loss.detach().item())
                    loss = torch.clamp(loss, min=0.0, max=10.0)
                    if unclamped_loss_value > 10.0:
                        clamp_hits += 1
                        if unclamped_loss_value > clamp_max_unclamped:
                            clamp_max_unclamped = unclamped_loss_value
                        print(f"[Clamp alarm] epoch {epoch} step {i}: unclamped={unclamped_loss_value:.2f} > 10.0 (gradients silently capped on edges)")
                    if not torch.isfinite(loss).all():
                        print(f"[Warning] Non-finite loss at step {i}, skipping batch")
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_batches = 0
                        continue

                    loss = loss / accumulation_steps

                    scaler.scale(loss).backward()
                    accumulated_batches += 1

                    # Store values before clearing
                    loss_value = loss.item()
                    train_tracker.update(loss_value * accumulation_steps, outputs, data.y, 
                                        edge_scores=data.edge_scores, pos=None)  # No pos needed for training metrics

                    del data, outputs, loss

                # Check for gradient update
                if accumulated_batches >= accumulation_steps:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if (not scaler.is_enabled()) and (not torch.isfinite(grad_norm)):
                        print(f"[Warning] Non-finite gradient norm at step {i}, skipping optimizer step")
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_batches = 0
                        continue

                    # scaler.step skips optimizer if grads are non-finite, scaler.update adjusts scale
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()

                    # Only update EMA if optimizer actually stepped (scale unchanged = step happened)
                    if ema_model is not None and scaler.get_scale() >= old_scale:
                        ema_model.update()

                    optimizer.zero_grad(set_to_none=True)
                    accumulated_batches = 0

                # Update EMA loss
                current_loss = loss_value * accumulation_steps
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * current_loss

                # Metrics already updated above before memory clearing

                # Update progress bar with current batch metrics
                current_metrics = train_tracker.get_averages()
                pbar_dict = {
                    'Lo': round(current_metrics['loss'], 4),
                    'LoS': round(ema_loss, 4),
                    'BAc': round(current_metrics['accuracy'], 3),
                    'Re': round(current_metrics['recall'], 3),
                    'Fb': round(current_metrics['fbeta'], 3),
                }
                if contrastive_criterion is not None and ema_focal is not None and ema_focal > 1e-6:
                    pbar_dict['ConR'] = round(ema_con / ema_focal, 2)
                    pbar_dict['ConA'] = round(ema_con, 3)
                tepoch.set_postfix(pbar_dict)
                tepoch.update(1)

                global_train_step += 1
                if (
                    wandb_logger.wandb is not None
                    and (global_train_step % wandb_train_log_interval == 0 or (i + 1) == total_steps)
                ):
                    wandb_logger.log_train_step(
                        global_train_step,
                        optimizer.param_groups[0]["lr"],
                        current_metrics,
                        batch_loss=current_loss,
                        ema_loss=ema_loss,
                    )
            tepoch.close()
            
        # Handle any remaining gradients at epoch end
        if accumulated_batches > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if (not scaler.is_enabled()) and (not torch.isfinite(grad_norm)):
                print("[Warning] Non-finite gradient norm at epoch end, skipping final optimizer step")
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
            else:
                scaler.step(optimizer)
                scaler.update()

                # Update EMA weights for final step
                if ema_model is not None:
                    ema_model.update()

                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0

        # Keep LR cycling per epoch, not within the epoch.
        lr_scheduler.step()

        if clamp_hits > 0:
            clamp_rate = 100.0 * clamp_hits / max(1, total_steps)
            severity = "HIGH" if clamp_rate > 1.0 else "low"
            print(f"[Clamp summary] epoch {epoch}: {clamp_hits}/{total_steps} steps clamped "
                  f"({clamp_rate:.2f}% — {severity}), worst unclamped={clamp_max_unclamped:.2f}")
            if clamp_rate > 1.0:
                print(f"[Clamp warning] >1% of steps hit the 10.0 ceiling — consider lowering boundary_max/refl_fp_penalty or raising the clamp")

        train_metrics = train_tracker.get_averages()

        # Loss component breakdown
        if ema_focal is not None:
            _focal_s = f"focal={ema_focal:.3f}"
            _rfp_s   = f" refl_fp={ema_refl_fp:.3f}" if refl_fp_criterion is not None else ""
            _con_s   = ""
            if contrastive_criterion is not None and ema_focal > 1e-6:
                con_ratio = ema_con / ema_focal
                flag = " ⚠ con>2x focal" if con_ratio > 2.0 else (" ✓" if con_ratio < 1.0 else "")
                _con_s = f" con={ema_con:.3f} (×{con_ratio:.2f} focal){flag}"
            print(f"Loss breakdown: {_focal_s}{_rfp_s}{_con_s}")

        # Compact flashlight / reflectance diagnostics
        def _short_layer_name(layer_name: str) -> str:
            if layer_name.startswith("sa1_module"):
                return "sa1"
            if layer_name.startswith("sa2_module"):
                return "sa2"
            if layer_name.startswith("sa3_module"):
                return "sa3"
            return layer_name.replace("_module", "")

        def _diag_float(value, default=None):
            if value is None:
                return default
            try:
                if isinstance(value, torch.Tensor):
                    return float(value.detach().cpu().item())
                return float(value)
            except Exception:
                return default

        kernel_metrics = []
        refl_rows = []
        refl_gain_list = []
        diag_warnings = []
        contrast_gate_list = []
        refl_stage_metrics = {}

        for name, module in model.named_modules():
            if hasattr(module, 'kernel_entropy'):
                kernel_metrics.append(module.kernel_entropy.item())
            if not isinstance(module, AnisotropicConv):
                continue

            layer_name = _short_layer_name(name.split('.conv')[0])

            if hasattr(module, 'diagnostics'):
                diag = module.diagnostics
                contrast_gate = diag.get('contrast_gate_mean', None)
                refl_gain = diag.get('cobright_lift_mean', diag.get('refl_edge_gain_mean', diag.get('refl_gain_mean', None)))
                contrast_gate_f = _diag_float(contrast_gate, None)
                refl_gain_f = _diag_float(refl_gain, None)

                if refl_gain_f is not None and np.isfinite(refl_gain_f):
                    refl_gain_list.append(refl_gain_f)
                if contrast_gate_f is not None and np.isfinite(contrast_gate_f):
                    contrast_gate_list.append(contrast_gate_f)

                gate_print = contrast_gate_f if contrast_gate_f is not None and np.isfinite(contrast_gate_f) else 0.0
                gain_print = refl_gain_f if refl_gain_f is not None and np.isfinite(refl_gain_f) else 1.0
                refl_rows.append(f"{layer_name}[gate={gate_print:.2f},gain={gain_print:.2f}x]")
                refl_stage_metrics[f"refl_gate_{layer_name}"] = gate_print
                refl_stage_metrics[f"refl_gain_{layer_name}"] = gain_print

                if 'warning' in diag:
                    diag_warnings.append(f"{layer_name}: {diag['warning']}")
                if bool(_diag_float(diag.get('has_nan', False), 0.0)):
                    diag_warnings.append(f"{layer_name}: NaN detected")

        model_metrics = {
            "kernel_entropy":   sum(kernel_metrics) / len(kernel_metrics) if kernel_metrics else 0.0,
            "refl_gate_mean": np.mean(contrast_gate_list) if contrast_gate_list else 0.0,
            "refl_gain_mean": np.mean(refl_gain_list) if refl_gain_list else 1.0,
        }
        model_metrics.update(refl_stage_metrics)

        if kernel_metrics:
            avg_entropy = sum(kernel_metrics) / len(kernel_metrics)
        else:
            avg_entropy = 0.0

        if not args.test:
            print(f"Kernels: {avg_entropy:.2f}" + (" | Refl: " + " | ".join(refl_rows) if refl_rows else ""))
            if diag_warnings:
                print(f"Warnings: {'; '.join(diag_warnings)}")

        if args.test:
            model.eval()
            refl_usage = {
                "all": float("nan"),
                "edge": float("nan"),
                "pure": float("nan"),
                "edge_wood": float("nan"),
                "edge_ratio": float("nan"),
            }
            if getattr(args, 'refl_diagnostics', False):
                refl_usage = compute_refl_dominance_diagnostic(model, test_loader, device, args)
            mean_dlogit_drefl = refl_usage.get("all", float("nan"))
            if not np.isnan(mean_dlogit_drefl):
                # Single-line diagnostic: refl strength, geometry structure per stage, refl gain per SA
                edge_usage = refl_usage.get("edge", float("nan"))
                pure_usage = refl_usage.get("pure", float("nan"))
                edge_wood_usage = refl_usage.get("edge_wood", float("nan"))
                edge_ratio = refl_usage.get("edge_ratio", float("nan"))
                usage_str = (
                    f"all={mean_dlogit_drefl:.2f},edge={edge_usage:.2f},"
                    f"edgeW={edge_wood_usage:.2f},pure={pure_usage:.2f},e/p={edge_ratio:.2f}"
                )
                print(f"\033[96mRefl Strength: {usage_str} | Refl: {' | '.join(refl_rows) if refl_rows else 'N/A'}\033[0m")
            else:
                print(f"Kernels: {avg_entropy:.2f}" + (" | Refl: " + " | ".join(refl_rows) if refl_rows else ""))
            if diag_warnings:
                print(f"Warnings: {'; '.join(diag_warnings)}")

            # Apply EMA weights for testing (if enabled)
            if ema_model is not None:
                ema_model.apply_shadow()

            test_metrics_with_refl = run_validation_pass(model, test_loader, device, "With Reflectance", "val_with_reflectance", args, track_groups=True)
            test_metrics_no_refl = run_validation_pass(model, test_loader, device, "No Reflectance", "val_no_reflectance", args)

            harmonic_metrics = calculate_harmonic_metrics(test_metrics_with_refl, test_metrics_no_refl)
            print_validation_summary(epoch, harmonic_metrics)
            edge_gain = harmonic_metrics.get('mcc_edge_with_refl', 0.0) - harmonic_metrics.get('mcc_edge_no_refl', 0.0)
            pure_gain = harmonic_metrics.get('mcc_pure_with_refl', 0.0) - harmonic_metrics.get('mcc_pure_no_refl', 0.0)
            usage_ratio = refl_usage.get('edge_ratio', float('nan'))
            usage_str = f" | usage edge/pure={usage_ratio:.2f}" if np.isfinite(usage_ratio) else ""
            con_str = f" | SupCon τ={contrastive_criterion.temperature:.3f} ramp={contrastive_criterion.ramp_factor:.2f}" if contrastive_criterion is not None else ""
            print(f"Reflectance ΔMCC: edge={edge_gain:+.4f} | pure={pure_gain:+.4f}{usage_str}{con_str}")
            print_focus_site_eval(model, focus_dataset, device, args, epoch, prefixes=('deu08', 'fin04', 'fin04-hard', 'esp190'))
            if adaptive_sampling_on:
                adaptive_sampler.update(epoch, test_metrics_with_refl.get('group_metrics', {}))
                print(f"Adaptive sampling: {adaptive_sampler.breakdown_str()}")
                try:
                    import wandb as _wandb
                    if _wandb.run is not None and adaptive_sampler.multipliers:
                        _wandb.log(
                            {f'adaptive_sampling/{g}': w for g, w in adaptive_sampler.multipliers.items()} | {'epoch': epoch}
                        )
                except Exception:
                    pass

            # Track best metrics across all epochs
            best_metrics_by_epoch[epoch] = {
                'h4_mcc': harmonic_metrics.get('h4_mcc', 0.0),
                'mcc_with_refl_pure': test_metrics_with_refl.get('mcc_pure', 0.0),
                'mcc_no_refl_pure': test_metrics_no_refl.get('mcc_pure', 0.0),
                'mcc_pure_h': harmonic_metrics.get('mcc_pure_h', 0.0),
                'mcc_with_refl_edge': test_metrics_with_refl.get('mcc_edge', 0.0),
                'mcc_no_refl_edge': test_metrics_no_refl.get('mcc_edge', 0.0),
                'mcc_edge_h': harmonic_metrics.get('mcc_edge_h', 0.0),
            }

            # Update test metrics with harmonic data for logging
            test_metrics = update_test_metrics_with_harmonic(test_metrics_with_refl.copy(), harmonic_metrics)
            test_metrics["refl_dominance"] = mean_dlogit_drefl if (np.isfinite(mean_dlogit_drefl) and not np.isnan(mean_dlogit_drefl)) else 0.0
            for key, value in refl_usage.items():
                test_metrics[f"refl_dominance_{key}"] = value if np.isfinite(value) else 0.0
            
            # Restore original weights for training
            if ema_model is not None:
                ema_model.restore()
        else:
            test_metrics = None

        history_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, test_metrics)
        wandb_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, test_metrics, model_metrics=model_metrics)

        if getattr(args, 'eval', False) and epoch % 10 == 0:
            if ema_model is not None:
                ema_model.apply_shadow()
            if getattr(args, 'eval', False):
                run_eval_visualization(model, args, device, epoch)
            if ema_model is not None:
                ema_model.restore()

        if epoch in args.checkpoints:
            manager.save_checkpoints(args, epoch)
        
        # Early stopping: monitor H4-MCC after 60% of training
        if getattr(args, 'early_stop', False) and getattr(args, 'test', False):
            min_delta = 0.001
            patience = 10
            start_epoch = max(1, int(args.num_epochs * 0.60))
            if epoch >= start_epoch:
                current_harm = harmonic_metrics.get('h4_mcc', None)
                if current_harm is not None:
                    if current_harm > early_stop_best + min_delta:
                        early_stop_best = current_harm
                        early_stop_bad_epochs = 0
                    else:
                        early_stop_bad_epochs += 1
                    if early_stop_bad_epochs >= patience:
                        print(f"\nEarly stopping at epoch {epoch}: no H4-MCC improvement ≥ {min_delta} for {patience} epochs (best={early_stop_best:.3f})")
                        best_epoch, best_acc = history_logger.get_best_epoch()
                        print(f"Best accuracy was {best_acc:.4f} at epoch {best_epoch}")
                        break

        # Save best model on H4-MCC: the single metric that demands good performance
        # across all four conditions (with/no refl × pure/edge).
        if args.test and epoch > int(args.num_epochs*0.25):
            if ema_model is not None:
                ema_model.apply_shadow()
            best_h4_mcc = manager.save_best_model(harmonic_metrics['h4_mcc'], best_h4_mcc, os.path.join(args.wdir,'model','h4mcc-' + os.path.basename(args.model)))
            if ema_model is not None:
                ema_model.restore()

        if epoch == args.num_epochs:
            print("Saving final GLOBAL model")
            if ema_model is not None:
                ema_model.apply_shadow()
            torch.save(manager.checkpoint_payload(), os.path.join(args.wdir,'model',args.model))
            if ema_model is not None:
                ema_model.restore()

    # Print final training summary with best metrics
    if best_metrics_by_epoch:
        print("\n" + "="*100)
        print("FINAL TRAINING SUMMARY")
        print("="*100)

        best_h4_val = max(m['h4_mcc'] for m in best_metrics_by_epoch.values())
        best_h4_epoch = [e for e, m in best_metrics_by_epoch.items() if m['h4_mcc'] == best_h4_val][0]

        best_pure_val = max(m['mcc_pure_h'] for m in best_metrics_by_epoch.values())
        best_pure_epoch = [e for e, m in best_metrics_by_epoch.items() if m['mcc_pure_h'] == best_pure_val][0]

        best_edge_val = max(m['mcc_edge_h'] for m in best_metrics_by_epoch.values())
        best_edge_epoch = [e for e, m in best_metrics_by_epoch.items() if m['mcc_edge_h'] == best_edge_val][0]

        print(f"Best H4-MCC:                {best_h4_val:.4f} (Epoch {best_h4_epoch})")
        print(f"Best Pure MCC (Harmonic):   {best_pure_val:.4f} (Epoch {best_pure_epoch})")
        print(f"Best Edge MCC (Harmonic):   {best_edge_val:.4f} (Epoch {best_edge_epoch})")
        print("="*100)
