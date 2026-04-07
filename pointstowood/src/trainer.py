from src.dataset import create_train_loader, create_test_loader, create_inference_loader
from src.logger import MetricsTracker, ModelManager, HistoryLogger, WandbLogger
from src.statistics import calculate_harmonic_metrics, print_validation_summary, update_test_metrics_with_harmonic
from src.AnisotropicConv import AnisotropicConv
from tqdm import tqdm
import numpy as np
import torch
import os
import glob
import pandas as pd
from src.loss import FocalLoss, ContrastiveBoundaryLoss, ReflectanceFPPenalty
from torch.optim import AdamW
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter_max, scatter_add
from src.io import save_file
import warnings
import copy
import math


class GroupWeightTracker:
    """Epoch-level GroupDRO weight tracker.

    After each training epoch, call update() with per-group mean losses.
    The weights q_g are used to reweight per-sample losses in the next epoch,
    pushing the model to improve on underperforming biome groups.

    η (eta): step size for weight update. Too high → oscillation. Too low → no effect.
    Typical range: 0.01–0.05.
    """
    def __init__(self, eta: float = 0.01):
        self.eta = eta
        self.weights: dict = {}

    def update(self, group_losses: dict):
        for g, loss_val in group_losses.items():
            if g not in self.weights:
                self.weights[g] = 1.0
            self.weights[g] *= math.exp(self.eta * float(loss_val))
        total = sum(self.weights.values())
        if total > 0:
            for g in self.weights:
                self.weights[g] /= total

    def weight(self, group_name: str) -> float:
        n = max(len(self.weights), 1)
        return self.weights.get(group_name, 1.0 / n)

    def log_str(self) -> str:
        return ' | '.join(f'{g}:{w:.3f}' for g, w in sorted(self.weights.items()))

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


def run_validation_pass(model, test_loader, device, mode_name, augmentation_mode, args):
    """Run a single validation pass with specified augmentation mode."""
    # Temporarily modify the dataset mode
    original_mode = test_loader.dataset.mode
    test_loader.dataset.mode = augmentation_mode

    # Edge voxel size for ASD normalization (hardcoded in dataset.py:51 as 0.25m)
    # This is the resolution used to detect mixed-label voxels (edge points)
    test_tracker = MetricsTracker(full_metrics=True, edge_voxel_size=0.25)

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

    return test_tracker.get_averages()


def compute_refl_dominance_diagnostic(model, test_loader, device, args):
    """Run one batch with reflectance.requires_grad, backward from output sum; return mean |∂logit/∂refl|."""
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
            mean_abs_grad = float(data.reflectance.grad.abs().mean().cpu().item())
        else:
            mean_abs_grad = float("nan")
    except Exception:
        mean_abs_grad = float("nan")
    finally:
        test_loader.dataset.mode = orig_mode
        model.zero_grad(set_to_none=True)
    return mean_abs_grad



def _eval_metrics(classified_with_refl, classified_no_refl, gt_xyz_label, collect_grid_size):
    """Compute MCC and FPR for both refl conditions vs GT at collect grid resolution.

    Concatenates pred and GT points into one voxel_grid so cluster IDs are consistent,
    then compares argmax-|p-0.5| pred label against majority-vote GT label per voxel.

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

        return pred_label[valid].numpy(), gt_label[valid].numpy(), pred_c, gt_c, clusters, valid, n_clusters

    def _mcc_fpr(p, g):
        tp = int(((p == 1) & (g == 1)).sum())
        tn = int(((p == 0) & (g == 0)).sum())
        fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = float(tp * tn - fp * fn) / denom if denom > 0 else 0.0
        fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
        return mcc, fpr

    def _edge_mask(gt_label_full, clusters_full, n_clusters):
        """Voxels whose GT label differs from at least one neighbour in the cluster space."""
        from torch_scatter import scatter_add as sa
        # Simple proxy: voxels where GT is mixed within a 2x-coarser grid
        coarse_size = collect_grid_size * 3.0
        return None  # fallback: skip edge metric if too expensive

    if gt_xyz_label is None or len(gt_xyz_label) == 0:
        return {}

    try:
        pred_w, gt_w, *_ = _agg_pred(classified_with_refl, gt_xyz_label, collect_grid_size)
        pred_n, gt_n, *_ = _agg_pred(classified_no_refl,   gt_xyz_label, collect_grid_size)

        mcc_w, fpr_w = _mcc_fpr(pred_w, gt_w)
        mcc_n, fpr_n = _mcc_fpr(pred_n, gt_n)

        # Edge voxels: GT label is 1 but surrounded by disagreement proxy
        # Simple: voxels where pred_w != gt_w OR pred_n != gt_n (model-uncertain boundary)
        is_edge = (pred_w != gt_w) | (pred_n != gt_n)
        mcc_edge_w = _mcc_fpr(pred_w[is_edge], gt_w[is_edge])[0] if is_edge.sum() > 10 else float('nan')
        mcc_edge_n = _mcc_fpr(pred_n[is_edge], gt_n[is_edge])[0] if is_edge.sum() > 10 else float('nan')

        return {
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


def run_eval_visualization(model, args, device, epoch):
    """Run inference on eval voxels, aggregate to 4cm grid (argmax |p-0.5| per voxel), save PLY for visualization.

    Writes two outputs per source: one with reflectance (e.g. prefix_eval.ply) and one with reflectance
    zeroed for XYZ-only visualization (e.g. prefix_eval_xyz.ply). Supports multiple source files.
    """
    import re

    eval_vxfile = getattr(args, 'eval_vxfile', None)
    if not eval_vxfile or not os.path.isdir(eval_vxfile):
        if epoch == 1 and getattr(args, 'eval', False):
            print('[Eval] No eval voxels found; put a .ply in data/<region>_eval and run with --preprocess --eval first.')
        return
    all_keys = glob.glob(os.path.join(eval_vxfile, '*.pt'))
    if not all_keys:
        if epoch == 1 and getattr(args, 'eval', False):
            print('[Eval] Eval voxel folder is empty; put a .ply in data/<region>_eval and run with --preprocess --eval first.')
        return

    # Group voxel files by source prefix
    # Pattern: {prefix}_voxel_{num}.pt or voxel_{num}.pt (no prefix)
    source_files = {}
    for key in all_keys:
        basename = os.path.basename(key)
        match = re.match(r'(.+)_voxel_\d+\.pt$', basename)
        if match:
            prefix = match.group(1)
        else:
            prefix = '_default_'
        if prefix not in source_files:
            source_files[prefix] = []
        source_files[prefix].append(key)

    vis_dir = os.path.join(os.path.dirname(eval_vxfile), 'visualisations')
    os.makedirs(vis_dir, exist_ok=True)
    grid_size_model = getattr(args, 'eval_grid_size', None)
    if grid_size_model is None:
        grid_size_model = args.grid_size[0] if isinstance(args.grid_size, (list, tuple)) else args.grid_size
    collect_grid_size = getattr(args, 'eval_collect_grid_size', 0.04)

    # Process each source file separately
    for prefix, keys in source_files.items():
        # Build minimal args for inference loader
        class EvalArgs:
            pass
        eval_args = EvalArgs()
        eval_args.vxfile = eval_vxfile
        eval_args.batch_size = getattr(args, 'batch_size', 4)
        eval_args.max_pts = getattr(args, 'max_pts', 16384)
        eval_args.grid_size = [float(grid_size_model)]
        eval_args.max_points_per_batch = getattr(args, 'max_points_per_batch', 50000)
        eval_args.verbose = False
        eval_args.wdir = args.wdir
        eval_args.model = args.model
        eval_args.eval_file_pattern = f'{prefix}_voxel_*.pt' if prefix != '_default_' else 'voxel_*.pt'

        eval_loader, _ = create_inference_loader(eval_args, device)
        model.eval()
        output_list = []
        output_list_xyz = []

        with torch.no_grad():
            for data in eval_loader:
                data = data.to(device, non_blocking=True)
                data.voxel_size = torch.tensor([float(grid_size_model)], dtype=torch.float32, device=data.pos.device)
                batch_ids = data.batch.cpu()
                pos = data.pos.cpu()
                local_shift = data.local_shift.cpu()

                # Pass 1: with reflectance
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    outputs = model(data)
                    outputs = torch.nan_to_num(outputs)
                    probs = torch.sigmoid(outputs).float()
                probs_np = probs.cpu().numpy()
                batch_counts = torch.bincount(batch_ids)
                splits = torch.cumsum(batch_counts, dim=0).numpy()
                starts = np.concatenate(([0], splits[:-1]))
                for b, (s, e) in enumerate(zip(starts, splits)):
                    shift = local_shift[3 * b : 3 * b + 3]
                    pos_global = (pos[s:e] + shift).numpy()
                    output_list.append(np.column_stack((pos_global, probs_np[s:e].ravel())))

                # Pass 2: reflectance zeroed (XYZ-only, for visualization)
                data.reflectance = torch.zeros_like(data.reflectance, device=data.reflectance.device)
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    outputs_xyz = model(data)
                    outputs_xyz = torch.nan_to_num(outputs_xyz)
                    probs_xyz = torch.sigmoid(outputs_xyz).float()
                probs_xyz_np = probs_xyz.cpu().numpy()
                for b, (s, e) in enumerate(zip(starts, splits)):
                    shift = local_shift[3 * b : 3 * b + 3]
                    pos_global = (pos[s:e] + shift).numpy()
                    output_list_xyz.append(np.column_stack((pos_global, probs_xyz_np[s:e].ravel())))
                del data, outputs, outputs_xyz, probs, probs_xyz

        if not output_list:
            continue

        def aggregate_and_save(classified_arr, out_path):
            pos_t = torch.as_tensor(classified_arr[:, :3], dtype=torch.float32, device='cpu')
            prob_t = torch.as_tensor(classified_arr[:, 3], dtype=torch.float32, device='cpu')
            cluster = voxel_grid(pos_t, collect_grid_size)
            cluster, _ = consecutive_cluster(cluster)
            # Argmax on |p - 0.5|: pick the most confident point (either direction) to decide voxel label
            conf = torch.abs(prob_t - 0.5)
            _, argmax_idx = scatter_max(conf, cluster, dim=0)
            winning_prob = prob_t[argmax_idx]
            voxel_label = (winning_prob >= 0.5).long().numpy()
            point_labels = voxel_label[cluster.numpy()]
            pos_np = pos_t.numpy()
            out_df = pd.DataFrame({'x': pos_np[:, 0], 'y': pos_np[:, 1], 'z': pos_np[:, 2], 'label': point_labels})
            save_file(out_path, out_df, additional_fields=['label'], verbose=False)

        classified = np.vstack(output_list)
        classified_xyz = np.vstack(output_list_xyz)
        del output_list, output_list_xyz

        # Load GT labels directly from .pt files (inference loader strips labels)
        gt_points = []
        for key in keys:
            try:
                raw = torch.load(key, map_location='cpu', weights_only=True)
            except Exception:
                raw = torch.load(key, map_location='cpu', weights_only=False)
            pc = raw['point_cloud'] if isinstance(raw, dict) else raw
            if pc.shape[1] >= 5:
                gt_points.append(pc[:, [0, 1, 2, 4]].numpy())
        gt_all = np.vstack(gt_points) if gt_points else None

        # Compute and print eval metrics
        eval_m = _eval_metrics(classified, classified_xyz, gt_all, collect_grid_size)
        if eval_m:
            tag = prefix if prefix != '_default_' else 'eval'
            print(
                f'[Eval {tag}] E{epoch} | '
                f'MCC refl={eval_m["mcc_with_refl"]:.4f} | '
                f'MCC no-refl={eval_m["mcc_no_refl"]:.4f} | '
                f'ReflGain={eval_m["refl_gain"]:+.4f} | '
                f'ReflGain(edge)={eval_m["refl_gain_edge"]:+.4f}'
                if not np.isnan(eval_m.get("refl_gain_edge", float("nan")))
                else
                f'[Eval {tag}] E{epoch} | '
                f'MCC refl={eval_m["mcc_with_refl"]:.4f} | '
                f'MCC no-refl={eval_m["mcc_no_refl"]:.4f} | '
                f'ReflGain={eval_m["refl_gain"]:+.4f}'
            )
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.log({f'eval_{tag}/mcc_with_refl':      eval_m['mcc_with_refl'],
                                f'eval_{tag}/mcc_no_refl':        eval_m['mcc_no_refl'],
                                f'eval_{tag}/refl_gain':          eval_m['refl_gain'],
                                f'eval_{tag}/refl_gain_edge':     eval_m.get('refl_gain_edge', 0),
                                f'eval_{tag}/mcc_edge_with_refl': eval_m.get('mcc_edge_with_refl', 0),
                                f'eval_{tag}/mcc_edge_no_refl':   eval_m.get('mcc_edge_no_refl', 0),
                                f'eval_{tag}/fpr_with_refl':      eval_m['fpr_with_refl'],
                                f'eval_{tag}/fpr_no_refl':        eval_m['fpr_no_refl'],
                                'epoch': epoch})
            except Exception:
                pass

        # Output filenames: one with refl, one XYZ-only (zeroed reflectance)
        if prefix == '_default_':
            out_path = os.path.join(vis_dir, 'eval_latest.ply')
            out_path_xyz = os.path.join(vis_dir, 'eval_latest_xyz.ply')
        else:
            out_path = os.path.join(vis_dir, f'{prefix}_eval.ply')
            out_path_xyz = os.path.join(vis_dir, f'{prefix}_eval_xyz.ply')

        aggregate_and_save(classified, out_path)
        aggregate_and_save(classified_xyz, out_path_xyz)
        if getattr(args, 'verbose', False):
            print(f'Eval visualization saved: {out_path}, {out_path_xyz}')


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
    num_kernel_points = getattr(args, 'num_kernel_points', 16)
    dualnorm_lite = getattr(args, 'dualnorm_lite', False)
    spatial_mix_lite = getattr(args, 'spatial_mix_lite', False)
    lr = getattr(args, 'lr', 1e-3)
    if 'eu' in args.model.lower() or 'global' in args.model.lower():
        from src.model import NetFull as Net
        model = Net(num_classes=1, C=128, num_kernel_points=num_kernel_points, learnable_kernels=learnable_kernels, drop_path_rate=drop_path_rate, dualnorm_lite=dualnorm_lite, spatial_mix_lite=spatial_mix_lite).to(device)
        weight_decay = 1e-2
    else:
        from src.model import NetLight as Net
        model = Net(num_classes=1, C=16, num_kernel_points=8, learnable_kernels=True, drop_path_rate=drop_path_rate, dualnorm_lite=dualnorm_lite, spatial_mix_lite=spatial_mix_lite).to(device)
        weight_decay = 1e-2
    if spatial_mix_lite:
        print("SpatialMix-lite enabled (SA2 residual blocks)")
    
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

    if args.test:
        test_loader, _ = create_test_loader(args, device)

    # Cyclical Focal Loss: gamma cycles 0 → gamma_max → 0 over training
    # Early: gamma=0 (pure BCE, strong gradients for learning)
    # Mid: gamma=gamma_max (focus on hard examples)
    # End: gamma→0 (stabilize predictions)
    # gamma_max=0 → plain BCE; label_smoothing=0 → no smoothing
    gamma_max = getattr(args, 'gamma_max', 2.0)
    gamma_peak_pct = float(getattr(args, 'gamma_peak_pct', 0.5))
    gamma_peak_pct = min(0.95, max(0.05, gamma_peak_pct))
    label_smoothing = getattr(args, 'label_smoothing', 0.1)
    focal_alpha = getattr(args, 'focal_alpha', None)
    criterion = FocalLoss(
        gamma_max=gamma_max,
        alpha=focal_alpha,
        label_smoothing=label_smoothing,
        cyclical=True,
        pct_peak=gamma_peak_pct,
    )
    print(
        f"Loss: Focal gamma_max={gamma_max}, peak={gamma_peak_pct:.2f}, alpha={focal_alpha}, label_smoothing={label_smoothing}"
        + (" (plain BCE)" if gamma_max == 0 else "")
    )

    # Multi-scale Contrastive Boundary Learning (CBL) - faithful to CVPR 2022 paper
    # Applied at all encoder stages with label propagation through sub-sampling
    cbl_weight = getattr(args, 'cbl_weight', 0.25)
    cbl_ramp = getattr(args, 'cbl_ramp', True)
    cbl_ramp_pct = getattr(args, 'cbl_ramp_pct', 0.33)
    cbl_criterion = ContrastiveBoundaryLoss(
        k=16,
        temperature=1.0,
        weight=cbl_weight,
        boundary_threshold=0.1,
        ramp=cbl_ramp,
        ramp_pct=cbl_ramp_pct,
    )
    if cbl_ramp:
        print(f"CBL weight={cbl_weight} (ramp 0→1 over first {cbl_ramp_pct:.2f} of training)")
    else:
        print(f"CBL weight={cbl_weight} (no ramp)")
    refl_fp_weight = getattr(args, 'refl_fp_penalty', 0.0)
    refl_fp_ramp = getattr(args, 'refl_fp_ramp', True)
    refl_fp_flat = getattr(args, 'refl_fp_flat', True)
    refl_fp_criterion = ReflectanceFPPenalty(margin=1.0, strength=2.0, weight=refl_fp_weight, ramp=refl_fp_ramp, flat=refl_fp_flat) if refl_fp_weight > 0 else None
    if refl_fp_criterion is not None:
        ramp_str = "ramp 0→1 over epochs" if refl_fp_ramp else "full weight from epoch 1"
        mode_str = "all leaf points" if refl_fp_flat else "high-refl weighted"
        print(f"FP penalty: weight={refl_fp_weight} ({ramp_str}, {mode_str})")

    # GroupDRO: biome-aware loss reweighting
    # --group-dro implies gamma=0 (plain BCE); GroupDRO handles difficulty weighting at group level
    use_group_dro = getattr(args, 'group_dro', False)
    group_tracker = None
    criterion_none = None
    if use_group_dro:
        group_dro_eta = getattr(args, 'group_dro_eta', 0.01)
        group_tracker = GroupWeightTracker(eta=group_dro_eta)
        # Per-point loss needed for group-level reweighting — same params as criterion but no reduction
        criterion_none = FocalLoss(
            gamma_max=0.0,  # plain BCE per point; GroupDRO handles difficulty weighting
            alpha=focal_alpha,
            label_smoothing=label_smoothing,
            cyclical=False,
            reduction='none',
        )
        # Force gamma=0 on main criterion too — avoid double hard-example focusing
        criterion.gamma_max = 0.0
        criterion.current_gamma = 0.0
        group_list = train_dataset.group_list
        print(f"GroupDRO enabled: eta={group_dro_eta}, groups={group_list}, gamma forced to 0")

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
        print("Loading model")
        try:
            manager.load_model(os.path.join(args.wdir,'model',args.model))
        except (KeyError, RuntimeError, Exception) as e:
            print(f"Failed to load (corrupted or incompatible): {e}")
            print("Creating new model...")
            torch.save(model.state_dict(), os.path.join(args.wdir,'model',args.model))
    else:
        print("\nModel not found, creating new file...")
        torch.save(model.state_dict(), os.path.join(args.wdir,'model',args.model))

    best_h4_mcc = 0.0
    early_stop_best = -float('inf')
    early_stop_bad_epochs = 0

    amp_enabled, amp_dtype, amp_name = _resolve_amp_config(device, getattr(args, 'amp_dtype', 'auto'))
    # Grad scaling is only needed for fp16; bf16 has fp32-like exponent range.
    scaler = torch.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))
    print(f"AMP: {amp_name}")

    # EMA Loss tracking
    ema_loss = None
    ema_alpha = 0.9

    # EMA Model tracking: updated every step; decay per step (0.999 ~1k steps, 0.99 ~100 steps)
    use_ema = getattr(args, 'ema', False)
    ema_decay = getattr(args, 'ema_decay', 0.999)
    if use_ema:
        ema_model = EMAModel(model, decay=ema_decay)
        ema_model.register()
        print(f"EMA enabled (decay={ema_decay})")
    else:
        ema_model = None

    accumulation_steps = getattr(args, 'accumulation_steps', 4)
    print(f"Gradient accumulation: {accumulation_steps} steps")
    optimizer.zero_grad(set_to_none=True)
    accumulated_batches = 0  # Track actual accumulated batches

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        print(f"\n{'='*100}\nEPOCH {epoch}\n{'='*100}")

        criterion.set_epoch(epoch, args.num_epochs)
        cbl_criterion.set_epoch(epoch, args.num_epochs)
        if refl_fp_criterion is not None:
            refl_fp_criterion.set_epoch(epoch, args.num_epochs)
        refl_str = f" | ReflFP: {refl_fp_criterion.ramp_factor:.3f}" if refl_fp_criterion is not None else ""
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f} | Gamma: {criterion.current_gamma:.3f} | CBL: {cbl_criterion.ramp_factor:.3f}{refl_str}")
        train_tracker = MetricsTracker(full_metrics=False)  # Fast metrics only for training
        epoch_group_losses: dict = {}  # group_name -> list of per-sample losses (for GroupDRO update)

        # Max points per batch to avoid OOM (adjust based on your GPU)
        max_points_per_batch = getattr(args, 'max_points_per_batch', 50000)
        verbose = getattr(args, 'verbose', False)

        # Limit steps per epoch (0 = use all data)
        epoch_steps = getattr(args, 'epoch_steps', 0)
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

                    if not torch.isfinite(outputs).all():
                        print(f"[Warning] Non-finite model outputs at step {i}, skipping batch")
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_batches = 0
                        continue

                    if use_group_dro and hasattr(data, 'group_idx') and data.group_idx is not None:
                        # Per-point BCE (gamma=0), aggregate to per-sample, apply group weights
                        loss_per_point = criterion_none(outputs, data.y.float())  # [N_points]
                        n_samples = int(data.batch.max().item()) + 1

                        # Per-sample mean loss
                        pt_ones = torch.ones(loss_per_point.size(0), device=device)
                        per_sample_sum   = scatter_add(loss_per_point, data.batch, dim=0, dim_size=n_samples)
                        per_sample_count = scatter_add(pt_ones, data.batch, dim=0, dim_size=n_samples).clamp(1)
                        per_sample_loss  = per_sample_sum / per_sample_count  # [B]

                        # Group weights for each sample in this batch
                        group_ids = data.group_idx.view(-1)[:n_samples]  # [B]
                        sample_weights = torch.tensor(
                            [group_tracker.weight(group_list[gid.item()]) for gid in group_ids],
                            device=device, dtype=torch.float32
                        )
                        # Scale so weights sum to n_samples (mean stays ~same magnitude)
                        sample_weights = sample_weights * n_samples / sample_weights.sum().clamp(min=1e-8)

                        loss = (per_sample_loss * sample_weights).mean()

                        # Accumulate per-group losses for end-of-epoch weight update
                        for s in range(n_samples):
                            g_name = group_list[group_ids[s].item()]
                            if g_name not in epoch_group_losses:
                                epoch_group_losses[g_name] = []
                            epoch_group_losses[g_name].append(per_sample_loss[s].item())
                    else:
                        loss = criterion(outputs, data.y.float())

                    if hasattr(model, 'encoder_stages') and model.encoder_stages:
                        cbl_loss = cbl_criterion(
                            encoder_stages=model.encoder_stages,
                            labels=data.y
                        )
                        loss = loss + cbl_loss

                    if refl_fp_criterion is not None and getattr(data, 'reflectance', None) is not None:
                        loss = loss + refl_fp_criterion(outputs, data.y.float(), data.reflectance)

                    # Clamp for stability
                    loss = torch.clamp(loss, min=0.0, max=10.0)
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

                    # Explicit memory clearing (avoid empty_cache every batch - it's slow)
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
                tepoch.set_postfix({
                    'Lo': round(current_metrics['loss'], 5),
                    'EMA': round(ema_loss, 5),
                    'BAc': round(current_metrics['accuracy'], 3),
                    'Pr': round(current_metrics['precision'], 3),
                    'Re': round(current_metrics['recall'], 3),
                    'Fb': round(current_metrics['fbeta'], 3),
                })
                tepoch.update(1)
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

        lr_scheduler.step()

        train_metrics = train_tracker.get_averages()

        # GroupDRO: update group weights from this epoch's per-group mean losses
        if use_group_dro and epoch_group_losses:
            group_mean_losses = {g: float(np.mean(v)) for g, v in epoch_group_losses.items()}
            group_tracker.update(group_mean_losses)
            print(f"GroupDRO weights: {group_tracker.log_str()}")
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.log({f'group_dro/{g}': w for g, w in group_tracker.weights.items()} | {'epoch': epoch})
            except Exception:
                pass

        # Compact flashlight / reflectance diagnostics
        def _short_layer_name(layer_name: str) -> str:
            if layer_name.startswith("sa1_module"):
                return "sa1"
            if layer_name.startswith("sa2_module"):
                return "sa2"
            if layer_name.startswith("sa3_module"):
                return "sa3"
            return layer_name.replace("_module", "")

        kernel_metrics = []
        flashlight_tags = []
        similarity_list = []
        gate_list = []
        diag_rows = []
        diag_warnings = []
        dualnorm_alphas = []
        dualnorm_betas = []

        for name, module in model.named_modules():
            if hasattr(module, 'kernel_entropy'):
                kernel_metrics.append(module.kernel_entropy.item())
            if not isinstance(module, AnisotropicConv):
                continue

            layer_name = _short_layer_name(name.split('.conv')[0])
            similarity = getattr(module, 'last_similarity', None)
            gate_val = getattr(module, 'last_refl_gate', None)

            if similarity is not None:
                similarity_list.append(similarity)
            if gate_val is not None:
                gate_list.append(gate_val)

            if similarity is not None and gate_val is not None:
                flashlight_tags.append(f"{layer_name}(s={similarity:.3f},g={gate_val:.3f})")
            elif similarity is not None:
                flashlight_tags.append(f"{layer_name}(s={similarity:.3f})")
            elif gate_val is not None:
                flashlight_tags.append(f"{layer_name}(g={gate_val:.3f})")

            if hasattr(module, 'diagnostics'):
                diag = module.diagnostics
                eig_dom = diag.get('eigvals_dominant_mean', 0.0)
                eig_std = diag.get('eigvals_dominant_std', 0.0)
                contrast = diag.get('contrast_strength_mean', 0.0)
                gate_mean = diag.get('refl_gate_mean', None)
                if gate_mean is not None:
                    diag_rows.append(f"{layer_name}[eig={eig_dom:.3f}±{eig_std:.3f},c={contrast:.3f},g={gate_mean:.3f}]")
                else:
                    diag_rows.append(f"{layer_name}[eig={eig_dom:.3f}±{eig_std:.3f},c={contrast:.3f}]")

                if 'warning' in diag:
                    diag_warnings.append(f"{layer_name}: {diag['warning']}")
                if diag.get('has_nan', False):
                    diag_warnings.append(f"{layer_name}: NaN detected")

            if getattr(module, 'use_dualnorm_lite', False):
                if hasattr(module, 'last_dualnorm_alpha'):
                    dualnorm_alphas.append(module.last_dualnorm_alpha)
                if hasattr(module, 'last_dualnorm_beta'):
                    dualnorm_betas.append(module.last_dualnorm_beta)

        # Collect model internals for wandb
        contrast_strengths = []
        for name, module in model.named_modules():
            if isinstance(module, AnisotropicConv) and hasattr(module, 'diagnostics'):
                cs = module.diagnostics.get('contrast_strength_mean', 0.0)
                contrast_strengths.append(cs)

        model_metrics = {
            "refl_gate":        np.mean(gate_list) if gate_list else 0.0,
            "contrast_strength": np.mean(contrast_strengths) if contrast_strengths else 0.0,
            "dualnorm_alpha":   np.mean(dualnorm_alphas) if dualnorm_alphas else 0.0,
            "kernel_entropy":   sum(kernel_metrics) / len(kernel_metrics) if kernel_metrics else 0.0,
            "cbl_sample_std":   getattr(cbl_criterion, 'last_sample_std', 0.0),
        }

        if kernel_metrics:
            avg_entropy = sum(kernel_metrics) / len(kernel_metrics)
            print(f"Kernel Entropy: {avg_entropy:.2f}")
        if flashlight_tags:
            print("Flashlight: " + " | ".join(flashlight_tags))
        if similarity_list:
            if gate_list:
                print(f"Summary: sim={np.mean(similarity_list):.3f}, gate={np.mean(gate_list):.3f}")
            else:
                print(f"Summary: sim={np.mean(similarity_list):.3f}")
        if dualnorm_alphas and dualnorm_betas:
            print(f"DualNorm-lite: alpha={np.mean(dualnorm_alphas):.3f}, beta={np.mean(dualnorm_betas):.3f}")
        if diag_rows:
            print("Diag: " + " | ".join(diag_rows))
        if diag_warnings:
            print(f"Warnings: {'; '.join(diag_warnings)}")

        if args.test:
            model.eval()
            mean_dlogit_drefl = compute_refl_dominance_diagnostic(model, test_loader, device, args)
            if not np.isnan(mean_dlogit_drefl):
                print(f"Refl dominance: mean |∂logit/∂refl| = {mean_dlogit_drefl:.4f}")

            # Apply EMA weights for testing (if enabled)
            if ema_model is not None:
                ema_model.apply_shadow()

            test_metrics_with_refl = run_validation_pass(model, test_loader, device, "With Reflectance", "val_with_reflectance", args)
            test_metrics_no_refl = run_validation_pass(model, test_loader, device, "No Reflectance", "val_no_reflectance", args)

            harmonic_metrics = calculate_harmonic_metrics(test_metrics_with_refl, test_metrics_no_refl)
            print_validation_summary(epoch, harmonic_metrics)

            # Update test metrics with harmonic data for logging
            test_metrics = update_test_metrics_with_harmonic(test_metrics_with_refl.copy(), harmonic_metrics)
            test_metrics["refl_dominance"] = mean_dlogit_drefl if (np.isfinite(mean_dlogit_drefl) and not np.isnan(mean_dlogit_drefl)) else 0.0
            
            # Restore original weights for training
            if ema_model is not None:
                ema_model.restore()
        else:
            test_metrics = None

        history_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, test_metrics)
        wandb_logger.log_epoch(epoch, optimizer.param_groups[0]["lr"], train_metrics, test_metrics, model_metrics=model_metrics)

        if getattr(args, 'eval', False):
            if ema_model is not None:
                ema_model.apply_shadow()
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
            best_h4_mcc = manager.save_best_model(harmonic_metrics['h4_mcc'], best_h4_mcc, os.path.join(args.wdir,'model','h4mcc-' + os.path.basename(args.model)))

        if epoch == args.num_epochs:
            print("Saving final GLOBAL model")
            if ema_model is not None:
                ema_model.apply_shadow()
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.wdir,'model',args.model))
            if ema_model is not None:
                ema_model.restore()
