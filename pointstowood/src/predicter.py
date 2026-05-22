from src.dataset import create_inference_loader
import math
import os
import sys
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from src.io import save_file
from collections import OrderedDict
import gc
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter_max, scatter_add

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.setrecursionlimit(10 ** 8)


def _gmm_threshold(probs: np.ndarray, max_samples: int = 50_000) -> tuple[float, bool]:
    """Fit a 2-component GMM to voxel-level pwood scores.

    Returns (threshold, use_any_wood). Only trusts the GMM when the two components
    straddle 0.5 (one clearly leaf, one clearly wood). Otherwise signals that argmax
    is safer (use_any_wood=False).
    """
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm as _norm

    p = probs if len(probs) <= max_samples else np.random.default_rng(0).choice(probs, max_samples, replace=False)
    try:
        gmm = GaussianMixture(n_components=2, random_state=0, max_iter=200)
        gmm.fit(p.reshape(-1, 1))
        means = gmm.means_.flatten()
        stds  = np.sqrt(gmm.covariances_.flatten())
        lo, hi = (0, 1) if means[0] <= means[1] else (1, 0)

        # Only trust GMM when one component is clearly leaf (<0.5) and one clearly wood (>0.5)
        if not (means[lo] < 0.5 < means[hi]):
            return 0.5, False  # ambiguous — caller should fall back to argmax

        # Scan for Bayesian crossover between the two component PDFs
        x = np.linspace(float(means[lo]), float(means[hi]), 1000)
        pdf_lo = gmm.weights_[lo] * _norm.pdf(x, means[lo], stds[lo])
        pdf_hi = gmm.weights_[hi] * _norm.pdf(x, means[hi], stds[hi])
        sign_changes = np.where(np.diff(np.sign(pdf_lo - pdf_hi)))[0]
        thr = float(x[sign_changes[0]]) if sign_changes.size > 0 else float((means[lo] + means[hi]) / 2)
        return thr, True
    except Exception:
        return 0.5, False
        
# Memory tracking removed for clean output


def _tqdm_label(text: str) -> str:
    if sys.stdout.isatty():
        return f"  \033[94m{text}\033[0m"
    return f"  {text}"

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


def _coerce_stage_kernel_points(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        values = [int(v) for v in value.detach().cpu().view(-1).tolist()]
    elif isinstance(value, (list, tuple)):
        values = [int(v) for v in value]
    else:
        values = [int(value)]

    if len(values) == 1:
        values = values * 3
    elif len(values) != 3:
        raise ValueError(f"Invalid kernel spec in checkpoint: {values}")

    return tuple(values)


def _format_stage_kernel_points(stage_kernel_points):
    if stage_kernel_points is None:
        return "?"
    if stage_kernel_points[0] == stage_kernel_points[1] == stage_kernel_points[2]:
        return str(stage_kernel_points[0])
    return "/".join(str(k) for k in stage_kernel_points)


def _infer_model_config(model_name, checkpoint, state_dict):
    name = (model_name or "").lower()
    saved_cfg = checkpoint.get('model_config', {}) if isinstance(checkpoint, dict) else {}
    distill_cfg = checkpoint.get('distill_config', {}) if isinstance(checkpoint, dict) else {}

    def _count_stage_blocks(stage_prefix: str) -> int:
        block_ids = set()
        prefix = f'{stage_prefix}.residual_blocks.'
        for key in state_dict.keys():
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            block_id = suffix.split('.', 1)[0]
            if block_id.isdigit():
                block_ids.add(int(block_id))
        return len(block_ids)

    seg0 = None
    for key in ('seg_head.input_proj.0.weight', 'seg_head.compress.0.weight', 'seg_head.0.weight'):
        candidate = state_dict.get(key, None)
        if isinstance(candidate, torch.Tensor) and candidate.ndim == 3 and candidate.shape[1] % 3 == 0:
            seg0 = candidate
            break
    if seg0 is not None:
        c_base = int(seg0.shape[1] // 3)
    else:
        c_base = int(saved_cfg.get('c_base', distill_cfg.get('student_c', 128 if ('eu' in name or 'global' in name) else 16)))

    sa1_count = _count_stage_blocks('sa1_module')
    sa2_count = _count_stage_blocks('sa2_module')
    sa3_count = _count_stage_blocks('sa3_module')

    if 'model_family' in saved_cfg:
        model_family = str(saved_cfg['model_family'])
    elif sa1_count >= 4 or sa2_count >= 6 or sa3_count >= 2:
        model_family = 'full'
    elif sa1_count > 0 or sa2_count > 0 or sa3_count > 0:
        model_family = 'light'
    else:
        model_family = 'full' if (('eu' in name or 'global' in name) and c_base >= 64) else 'light'

    stage_kernel_points = _coerce_stage_kernel_points(saved_cfg.get('stage_kernel_points'))
    if stage_kernel_points is None:
        inferred_stage_kernel_points = []
        for stage_name in ('sa1_module', 'sa2_module', 'sa3_module'):
            k_tensor = state_dict.get(f'{stage_name}.conv.kernel_points', None)
            if not isinstance(k_tensor, torch.Tensor):
                inferred_stage_kernel_points = []
                break
            inferred_stage_kernel_points.append(int(k_tensor.shape[0]))
        if len(inferred_stage_kernel_points) == 3:
            stage_kernel_points = tuple(inferred_stage_kernel_points)
        else:
            stage_kernel_points = _coerce_stage_kernel_points(
                saved_cfg.get('num_kernel_points', distill_cfg.get('student_kernels', [16, 16, 16]))
            )

    num_kernel_points = int(stage_kernel_points[0])

    has_kernel_dirs = any(k.endswith('conv.kernel_dirs') for k in state_dict.keys())
    learnable_kernels = bool(saved_cfg.get('learnable_kernels', not has_kernel_dirs))
    if 'learnable_kernels' not in saved_cfg and 'student_learnable_kernels' in distill_cfg and model_family == 'light':
        learnable_kernels = bool(distill_cfg['student_learnable_kernels'])

    spatial_mix_lite = bool(saved_cfg.get('spatial_mix_lite', any('spatial_mix_scale' in k for k in state_dict.keys())))
    flash_dim = saved_cfg.get('flash_dim', None)
    if flash_dim is None:
        flash_w = state_dict.get('sa1_module.conv.flashlight_mlp.6.weight')
        flash_dim = int(flash_w.shape[0]) if isinstance(flash_w, torch.Tensor) and flash_w.ndim == 2 else 0
    flash_dim = int(flash_dim)
    memory_efficient_conv = bool(saved_cfg.get('memory_efficient_conv', False))
    sparse_max = bool(saved_cfg.get('sparse_max', True))
    if 'compressed_head' in saved_cfg:
        compressed_head = bool(saved_cfg['compressed_head'])
    else:
        compressed_head = any(k.startswith('seg_head.compress.') for k in state_dict.keys())

    # Infer compressed head dim directly from the compress layer weight shape
    _compress_w = state_dict.get('seg_head.compress.0.weight')
    if _compress_w is not None and isinstance(_compress_w, torch.Tensor):
        compressed_head_dim = int(_compress_w.shape[0])
    else:
        compressed_head_dim = int(saved_cfg.get('compressed_head_dim', 64))

    # Block counts per SA stage — read from checkpoint if present, else infer from state dict
    sa1_blocks = int(saved_cfg.get('sa1_blocks', sa1_count if sa1_count > 0 else (distill_cfg.get('sa1_blocks', 1) if model_family == 'light' else 4)))
    sa2_blocks = int(saved_cfg.get('sa2_blocks', sa2_count if sa2_count > 0 else (distill_cfg.get('sa2_blocks', 2) if model_family == 'light' else 6)))
    sa3_blocks = int(saved_cfg.get('sa3_blocks', sa3_count if sa3_count > 0 else (distill_cfg.get('sa3_blocks', 1) if model_family == 'light' else 2)))

    k_neighbors = int(saved_cfg.get('k_neighbors', distill_cfg.get('student_k_neighbors', 16)))

    return {
        'model_family': model_family,
        'c_base': c_base,
        'k_neighbors': k_neighbors,
        'stage_kernel_points': stage_kernel_points,
        'num_kernel_points': num_kernel_points,
        'learnable_kernels': learnable_kernels,
        'spatial_mix_lite': spatial_mix_lite,
        'flash_dim': flash_dim,
        'memory_efficient_conv': memory_efficient_conv,
        'sparse_max': sparse_max,
        'sa1_blocks': sa1_blocks,
        'sa2_blocks': sa2_blocks,
        'sa3_blocks': sa3_blocks,
        'compressed_head': compressed_head,
        'compressed_head_dim': compressed_head_dim,
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
    print(f"  Checkpoint   loaded {len(compatible)}/{len(model_state)} tensors ({coverage * 100:.1f}%)")
    if skipped_shape:
        print(f"  Checkpoint   skipped {len(skipped_shape)} tensors due to shape mismatch")
    if missing:
        print(f"  Checkpoint   missing {len(missing)} tensors after load")
    if unexpected:
        print(f"  Checkpoint   unexpected {len(unexpected)} tensors in checkpoint")
    return model
    
def _seeded_bfs_hysteresis(voxel_prob, voxel_centres, high_t, low_t, grid_size):
    """Seeded BFS on 26-connected voxel graph.

    Seeds (pwood >= high_t) anchor the wood skeleton. Any candidate voxel
    (pwood >= low_t) reachable from a seed through connected candidates is
    promoted to wood. Isolated low-probability voxels stay leaf.

    Fully vectorised adjacency construction (one numpy searchsorted pass per
    offset); single-pass scipy connected_components; O(26M log M) total.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    device = voxel_prob.device
    seed_t = voxel_prob >= high_t
    cand_t = voxel_prob >= low_t

    if not seed_t.any():
        return seed_t

    cand_idx = cand_t.nonzero(as_tuple=True)[0]
    M = int(cand_idx.shape[0])
    if M == 0:
        return seed_t

    # Transfer only candidates to CPU — float32 sufficient at 4cm grid spacing
    cpos = voxel_centres[cand_idx].cpu().numpy().astype(np.float32)
    is_seed = seed_t[cand_idx].cpu().numpy()

    # Integer voxel coords, normalised so minimum is at origin
    gi = np.floor((cpos - cpos.min(axis=0)) / grid_size).astype(np.int32)
    ext = gi.max(axis=0) + 2
    sy = int(ext[2])
    sx = int(ext[1]) * sy
    # Packed coords must be int64 — sx can reach ~6M for large 100m clouds
    g0 = gi[:, 0].astype(np.int64)
    g1 = gi[:, 1].astype(np.int64)
    g2 = gi[:, 2].astype(np.int64)
    packed = g0 * sx + g1 * sy + g2

    order = np.argsort(packed, kind='stable').astype(np.int32)
    sp = packed[order]

    # Build upper-triangle edge list: one vectorised pass per 26 offset
    OFFSETS = [(dx, dy, dz)
               for dx in (-1, 0, 1)
               for dy in (-1, 0, 1)
               for dz in (-1, 0, 1)
               if dx or dy or dz]

    src_parts, dst_parts = [], []
    arange_M = np.arange(M, dtype=np.int32)

    for dx, dy, dz in OFFSETS:
        nb = (g0 + dx) * sx + (g1 + dy) * sy + g2 + dz
        pos = np.searchsorted(sp, nb).clip(0, M - 1)
        hit = sp[pos] == nb
        s = arange_M[hit]
        d = order[pos[hit]]
        keep = s < d
        src_parts.append(s[keep])
        dst_parts.append(d[keep])

    del gi, g0, g1, g2, packed, sp, order, cpos

    all_src = np.concatenate(src_parts).astype(np.int32)
    all_dst = np.concatenate(dst_parts).astype(np.int32)
    del src_parts, dst_parts

    if len(all_src) == 0:
        return seed_t

    adj = csr_matrix((np.ones(len(all_src), dtype=np.bool_), (all_src, all_dst)), shape=(M, M))
    adj = (adj + adj.T).astype(np.bool_)
    del all_src, all_dst

    _, labels = connected_components(adj, directed=False)
    del adj

    seed_comps = np.unique(labels[is_seed])
    promoted = np.isin(labels, seed_comps)

    out = seed_t.clone()
    out[cand_idx] = torch.as_tensor(promoted, dtype=torch.bool, device=device)
    return out


class GridCloudClassifier:
    """Aggregate per-voxel.

    Default behavior is argmax |p-0.5| per voxel; passing ``any_wood`` switches
    to the "any point above threshold => wood" rule; passing ``hysteresis``
    uses seeded BFS — high-confidence seeds flood through connected candidates.
    """
    def __init__(self, any_wood: float, grid_size: float, max_probability: bool = False,
                 min_votes: int = 1, hysteresis=None, auto_threshold: bool = False):
        self.any_wood = any_wood
        self.grid_size = grid_size
        self.max_probability = max_probability
        self.min_votes = max(1, int(min_votes))
        self.hysteresis = hysteresis  # (high_t, low_t) tuple or None
        self.auto_threshold = auto_threshold

    def collect_predictions(self, class_pos: np.ndarray, class_preds: np.ndarray,
                            class_probs: np.ndarray, original: pd.DataFrame) -> pd.DataFrame:
        """Aggregate predictions to voxels using vectorized scatter ops on GPU when available.

        Large post-inference clouds (50M+ points) dominate runtime if aggregation stays on
        CPU. voxel_grid / consecutive_cluster / scatter_* all run on CUDA, so we keep the
        whole pipeline device-side and only cross to CPU once for the final DataFrame merge.
        """
        original = original.drop(columns=[c for c in original.columns if c in ['prediction', 'pwood', 'pleaf']])

        orig_pos_np = original[['x','y','z']].values.astype(np.float32)
        class_probs = np.clip(class_probs, 0.0, 1.0).astype(np.float32)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        orig_pos_t = torch.as_tensor(orig_pos_np, dtype=torch.float32, device=device)
        class_pos_t = torch.as_tensor(class_pos, dtype=torch.float32, device=device)

        # voxel_grid uses the input cloud's minimum as origin, so calling it separately
        # on orig and class produces incompatible cluster IDs even for the same physical
        # voxel. Must call on the joint cloud so both share one consistent grid origin.
        n_orig = len(orig_pos_np)
        all_pos_t = torch.cat([orig_pos_t, class_pos_t])
        all_cluster = voxel_grid(all_pos_t, self.grid_size)
        del orig_pos_t, all_pos_t
        # Keep class_pos_t alive for hysteresis voxel-centre computation
        if self.hysteresis is None:
            del class_pos_t
            class_pos_t = None

        all_cluster, _ = consecutive_cluster(all_cluster)
        n_clusters = int(all_cluster.max().item()) + 1
        orig_cluster = all_cluster[:n_orig]
        class_cluster = all_cluster[n_orig:]
        del all_cluster

        prob_t = torch.as_tensor(class_probs, dtype=torch.float32, device=device)

        if self.max_probability:
            # Vectorized argmax |p-0.5| per voxel via scatter_max
            conf = torch.abs(prob_t - 0.5)
            _, argmax_idx = scatter_max(conf, class_cluster, dim=0, dim_size=n_clusters)
            argmax_idx = argmax_idx.clamp(0, len(class_probs) - 1)
            class_preds_t = torch.as_tensor(class_preds, dtype=torch.int64, device=device)
            voxel_label = class_preds_t[argmax_idx]

            # Mean prob per voxel as a fast proxy for median
            prob_sum = scatter_add(prob_t, class_cluster, dim=0, dim_size=n_clusters)
            count = scatter_add(torch.ones_like(prob_t), class_cluster, dim=0, dim_size=n_clusters)
            voxel_prob = prob_sum / count.clamp(min=1)
        else:
            voxel_prob, _ = scatter_max(prob_t, class_cluster, dim=0, dim_size=n_clusters)
            voxel_prob = voxel_prob.clamp(0.0, 1.0)

            # Adaptive threshold: fit 2-component GMM to voxel-level max pwood.
            # Running on voxel_prob (not raw point probs) gives a clean bimodal
            # distribution that works regardless of scene wood fraction.
            any_wood = self.any_wood
            if self.auto_threshold:
                vp_np = voxel_prob.cpu().numpy()
                gmm_thr, gmm_ok = _gmm_threshold(vp_np)
                if gmm_ok:
                    any_wood = gmm_thr
                    print(f'\n  Auto-threshold  GMM  → {any_wood:.3f}  '
                          f'(voxel pwood [{vp_np.min():.3f}, {vp_np.max():.3f}])')
                else:
                    print(f'\n  Auto-threshold  GMM ambiguous → threshold 0.5  '
                          f'(voxel pwood [{vp_np.min():.3f}, {vp_np.max():.3f}])')

            if self.hysteresis is not None:
                # Voxel centres via scatter mean of class positions
                pos_sum = torch.zeros(n_clusters, 3, dtype=torch.float32, device=device)
                pos_sum.scatter_add_(0, class_cluster.unsqueeze(1).expand(-1, 3), class_pos_t)
                cnt = scatter_add(torch.ones(class_cluster.shape[0], dtype=torch.float32, device=device),
                                  class_cluster, dim=0, dim_size=n_clusters)
                voxel_centres = pos_sum / cnt.clamp(min=1).unsqueeze(1)
                del class_pos_t, pos_sum, cnt
                high_t, low_t = self.hysteresis
                voxel_label = _seeded_bfs_hysteresis(
                    voxel_prob, voxel_centres, high_t, low_t, self.grid_size
                ).to(torch.int64)
                del voxel_centres
            elif self.min_votes > 1:
                above = (prob_t >= any_wood).float()
                vote_count = scatter_add(above, class_cluster, dim=0, dim_size=n_clusters)
                voxel_label = (vote_count >= self.min_votes).to(torch.int64)
            else:
                voxel_label = (voxel_prob >= any_wood).to(torch.int64)

        # Gather per-point on GPU, then single D2H transfer for the merge
        point_labels = voxel_label[orig_cluster].cpu().numpy()
        point_probs = voxel_prob[orig_cluster].cpu().numpy()
        original.loc[:, ['prediction', 'pwood']] = np.column_stack([point_labels, point_probs])
        return original

def load_inference_model(args, device=None):
    """Load checkpoint, build network, and return a model ready for inference.

    Separated from SemanticSegmentation so callers processing multiple files can
    load once and pass the result into each SemanticSegmentation call, avoiding
    repeated disk I/O and GPU allocation.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = os.path.join(args.wdir, 'model', args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'No model found at {model_path}')

    checkpoint, checkpoint_state = _load_checkpoint(model_path, device)
    inferred = _infer_model_config(args.model, checkpoint, checkpoint_state)

    if inferred['model_family'] == 'full':
        from src.model import NetFull as Net
        model = Net(
            num_classes=1,
            C=inferred['c_base'],
            num_kernel_points=inferred['stage_kernel_points'],
            learnable_kernels=inferred['learnable_kernels'],
            drop_path_rate=0.0,
            spatial_mix_lite=inferred['spatial_mix_lite'],
            flash_dim=inferred['flash_dim'],
            memory_efficient_conv=inferred['memory_efficient_conv'],
            sparse_max=inferred['sparse_max'],
            compressed_head=inferred['compressed_head'],
            compressed_head_dim=inferred['compressed_head_dim'],
            k_neighbors=inferred['k_neighbors'],
        ).to(device)
        model_label = "EU/Global"
    else:
        from src.model import NetLight as Net
        model = Net(
            num_classes=1,
            C=inferred['c_base'],
            num_kernel_points=inferred['stage_kernel_points'],
            learnable_kernels=inferred['learnable_kernels'],
            drop_path_rate=0.0,
            spatial_mix_lite=inferred['spatial_mix_lite'],
            sa1_blocks=inferred['sa1_blocks'],
            sa2_blocks=inferred['sa2_blocks'],
            sa3_blocks=inferred['sa3_blocks'],
            k_neighbors=inferred['k_neighbors'],
            flash_dim=inferred['flash_dim'],
            memory_efficient_conv=inferred['memory_efficient_conv'],
            sparse_max=inferred['sparse_max'],
            compressed_head=inferred['compressed_head'],
            compressed_head_dim=inferred['compressed_head_dim'],
        ).to(device)
        model_label = "Biome"

    kernel_mode = "learnable" if inferred['learnable_kernels'] else "fixed"
    param_count = sum(p.numel() for p in model.parameters())
    param_str = f"{param_count / 1e6:.2f}M" if param_count >= 1e6 else f"{param_count / 1e3:.1f}K"
    print(
        f"  Model        {model_label} | C={inferred['c_base']} | k={inferred['k_neighbors']} | "
        f"K={_format_stage_kernel_points(inferred['stage_kernel_points'])} | kernels={kernel_mode} | {param_str} params"
    )

    try:
        load_model(model, checkpoint_state)
    except KeyError:
        raise Exception(f'No model loaded at {os.path.join(args.wdir, "model", args.model)}')

    # Auto-scale point budget for lighter models once at load time.
    _DEFAULT_MAX_PTS = 32768
    _REF_PARAMS = 20_000_000
    if getattr(args, 'batch_size', 0) == 0 and getattr(args, 'max_points_per_batch', _DEFAULT_MAX_PTS) == _DEFAULT_MAX_PTS:
        _scale = min(_REF_PARAMS / max(param_count, 100_000), 8.0)
        if _scale > 1.05:
            args.max_points_per_batch = int(_DEFAULT_MAX_PTS * _scale)

    model.eval()
    return model


def SemanticSegmentation(args, model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model is None:
        model = load_inference_model(args, device)

    test_loader, test_dataset = create_inference_loader(args, device)

    model.eval()
    default_grid_size = args.grid_size[0] if isinstance(args.grid_size, (list, tuple)) else args.grid_size

    grid_size = getattr(args, 'collect_grid_size', 0.04) or 0.04
    use_any_wood = getattr(args, 'any_wood', None) is not None
    use_hysteresis = getattr(args, 'hysteresis', None) is not None


    # bf16 on Ampere+, fp16 otherwise. bf16 avoids grad scaling (inference is moot anyway)
    # and keeps fp32 exponent range — generally faster and more stable than fp16.
    from src.trainer import _resolve_amp_config
    amp_enabled, amp_dtype, _ = _resolve_amp_config(device, getattr(args, 'amp_dtype', 'auto'))

    # Collect all predictions first (unavoidable for voxel remapping), but do minimal processing
    all_pos = []
    all_preds = []
    all_probs = []

    dual_perspective = getattr(args, 'dual_perspective', False)
    base_passes = [("Inference", False)]
    if dual_perspective:
        base_passes.append(("Inference (no-refl)", True))

    # TTA: z-axis rotations. Rotates xyz in the local voxel frame so the
    # Fibonacci-sphere kernels see each neighbourhood from different angles.
    # Reported positions stay in the unrotated frame (global coords must not change).
    # Reflectance is scalar and rotation-invariant — untouched.
    tta_count = max(1, int(getattr(args, 'tta', 1)))
    if getattr(args, '_tta_angles', None) is not None:
        # Explicit angle list from sweep scripts — overrides all logic below.
        tta_angles = list(args._tta_angles)
        tta_count = len(tta_angles)
    elif tta_count == 2:
        # K=16 Fibonacci sphere: 15 directional kernels have azimuthal gaps of
        # 3.8°–32.5°. The largest gap is 32.46°; rotating by half that (16.23°)
        # places the second pass's kernels at the centre of every widest blind spot.
        tta_angles = [0.0, math.radians(16.23)]
    else:
        tta_angles = [(2.0 * math.pi * i) / tta_count for i in range(tta_count)]

    passes = []
    for theta in tta_angles:
        for pass_desc, zero_refl in base_passes:
            if tta_count > 1:
                label = f"{pass_desc} rot{int(round(math.degrees(theta))):>3d}°"
            else:
                label = pass_desc
            passes.append((label, zero_refl, theta))

    for pass_desc, zero_refl, theta in passes:
        print()
        # Precompute rotation matrix on the correct device/dtype lazily per pass.
        rotate = abs(theta) > 1e-9
        with tqdm(total=len(test_loader), colour='white', ascii="▒█", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', desc=_tqdm_label(pass_desc)) as pbar:
            for batch_idx, data in enumerate(test_loader):
                data = data.to(device, non_blocking=True)
                # Use batch voxel_size when collate set it (single-resolution batch); else fallback to args
                if getattr(data, 'voxel_size', None) is None:
                    data.voxel_size = torch.tensor([float(default_grid_size)], dtype=torch.float32, device=data.pos.device)
                elif data.voxel_size.device != data.pos.device:
                    data.voxel_size = data.voxel_size.to(data.pos.device)

                if zero_refl:
                    data.reflectance = torch.zeros_like(data.reflectance)

                # Keep a copy of unrotated xyz — predictions are reported at these
                # coords so global position (local_shift + xyz) stays correct.
                if rotate:
                    orig_xyz = data.pos[:, :3].clone()
                    c, s = math.cos(theta), math.sin(theta)
                    R = torch.tensor(
                        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                        dtype=data.pos.dtype, device=data.pos.device,
                    )
                    # Rotate x,y in place; z is preserved by R (gravity-aware).
                    data.pos = torch.cat([orig_xyz @ R.T, data.pos[:, 3:]], dim=1) if data.pos.size(1) > 3 else orig_xyz @ R.T
                else:
                    orig_xyz = None

                with torch.inference_mode(), torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
                    outputs = model(data)
                    outputs = torch.nan_to_num(outputs)
                    probs = torch.sigmoid(outputs).float()

                # Compute preds on-device to save one numpy pass, then transfer to CPU.
                preds_gpu = (probs >= args.is_wood).to(torch.int64)
                batch_ids = data.batch.cpu().numpy()
                # Report at original (unrotated) xyz so global coords stay correct.
                pos_src = orig_xyz if orig_xyz is not None else data.pos[:, :3]
                pos = pos_src.cpu().numpy()
                probs_np = probs.cpu().numpy().astype(np.float32)
                preds_np = preds_gpu.cpu().numpy()
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
                pbar.update(1)

    # Concatenate all predictions
    classified_pos = np.concatenate(all_pos, dtype=np.float32)
    classified_preds = np.concatenate(all_preds, dtype=np.int64)
    classified_probs = np.concatenate(all_probs, dtype=np.float32)

    del all_pos, all_preds, all_probs
    gc.collect()

    use_auto_threshold = getattr(args, 'auto_threshold', False)
    use_adaptive = use_auto_threshold or use_any_wood

    grid_classifier = GridCloudClassifier(
        any_wood=args.any_wood if use_any_wood else 0.5,
        grid_size=grid_size,
        max_probability=not use_adaptive and not use_hysteresis,
        min_votes=getattr(args, 'min_votes', 1),
        hysteresis=tuple(args.hysteresis) if use_hysteresis else None,
        auto_threshold=use_auto_threshold,
    )

    args.pc = grid_classifier.collect_predictions(classified_pos, classified_preds, classified_probs, args.pc)

    headers = list(dict.fromkeys(args.headers + ['prediction', 'pwood']))
    save_file(args.odir, args.pc.copy(), additional_fields=headers, verbose=False)

    return args
