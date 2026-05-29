import datetime
start = datetime.datetime.now()
import time
import resource
import os
import os.path as OP
import argparse
from src.preprocessing import preprocess
from src.predicter import SemanticSegmentation, load_inference_model
from src.utils import preprocess_point_cloud_data
import torch
import shutil
import sys
import numpy as np
import re
from src.io import load_file
from src.utils import configure_threads
from src.memory_utils import format_memory_info
import psutil


def _z_quantile_edges(z: np.ndarray, bins: int) -> np.ndarray:
    return np.quantile(z, np.linspace(0.0, 1.0, int(bins) + 1))


def _z_stratified_sample_indices(xyz: np.ndarray, sample_size: int = 2048, z_bins: int = 24) -> np.ndarray:
    """Sample indices approximately uniformly over z-quantile bins."""
    n = int(xyz.shape[0])
    if n <= sample_size:
        return np.arange(n, dtype=np.int64)

    z = xyz[:, 2]
    bins = int(max(1, min(z_bins, sample_size)))
    q = _z_quantile_edges(z, bins)
    per_bin = max(1, sample_size // bins)

    selected = []
    for i in range(bins):
        if i < bins - 1:
            mask = (z >= q[i]) & (z < q[i + 1])
        else:
            mask = (z >= q[i]) & (z <= q[i + 1])
        ids = np.where(mask)[0]
        if ids.size == 0:
            continue
        take = min(per_bin, ids.size)
        selected.append(np.random.choice(ids, size=take, replace=False))

    if selected:
        idx = np.concatenate(selected)
    else:
        idx = np.random.choice(n, size=sample_size, replace=False)

    if idx.size < sample_size:
        rem = np.setdiff1d(np.arange(n, dtype=np.int64), idx, assume_unique=False)
        if rem.size > 0:
            extra = np.random.choice(rem, size=min(sample_size - idx.size, rem.size), replace=False)
            idx = np.concatenate([idx, extra])

    if idx.size > sample_size:
        idx = np.random.choice(idx, size=sample_size, replace=False)
    return idx.astype(np.int64, copy=False)


def _robust_spacing_from_dists(dists: np.ndarray, k_use: int = 3, eps: float = 1e-8) -> np.ndarray:
    """Per-point spacing from the nearest positive neighbors.

    Ignore self/duplicate zero-distance neighbors, then use the median of the
    first `k_use` positive distances for each sampled point.
    """
    dists = np.asarray(dists)
    if dists.ndim == 1:
        dists = dists[:, None]

    spacing = np.full(dists.shape[0], np.nan, dtype=np.float64)
    for i in range(dists.shape[0]):
        row = dists[i]
        valid = row[np.isfinite(row) & (row > eps)]
        if valid.size == 0:
            continue
        spacing[i] = float(np.median(valid[:min(k_use, valid.size)]))
    return spacing


def _estimate_nn_spacing_m(xyz: np.ndarray, sample_size: int = 2048) -> float:
    """Estimate native point spacing from nearest-neighbor distances.

    Important: query sampled points against a KD-tree built on the full cloud.
    Using sample->sample NN overestimates spacing on large clouds.
    """
    n = int(xyz.shape[0])
    if n < 4:
        return float("nan")

    m = min(sample_size, n)
    z_bins = int(max(1, min(24, m)))
    z_full = xyz[:, 2]
    q = _z_quantile_edges(z_full, z_bins)
    sel = _z_stratified_sample_indices(xyz, sample_size=m, z_bins=z_bins)
    sample = xyz[sel].astype(np.float64, copy=False)
    xyz_full = xyz.astype(np.float64, copy=False)
    query_k = min(max(8, 4), n)

    try:
        from pykdtree.kdtree import KDTree
        tree = KDTree(xyz_full)
        dists, _ = tree.query(sample, k=query_k)
    except Exception:
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=query_k, algorithm='kd_tree', n_jobs=1).fit(xyz_full)
            dists, _ = nn.kneighbors(sample)
        except Exception:
            # Last-resort fallback for environments without KD-tree deps.
            t = torch.as_tensor(sample, dtype=torch.float32)
            d = torch.cdist(t, t)
            d.fill_diagonal_(float('inf'))
            dists = d.topk(k=min(4, d.shape[1]), dim=1, largest=False).values.cpu().numpy()

    point_spacing = _robust_spacing_from_dists(dists, k_use=3)
    sample_z = sample[:, 2]

    # Preserve the z-stratified logic: estimate a typical spacing per vertical
    # slice, then take the median across slices.
    slice_medians = []
    for i in range(z_bins):
        if i < z_bins - 1:
            mask = (sample_z >= q[i]) & (sample_z < q[i + 1])
        else:
            mask = (sample_z >= q[i]) & (sample_z <= q[i + 1])
        if not np.any(mask):
            continue
        slice_spacing = point_spacing[mask]
        slice_spacing = slice_spacing[np.isfinite(slice_spacing) & (slice_spacing > 0)]
        if slice_spacing.size > 0:
            slice_medians.append(float(np.median(slice_spacing)))

    if slice_medians:
        return float(np.median(np.asarray(slice_medians, dtype=np.float64)))

    point_spacing = point_spacing[np.isfinite(point_spacing) & (point_spacing > 0)]
    if point_spacing.size == 0:
        return float("nan")
    return float(np.median(point_spacing))


def _auto_grid_size_from_xyz(xyz: np.ndarray, min_grid: float = 1.0):
    """Infer grid size from native spacing: spacing(cm) -> nearest whole-meter grid (min 1m)."""
    spacing_m = _estimate_nn_spacing_m(xyz)
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        return [float(min_grid)], {
            'spacing_m': float('nan'),
            'rounded_spacing_m': float(min_grid) / 100.0,
            'chosen_grid': float(min_grid),
        }

    spacing_cm = spacing_m * 100.0
    rounded_cm = max(float(min_grid), float(np.round(spacing_cm)))
    chosen_grid = max(float(min_grid), float(np.round(rounded_cm)))
    return [float(chosen_grid)], {
        'spacing_m': float(spacing_m),
        'rounded_spacing_m': float(rounded_cm) / 100.0,
        'chosen_grid': float(chosen_grid),
    }


def _auto_collect_grid_size_m(resolution_m: float, grid_sizes_m):
    """Auto-set post-aggregation grid as 2x effective input spacing, floored at 4cm."""
    spacing_m = _effective_point_resolution_m(resolution_m, grid_sizes_m)
    collect_m = max(0.04, 2.0 * spacing_m)
    collect_m = round(collect_m * 100.0) / 100.0  # nearest cm in meters
    return collect_m, spacing_m


def _print_heading(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _print_item(label: str, value: str) -> None:
    line = f"  {label:<12} {value}"
    print(line)


def _fmt_yes_no(flag: bool) -> str:
    return "yes" if bool(flag) else "no"


def _accent(text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[38;5;214m{text}\033[0m"
    return text


def _green(text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[32m{text}\033[0m"
    return text


def _effective_point_resolution_m(resolution_m: float | None, grid_sizes_m) -> float:
    if resolution_m is not None and float(resolution_m) > 0.0:
        return float(resolution_m)
    if not grid_sizes_m:
        return 0.02
    return max(1e-4, float(min(grid_sizes_m)) / 100.0)


def _format_point_resolution_choice(resolution_m: float | None, user_provided: bool, auto_meta=None, grid_sizes_m=None) -> str:
    effective_m = _effective_point_resolution_m(resolution_m, grid_sizes_m)
    eff_str = _accent(f"{effective_m:.4f} m")
    if user_provided:
        return f"user -> {eff_str}"
    if auto_meta is None:
        return f"auto -> {eff_str}"
    spacing_m = auto_meta.get('spacing_m', float('nan'))
    if np.isfinite(spacing_m) and spacing_m > 0:
        return f"{spacing_m:.4f} m -> {eff_str}"
    return f"unavailable -> {eff_str}"


def _format_grid_choice(grid_sizes, user_provided: bool, auto_meta=None) -> str:
    grid_list = grid_sizes if isinstance(grid_sizes, (list, tuple)) else [grid_sizes]
    rendered = ", ".join(f"{float(g):.1f} m" for g in grid_list)
    if user_provided:
        return f"user -> {_accent(rendered)}"
    if auto_meta is None:
        return f"auto -> {_accent(rendered)}"
    chosen_grid = auto_meta.get('chosen_grid', float(grid_list[0]))
    return f"auto -> {_accent(f'{chosen_grid:.1f} m')}"


def _format_collect_choice(collect_grid_size: float, user_provided: bool) -> str:
    collect_m = float(collect_grid_size)
    if user_provided:
        return f"user -> {_accent(f'{collect_m:.4f} m')}"
    return f"auto -> {_accent(f'{collect_m:.4f} m')}"


class PerformanceTracker:
    def __init__(self, process_name, reset_gpu_stats=True):
        self.process_name = process_name
        self.process = psutil.Process(os.getpid())
        self.start_time = time.perf_counter()

        # Track both resource (peak) and current RSS at start
        self.start_rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / (1024**3)
        self.start_rss_current = self.process.memory_info().rss / (1024**3)

        self.start_gpu = 0
        if torch.cuda.is_available():
            # Only reset peak stats for inference, not preprocessing
            if reset_gpu_stats:
                torch.cuda.reset_peak_memory_stats()
            self.start_gpu = torch.cuda.memory_allocated() / (1024**3)
            self.start_gpu_peak = torch.cuda.max_memory_allocated() / (1024**3)

    def finish(self):
        # Final measurements
        end_time = time.perf_counter()

        # CPU measurements - both peak tracking methods
        end_rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / (1024**3)
        end_rss_current = self.process.memory_info().rss / (1024**3)
        current_vms = self.process.memory_info().vms / (1024**3)

        gpu_current = 0
        gpu_peak = 0
        if torch.cuda.is_available():
            gpu_current = torch.cuda.memory_allocated() / (1024**3)
            gpu_peak = torch.cuda.max_memory_allocated() / (1024**3)

        # Calculate differences
        duration = end_time - self.start_time
        cpu_peak_increase = end_rss_peak - self.start_rss_peak  # Peak memory increase during this stage
        cpu_current_increase = end_rss_current - self.start_rss_current  # Current memory increase
        gpu_peak_used = max(gpu_peak - self.start_gpu_peak, gpu_peak - self.start_gpu)

        return {
            'name': self.process_name,
            'duration': duration,
            'cpu_current_rss': end_rss_current,
            'cpu_current_vms': current_vms,
            'cpu_peak_increase': cpu_peak_increase,  # Memory increase during this stage
            'cpu_peak_absolute': end_rss_peak,      # Absolute peak reached
            'cpu_current_increase': cpu_current_increase,
            'gpu_current': gpu_current,
            'gpu_peak_used': gpu_peak_used
        }

def get_path(location: str = "") -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, location).replace("\\", "/") if location else base


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='PointsToWood: wood/leaf segmentation for point clouds. '
                    'Run without flags for standard quality, add --fast for a quick preview.')
    parser.add_argument('--point-cloud', '-p', required=True, nargs='+', type=str,
                        help='One or more input point cloud files.')
    parser.add_argument('--model', type=str, default=None,
                        help='Model checkpoint name inside pointstowood/model. If omitted, resolved from --region.')
    parser.add_argument('--region', type=str, default=None,
                        choices=['eu', 'finland', 'poland', 'spain'],
                        help='Biome region — auto-selects the distilled student model (finland/poland/spain) '
                             'or EU teacher (eu, default).')
    parser.add_argument('--thorough', action='store_true',
                        help='Thorough mode: two scales, 8 overlap offsets per scale, 4× z-TTA, dual-perspective '
                             '(reflectance + geometry). Significantly slower but maximum accuracy.')
    parser.add_argument('--auto-threshold', action='store_true',
                        help='Adaptive wood threshold: fits a 2-component GMM to the per-scene '
                             'pwood distribution and thresholds at the Bayesian crossover. '
                             'Density-agnostic — works regardless of scene wood fraction.')
    parser.add_argument('--any-wood', type=float, default=None, nargs='?', const=0.5,
                        help='Higher-recall mode: label a voxel as wood if any point has pwood >= threshold (default 0.5). '
                             'Catches more branches at the cost of some precision. '
                             'Default (no flag) uses argmax aggregation for higher precision.')

    adv = parser.add_argument_group('advanced')
    adv.add_argument('--inference-level', type=int, default=2, choices=[1, 2, 3, 4],
                     help='1=single scale no overlap, 2=two scales no overlap (default), '
                          '3=two scales 8 overlaps, 4=level 3 + dual-perspective.')
    adv.add_argument('--tta', type=int, default=2, choices=[1, 2, 4, 8],
                     help='Test-time augmentation: N z-axis yaw rotations (default 2).')
    adv.add_argument('--batch-size', default=0, type=int,
                     help='Mini-batch size. 0=adaptive (default).')
    adv.add_argument('--num-procs', default=-1, type=int,
                     help='CPU cores. -1=auto.')
    adv.add_argument('--resolution', type=float, default=0.0,
                     help='Representative spacing before voxel blocks [m]. 0=adaptive (default).')
    adv.add_argument('--grid-size', type=float, nargs='+', default=None,
                     help='Voxel block size(s) in metres. Auto-estimated from point spacing if omitted (min 1m).')
    adv.add_argument('--min-pts', type=int, default=512,
                     help='Minimum points per voxel (default 512).')
    adv.add_argument('--max-pts', type=int, default=32768,
                     help='Maximum points per voxel after preprocessing (default 32768).')
    adv.add_argument('--max-points-per-batch', type=int, default=32768,
                     help='Point budget per batch when --batch-size=0 (default 32768).')
    adv.add_argument('--memory-fraction', type=float, default=0.7,
                     help='Fraction of GPU memory for batching (default 0.7).')
    adv.add_argument('--verbose', action='store_true', help='Print extra debug info.')
    adv.add_argument('--denoise', action='store_true', default=False,
                     help='Enable denoising.')
    adv.add_argument('--is-wood', default=0.5, type=float,
                     help='Point-level probability threshold for argmax aggregation (default 0.5).')
    adv.add_argument('--hysteresis', type=float, nargs=2, default=None, metavar=('HIGH', 'LOW'),
                     help='Seeded-BFS hysteresis: HIGH anchors wood skeleton, LOW promotes connected candidates.')
    adv.add_argument('--min-votes', type=int, default=1,
                     help='With --any-wood: passes that must exceed threshold to classify as wood (default 1).')
    adv.add_argument('--grid-method', type=str, default='max', choices=['mean', 'max'],
                     help="Pre-downsampling voxel representative: 'max' keeps highest-reflectance point (default).")
    adv.add_argument('--collect-grid-size', type=float, default=None,
                     help='Post-aggregation voxel size (m). Auto = max(0.04, 2x spacing).')
    adv.add_argument('--sor', action='store_true',
                     help='Statistical outlier removal before subsampling (k=10, 1 std).')
    adv.add_argument('--sor-k', type=int, default=10, help='SOR neighbours (default 10).')
    adv.add_argument('--sor-std', type=float, default=1.0, help='SOR threshold multiplier (default 1.0).')
    adv.add_argument('--no-refl', action='store_true', default=False,
                     help='Zero reflectance channel (geometry-only mode).')
    adv.add_argument('--low-memory', action='store_true', default=False,
                     help='Write voxels to disk on demand (slower but lower RAM for very large clouds).')

    args = parser.parse_args()

    # Resolve model from --region if --model not explicitly provided
    if args.model is None:
        region = args.region or 'eu'
        if region == 'eu':
            args.model = 'h4mcc-eu.pth'
        else:
            args.model = f'h4mcc-{region}.pth'

    if args.thorough:
        args.inference_level = 4
        args.tta = 4
    grid_size_user_provided = args.grid_size is not None
    collect_grid_user_provided = args.collect_grid_size is not None
    resolution_user_provided = args.resolution is not None and float(args.resolution) > 0.0

    # Derive in_memory from low_memory flag (in_memory is the default fast path)
    args.in_memory = not args.low_memory

    configure_threads(args.num_procs)

    # Warm up CUDA early so first preprocessing message isn't delayed by context init
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    args.wdir = get_path()
    args.mode = 'predict' if 'predict' in sys.argv[0] else 'train'
    args.reflectance = False
    
    total_pre_time = 0.0
    total_inf_time = 0.0
    peak_gpu_bytes_overall = 0
    all_preprocessing_stats = []
    all_inference_stats = []
    wood_stats = []

    for point_cloud_file in args.point_cloud:
        if not os.path.isfile(point_cloud_file):
            raise FileNotFoundError(f'Point cloud file not found: {point_cloud_file}')

    _print_heading("Model")
    _inference_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _inference_model = load_inference_model(args, _inference_device)
    
    
    path = OP.dirname(args.point_cloud[0])
    args.vxfile = OP.join(path, "voxels")

    if not args.in_memory and os.path.exists(args.vxfile):
        shutil.rmtree(args.vxfile)

    for point_cloud_file in args.point_cloud:

        
        path = OP.dirname(point_cloud_file)
        file = OP.splitext(OP.basename(point_cloud_file))[0] + "-ptw.ply"
        args.odir = OP.join(path, file)

        if os.path.exists(args.odir):
            try:
                os.remove(args.odir)
            except Exception as e:
                print(f"Warning: could not delete existing output {args.odir}: {e}")

        if not args.in_memory:
            os.makedirs(args.vxfile, exist_ok=True)
        args.pc, args.headers = load_file(filename=point_cloud_file, additional_headers=True, verbose=False)
        args.pc, args.headers, args.reflectance = preprocess_point_cloud_data(args.pc, zero_reflectance=args.no_refl)
        args.inference_voxels = None
        refl_col = next((c for c in args.headers if 'reflectance' in c.lower()), None)
        _print_heading(f"Input: {OP.basename(point_cloud_file)}")
        _print_item("File", point_cloud_file)
        _print_item("Points", f"{len(args.pc):,}")
        refl_msg = _fmt_yes_no(args.reflectance)
        if refl_col:
            refl_msg += f" (column: {refl_col})"
        if args.no_refl:
            refl_msg += " | overridden to zeros"
        _print_item("Reflectance", refl_msg)

        auto_meta = None
        if not grid_size_user_provided:
            xyz_np = args.pc[['x', 'y', 'z']].to_numpy(dtype=np.float32, copy=False)
            args.grid_size, auto_meta = _auto_grid_size_from_xyz(xyz_np, min_grid=1.0)

        # --inference-level sets grid scales and overlap (unless user explicitly provided --grid-size)
        inference_level = getattr(args, 'inference_level', 2)
        if not grid_size_user_provided and inference_level >= 2:
            base = args.grid_size[0]
            args.grid_size = [base, base * 2.0]
        overlap_offsets = {1: 0, 2: 0, 3: 8, 4: 8}[inference_level]
        args.overlap = overlap_offsets
        inference_labels = {
            1: "single scale",
            2: "standard (2-scale, TTA2, no overlap)",
            3: "thorough (2-scale, 8 overlaps, TTA4)",
            4: "thorough + dual-perspective",
        }
        args.dual_perspective = bool(inference_level >= 4 and not args.no_refl)

        if not collect_grid_user_provided:
            auto_collect, _ = _auto_collect_grid_size_m(args.resolution, args.grid_size)
            args.collect_grid_size = auto_collect
        _print_item("Inference", f"level {inference_level} ({inference_labels[inference_level]})")
        _print_item("Point res.", _format_point_resolution_choice(args.resolution, resolution_user_provided, auto_meta, args.grid_size))
        _print_item("Grid size", _format_grid_choice(args.grid_size, grid_size_user_provided, auto_meta))
        _print_item("Overlap", f"{overlap_offsets} offsets per scale")
        _print_item("Collect res.", _format_collect_choice(args.collect_grid_size, collect_grid_user_provided))
        _print_item("Perspective", "dual (refl + no-refl)" if args.dual_perspective else ("geometry only" if args.no_refl else "reflectance"))
        _print_item("TTA", "off" if int(args.tta) <= 1 else f"{int(args.tta)}× z-rotations")
        _print_item("Voxel cache", "memory" if args.in_memory else f"disk/low-memory ({args.vxfile})")

        _print_heading("Preprocessing")

        if args.verbose:
            point_count = len(args.pc)
            memory_info = format_memory_info(point_count, args.reflectance, args.grid_size, args.resolution)
            _print_item("Estimate", memory_info)

        # Track preprocessing performance - don't reset GPU stats to capture true peak
        preprocessing_tracker = PerformanceTracker("Preprocessing", reset_gpu_stats=False)
        preprocess_result = preprocess(args)
        if args.in_memory:
            args.inference_voxels = preprocess_result
            if args.verbose:
                _print_item("Voxel count", f"{len(args.inference_voxels):,} kept in RAM")
        preprocessing_stats = preprocessing_tracker.finish()
        all_preprocessing_stats.append(preprocessing_stats)
        total_pre_time += preprocessing_stats['duration']
        
        _print_heading("Inference")

        inference_tracker = PerformanceTracker("Inference")
        SemanticSegmentation(args, model=_inference_model)
        inference_stats = inference_tracker.finish()
        all_inference_stats.append(inference_stats)
        total_inf_time += inference_stats['duration']
        if 'prediction' in args.pc.columns:
            pred = args.pc['prediction'].to_numpy(copy=False)
            wood_pct = 100.0 * float(np.mean(pred >= 0.5)) if len(pred) else 0.0
            wood_stats.append((OP.basename(point_cloud_file), wood_pct))

        peak_gpu_bytes_overall = max(peak_gpu_bytes_overall, inference_stats['gpu_peak_used'] * (1024**3))

        args.inference_voxels = None
        if not args.in_memory and os.path.exists(args.vxfile):
            shutil.rmtree(args.vxfile)

    # Final allocator purge once all clouds are processed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Calculate aggregated stats
    total_preprocessing_cpu_peak = sum(stats['cpu_peak_increase'] for stats in all_preprocessing_stats)
    total_preprocessing_gpu = max((stats['gpu_peak_used'] for stats in all_preprocessing_stats), default=0)
    total_inference_cpu_peak = sum(stats['cpu_peak_increase'] for stats in all_inference_stats)
    total_inference_gpu = max((stats['gpu_peak_used'] for stats in all_inference_stats), default=0)

    # Overall peak tracking
    final_cpu_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / (1024**3)
    initial_cpu_peak = 0.6  # Approximate baseline before processing starts

    _print_heading("Performance Summary")
    _print_item("Preprocess", f"{total_pre_time:.3f} s | CPU peak +{total_preprocessing_cpu_peak:.3f} GB | GPU peak {total_preprocessing_gpu:.3f} GB")
    _print_item("Inference", f"{total_inf_time:.3f} s | CPU peak +{total_inference_cpu_peak:.3f} GB | GPU peak {total_inference_gpu:.3f} GB")
    _print_item("Total", f"{total_pre_time + total_inf_time:.3f} s | CPU peak {final_cpu_peak:.3f} GB | GPU peak {peak_gpu_bytes_overall / (1024**3):.3f} GB")
    if len(wood_stats) == 1:
        _print_item("Wood", f"{_green(f'{wood_stats[0][1]:.1f}%')} of input points")
    elif wood_stats:
        joined = " | ".join(f"{name} {_green(f'{pct:.1f}%')}" for name, pct in wood_stats)
        _print_item("Wood", joined)
    if wood_stats:
        print()
