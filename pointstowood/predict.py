import datetime
start = datetime.datetime.now()
import time
import resource
import os
import os.path as OP
import argparse
from src.preprocessing import preprocess
from src.predicter import SemanticSegmentation
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
import gc


def _z_stratified_sample_indices(xyz: np.ndarray, sample_size: int = 2048, z_bins: int = 24) -> np.ndarray:
    """Sample indices approximately uniformly over z-quantile bins."""
    n = int(xyz.shape[0])
    if n <= sample_size:
        return np.arange(n, dtype=np.int64)

    z = xyz[:, 2]
    bins = int(max(1, min(z_bins, sample_size)))
    q = np.quantile(z, np.linspace(0.0, 1.0, bins + 1))
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


def _estimate_nn_spacing_m(xyz: np.ndarray, sample_size: int = 2048) -> float:
    """Estimate native point spacing from nearest-neighbor distances.

    Important: query sampled points against a KD-tree built on the full cloud.
    Using sample->sample NN overestimates spacing on large clouds.
    """
    n = int(xyz.shape[0])
    if n < 4:
        return float("nan")

    m = min(sample_size, n)
    sel = _z_stratified_sample_indices(xyz, sample_size=m, z_bins=24)
    sample = xyz[sel].astype(np.float64, copy=False)
    xyz_full = xyz.astype(np.float64, copy=False)

    try:
        from pykdtree.kdtree import KDTree
        tree = KDTree(xyz_full)
        dists, _ = tree.query(sample, k=2)
        nn_d = dists[:, 1]
    except Exception:
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree', n_jobs=1).fit(xyz_full)
            dists, _ = nn.kneighbors(sample)
            nn_d = dists[:, 1]
        except Exception:
            # Last-resort fallback for environments without KD-tree deps.
            t = torch.as_tensor(sample, dtype=torch.float32)
            d = torch.cdist(t, t)
            d.fill_diagonal_(float('inf'))
            nn_d = d.min(dim=1).values.cpu().numpy()

    nn_d = nn_d[np.isfinite(nn_d)]
    if nn_d.size == 0:
        return float("nan")
    return float(np.median(nn_d))


def _auto_grid_size_from_xyz(xyz: np.ndarray, min_grid: float = 2.0):
    """Infer grid size from native spacing: spacing(cm) -> nearest whole-meter grid (min 2m)."""
    spacing_m = _estimate_nn_spacing_m(xyz)
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        return [float(min_grid)], {'spacing_cm': float('nan'), 'rounded_cm': float(min_grid), 'chosen_grid': float(min_grid)}

    spacing_cm = spacing_m * 100.0
    rounded_cm = max(float(min_grid), float(np.round(spacing_cm)))
    chosen_grid = max(float(min_grid), float(np.round(rounded_cm)))
    return [float(chosen_grid)], {'spacing_cm': float(spacing_cm), 'rounded_cm': float(rounded_cm), 'chosen_grid': float(chosen_grid)}


def _auto_collect_grid_size_m(resolution_m: float, grid_sizes_m):
    """Auto-set post-aggregation grid as 2x effective input spacing."""
    if resolution_m is not None and float(resolution_m) > 0.0:
        spacing_m = float(resolution_m)
    else:
        if not grid_sizes_m:
            spacing_m = 0.02
        else:
            spacing_m = max(1e-4, float(min(grid_sizes_m)) / 100.0)

    collect_m = max(0.01, 2.0 * spacing_m)
    collect_m = round(collect_m * 100.0) / 100.0  # nearest cm in meters
    return collect_m, spacing_m


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

def get_path(location_in_pointstowood: str = "") -> str:
    current_wdir = os.getcwd()
    match = re.search(r'PointsToWood.*?pointstowood', current_wdir, re.IGNORECASE)
    if not match:
        raise ValueError('"PointsToWood/pointstowood" not found in the current working directory path')
    last_index = match.end()
    output_path = current_wdir[:last_index]
    if location_in_pointstowood:
        output_path = os.path.join(output_path, location_in_pointstowood)
    return output_path.replace("\\", "/")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--point-cloud', '-p', default=[], nargs='+', type=str, help='list of point cloud files')    
    parser.add_argument('--odir', type=str, default='.', help='output directory')
    parser.add_argument('--batch-size', default=0, type=int,
                        help="Mini-batch size. 0=adaptive GPU memory-aware batching (default), >0=fixed sample count per batch")
    parser.add_argument('--num-procs', default=-1, type=int, help="Number of CPU cores you want to use. If you run out of RAM, lower this.")
    parser.add_argument('--resolution', type=float, default=0.0,
                        help='Pre-voxel downsampling spacing [m]. Default 0 (adaptive). Set e.g. 0.02 for fixed 2cm.')
    parser.add_argument('--grid-size', type=float, nargs='+', default=None,
                        help='Voxel grid size in metres. If omitted, auto-estimate from point spacing and snap to nearest whole meter (min 2m).')
    parser.add_argument('--overlap', type=int, default=8, choices=[0, 2, 4, 8],
                        help='Number of XY grid offsets for edge coverage. 0=multi-resolution, '
                             '2=half-step XY (minimal overlap), 4=corners (50%% overlap), '
                             '8=corners+edges (denser, default for inference).')
    parser.add_argument('--min-pts', type=int, default=512,
                        help='Minimum number of points per voxel (default: 512 for inference coverage).')
    parser.add_argument('--max-pts', type=int, default=16384, help='Maximum number of points in voxel')
    parser.add_argument('--max-points-per-batch', type=int, default=50000,
                        help='Target point budget per inference batch when --batch-size=0 (default: 50000). '
                             'Set to 0 to auto-estimate.')
    parser.add_argument('--memory-fraction', type=float, default=0.7,
                        help='Fraction of GPU memory to use for batching (default: 0.7)')
    parser.add_argument('--model', type=str, default='h4mcc-eu.pth',
                        help='Model checkpoint name inside pointstowood/model (default: h4mcc-eu.pth)')
    parser.add_argument('--output-fmt', default='ply', help="file type of output")
    parser.add_argument('--verbose', action='store_true', help="print stuff")
    parser.add_argument('--boost-perspective', action='store_true', default=False,
                         help="Enable multi-perspective inference with 7 augmented views")
    parser.add_argument('--denoise', action='store_true', default=False,
                        help="Enable denoising")
    parser.add_argument('--is-wood', default=0.5, type=float,
                        help='Probability threshold when using median-based classification (only with --any-wood).')
    parser.add_argument('--any-wood', type=float, default=None, nargs='?', const=0.5,
                        help='If passed: label voxel as wood when any point prob >= this (default 0.5). Omit for argmax |p-0.5| per voxel (default).')
    parser.add_argument('--hysteresis', action='store_true', default=False, help='Enable hysteresis labeling')
    parser.add_argument('--grid-method', type=str, default='max', choices=['mean', 'max'],
                        help="Voxel representative method: 'mean' (mean xyz/refl) or 'max' (select point with max reflectance)")
    parser.add_argument('--collect-grid-size', type=float, default=None,
                        help='Post-aggregation voxel size (m). If omitted, auto = 2x effective spacing (e.g. 2cm->4cm, 4cm->8cm).')
    parser.add_argument('--sor', action='store_true', help='Per-voxel SOR before subsampling when writing voxels (k=10, 1 std)')
    parser.add_argument('--sor-k', type=int, default=10, help='SOR neighbours per voxel (default 10)')
    parser.add_argument('--sor-std', type=float, default=1.0, help='SOR threshold = mean + sor_std*std (default 1.0)')
    parser.add_argument('--learnable-kernels', action='store_true', help='Use learnable kernel directions (must match model checkpoint; NetFull only)')
    parser.add_argument('--dualnorm-lite', dest='dualnorm_lite', action='store_true',
                        help='Force enable DualNorm-lite in AnisotropicConv during inference.')
    parser.add_argument('--no-dualnorm-lite', dest='dualnorm_lite', action='store_false',
                        help='Force disable DualNorm-lite in AnisotropicConv during inference.')
    parser.add_argument('--no-refl', action='store_true', default=False,
                        help='Zero out reflectance channel before inference (geometry-only mode).')
    parser.set_defaults(dualnorm_lite=None)

    args = parser.parse_args()
    grid_size_user_provided = args.grid_size is not None
    collect_grid_user_provided = args.collect_grid_size is not None

    configure_threads(args.num_procs)

    # Warm up CUDA early so first preprocessing message isn't delayed by context init
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    if args.verbose:
        print('\n---- parameters used ----')
        for k, v in args.__dict__.items():
            if k == 'pc': v = '{} points'.format(len(v))
            if k == 'global_shift': v = v.values
            print('{:<35}{}'.format(k, v)) 

    args.wdir = get_path()
    args.mode = 'predict' if 'predict' in sys.argv[0] else 'train'
    args.reflectance = False

    if args.point_cloud == '':
        raise Exception('no input specified, please specify --point-cloud')
    
    total_pre_time = 0.0
    total_inf_time = 0.0
    peak_gpu_bytes_overall = 0
    all_preprocessing_stats = []
    all_inference_stats = []

    for point_cloud_file in args.point_cloud:
        if not os.path.isfile(point_cloud_file):
            raise FileNotFoundError(f'Point cloud file not found: {point_cloud_file}')
    
    
    path = OP.dirname(args.point_cloud[0])
    args.vxfile = OP.join(path, "voxels")

    if os.path.exists(args.vxfile): shutil.rmtree(args.vxfile)

    for point_cloud_file in args.point_cloud:

        
        path = OP.dirname(point_cloud_file)
        file = OP.splitext(OP.basename(point_cloud_file))[0] + "_p2w.ply"
        args.odir = OP.join(path, file)

        if os.path.exists(args.odir):
            try:
                os.remove(args.odir)
            except Exception as e:
                print(f"Warning: could not delete existing output {args.odir}: {e}")

        if args.verbose: print('\n----- Preprocessing started -----')

        print(f"Loading point cloud: {point_cloud_file} ...")
        os.makedirs(args.vxfile, exist_ok=True)
        args.pc, args.headers = load_file(filename=point_cloud_file, additional_headers=True, verbose=False)
        args.pc, args.headers, args.reflectance = preprocess_point_cloud_data(args.pc, zero_reflectance=args.no_refl)
        refl_col = next((c for c in args.headers if 'reflectance' in c.lower()), None)
        print(f"Reflectance detected: {args.reflectance} (column: {refl_col}, overridden to zeros: {args.no_refl})")
        if not grid_size_user_provided:
            xyz_np = args.pc[['x', 'y', 'z']].to_numpy(dtype=np.float32, copy=False)
            args.grid_size, auto_meta = _auto_grid_size_from_xyz(xyz_np, min_grid=2.0)
            print(
                f"Auto grid-size: spacing≈{auto_meta['spacing_cm']:.2f}cm "
                f"-> rounded {auto_meta['rounded_cm']:.0f}cm -> grid {auto_meta['chosen_grid']:.1f}m"
            )
        if not collect_grid_user_provided:
            auto_collect, eff_spacing = _auto_collect_grid_size_m(args.resolution, args.grid_size)
            args.collect_grid_size = auto_collect
            print(
                f"Auto collect-grid-size: spacing≈{eff_spacing * 100.0:.2f}cm -> collect {args.collect_grid_size * 100.0:.0f}cm"
            )
        print("Voxelising ...")

        # Clear any existing GPU memory before tracking
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        if args.verbose: print(f'Using model: {args.model}')
        
        if args.verbose:
            print(f'Voxelising to {args.grid_size} grid sizes')
            # Show memory info for adaptive device selection
            point_count = len(args.pc)
            memory_info = format_memory_info(point_count, args.reflectance, args.grid_size, args.resolution)
            print(memory_info)

        # Multiple grid sizes → use multi-resolution (overlap=0); single grid → use --overlap as set
        if len(args.grid_size) > 1 and getattr(args, 'overlap', 0) != 0:
            args.overlap = 0
            print("Multiple grid sizes: using multi-resolution voxelisation (overlap=0)")

        # Track preprocessing performance - don't reset GPU stats to capture true peak
        preprocessing_tracker = PerformanceTracker("Preprocessing", reset_gpu_stats=False)
        preprocess(args)
        preprocessing_stats = preprocessing_tracker.finish()
        all_preprocessing_stats.append(preprocessing_stats)
        total_pre_time += preprocessing_stats['duration']
        
        if args.verbose: print('\n----- Semantic segmenation started -----')

        # Clear memory and track inference performance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        inference_tracker = PerformanceTracker("Inference")
        SemanticSegmentation(args)
        inference_stats = inference_tracker.finish()
        all_inference_stats.append(inference_stats)
        total_inf_time += inference_stats['duration']

        peak_gpu_bytes_overall = max(peak_gpu_bytes_overall, inference_stats['gpu_peak_used'] * (1024**3))
        torch.cuda.empty_cache()

        if os.path.exists(args.vxfile):
            shutil.rmtree(args.vxfile)

        if args.verbose:
            print(f'CPU peak RSS so far: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / (1024**3):.3f} GiB')

    # Calculate aggregated stats
    total_preprocessing_cpu_peak = sum(stats['cpu_peak_increase'] for stats in all_preprocessing_stats)
    total_preprocessing_gpu = max((stats['gpu_peak_used'] for stats in all_preprocessing_stats), default=0)
    total_inference_cpu_peak = sum(stats['cpu_peak_increase'] for stats in all_inference_stats)
    total_inference_gpu = max((stats['gpu_peak_used'] for stats in all_inference_stats), default=0)

    # Overall peak tracking
    final_cpu_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / (1024**3)
    initial_cpu_peak = 0.6  # Approximate baseline before processing starts

    print('\n' + '='*60)
    print('PERFORMANCE SUMMARY')
    print('='*60)
    print(f'PREPROCESSING:')
    print(f'  Time: {total_pre_time:.3f} seconds')
    print(f'  CPU Memory Peak: {total_preprocessing_cpu_peak:.3f} GB')
    print(f'  GPU Memory Peak: {total_preprocessing_gpu:.3f} GB')
    print()
    print(f'INFERENCE:')
    print(f'  Time: {total_inf_time:.3f} seconds')
    print(f'  CPU Memory Peak: {total_inference_cpu_peak:.3f} GB')
    print(f'  GPU Memory Peak: {total_inference_gpu:.3f} GB')
    print()
    print(f'TOTAL:')
    print(f'  Time: {total_pre_time + total_inf_time:.3f} seconds')
    print(f'  CPU Memory Peak (Overall): {final_cpu_peak:.3f} GB')
    print(f'  GPU Memory Peak (Overall): {peak_gpu_bytes_overall / (1024**3):.3f} GB')
    print('='*60)
