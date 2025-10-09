import datetime
start = datetime.datetime.now()
import time
import resource
import os
import os.path as OP
import argparse
from src.preprocessing import preprocess
from src.predicter import SemanticSegmentation
import torch
import shutil
import sys
import numpy as np
import re
from src.io import load_file
from src.utils import configure_threads
import psutil
import gc

class PerformanceTracker:
    def __init__(self, process_name):
        self.process_name = process_name
        self.process = psutil.Process(os.getpid())
        self.start_time = time.perf_counter()

        # Track both resource (peak) and current RSS at start
        self.start_rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / (1024**3)
        self.start_rss_current = self.process.memory_info().rss / (1024**3)

        self.start_gpu = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_gpu = torch.cuda.memory_allocated() / (1024**3)

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
        gpu_peak_used = gpu_peak - self.start_gpu

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

def preprocess_point_cloud_data(df, zero_reflectance=False):
    canon_map = {
        'label': ['label'],
        'reflectance': ['reflectance', 'refl', 'intensity'],
    }

    new_columns = {}
    for col in df.columns:
        clean = col.lower().replace('scalar_', '') 
        mapped = None
        for target, aliases in canon_map.items():
            if any(alias in clean for alias in aliases):
                mapped = target
                break
        new_columns[col] = mapped if mapped is not None else clean

    df = df.rename(columns=new_columns)

    if 'truth' in df.columns and 'label' in df.columns:
        df = df.drop(columns=['label'])

    df = df.loc[:, ~df.columns.duplicated()]

    drop_tokens = ["prediction", "pwood"]
    cols_to_drop = [c for c in df.columns if any(tok in c for tok in drop_tokens)]
    if len(cols_to_drop):
        df = df.drop(columns=cols_to_drop, errors='ignore')

    if 'reflectance' not in df.columns:
        df['reflectance'] = np.zeros(len(df))
        print('No reflectance detected, column added with zeros.')
    else:
        print('Reflectance detected')
    
    if zero_reflectance:
        df['reflectance'] = np.zeros(len(df))
        print('Reflectance set to zeros as requested.')
    
    xyz_cols = ['x', 'y', 'z']
    required_order = xyz_cols + ['reflectance']
    other_cols = [col for col in df.columns if col not in required_order]
    final_cols = required_order + other_cols
    df = df[final_cols]

    headers = [c for c in df.columns if c not in xyz_cols]
    return df, headers, ('reflectance' in df.columns)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--point-cloud', '-p', default=[], nargs='+', type=str, help='list of point cloud files')    
    parser.add_argument('--odir', type=str, default='.', help='output directory')
    parser.add_argument('--batch-size', default=0, type=int,
                        help="Mini-batch size. 0=adaptive GPU memory-aware batching (default), >0=fixed sample count per batch")
    parser.add_argument('--num-procs', default=-1, type=int, help="Number of CPU cores you want to use. If you run out of RAM, lower this.")
    parser.add_argument('--resolution', type=float, default=0.0,
                        help='Voxel down-sample resolution [m]. 0 → adaptive per grid (1m→0.01m, 2m→0.02m, 4m→0.04m).')
    parser.add_argument('--grid-size', type=float, nargs='+', default=[1.0, 2.0, 4.0],
                        help='Voxel grid size in metres (default: 1.0 2.0 4.0)')
    parser.add_argument('--min-pts', type=int, default=512,
                        help='Minimum number of points per voxel (default: 512)')
    parser.add_argument('--max-pts', type=int, default=16384, help='Maximum number of points in voxel')
    parser.add_argument('--memory-fraction', type=float, default=0.7,
                        help='Fraction of GPU memory to use for batching (default: 0.7)')
    parser.add_argument('--model', type=str, default='fbeta-eu.pth',
                        help='Model checkpoint name inside pointstowood/model (default: fbeta-eu.pth)')
    parser.add_argument('--output-fmt', default='ply', help="file type of output")
    parser.add_argument('--verbose', action='store_true', help="print stuff")
    parser.add_argument('--boost-perspective', action='store_true', default=False,
                         help="Enable multi-perspective inference with 7 augmented views")
    parser.add_argument('--denoise', action='store_true', default=False,
                        help="Enable denoising")
    parser.add_argument('--is-wood', default=0.5, type=float, help='a probability above which points within KNN are classified as wood')
    parser.add_argument('--any-wood', default=0.5, type=float, help='a probability above which ANY point within KNN is classified as wood')
    parser.add_argument('--hysteresis', action='store_true', default=False, help='Enable hysteresis labeling')
    parser.add_argument('--model-type', type=str, default='optimised', 
                        choices=['optimised', 'harmonic', 'reflectance'],
                        help="Model type: 'optimised' (auto-selects based on reflectance), 'harmonic', or 'reflectance'")
    parser.add_argument('--grid-method', type=str, default='max', choices=['mean', 'max'],
                        help="Voxel representative method: 'mean' (mean xyz/refl) or 'max' (select point with max reflectance)")
    parser.add_argument('--collect-grid-size', type=float, default=0.04,
                        help='Optional post-aggregation voxel size (m). If >0, assign labels per voxel: if any pwood ≥ any-wood threshold within a voxel, all its points are labeled wood.')

    args = parser.parse_args()

    configure_threads(args.num_procs)

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

        os.makedirs(args.vxfile, exist_ok=True)
        args.pc, args.headers = load_file(filename=point_cloud_file, additional_headers=True, verbose=False)
        args.pc, args.headers, args.reflectance = preprocess_point_cloud_data(args.pc, False)

        # Clear any existing GPU memory before tracking
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        def get_model_suffix(has_reflectance, model_type):
            if model_type == 'optimised':
                return '' if has_reflectance else '-xyz'
            elif model_type == 'harmonic':
                return '-harmonic'
            elif model_type == 'reflectance':
                return ''
            
        base_model = args.model.replace('.pth', '')
        for sfx in ['-xyz', '-harmonic']:
            if base_model.endswith(sfx):
                base_model = base_model[: -len(sfx)]
        model_suffix = get_model_suffix(args.reflectance, args.model_type)
        if model_suffix:
            if '-' in base_model:
                prefix, region = base_model.rsplit('-', 1)
                args.model = f"{prefix}{model_suffix}-{region}.pth"
            else:
                args.model = f"{base_model}{model_suffix}.pth"
        else:
            args.model = f"{base_model}.pth"
        
        if args.verbose: print(f'Using model: {args.model} (type: {args.model_type}, reflectance: {args.reflectance})')
        
        if args.verbose: print(f'Voxelising to {args.grid_size} grid sizes')

        # Track preprocessing performance
        preprocessing_tracker = PerformanceTracker("Preprocessing")
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
