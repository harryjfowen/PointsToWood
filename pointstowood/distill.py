import datetime
start = datetime.datetime.now()
import resource

import argparse, glob, os
import numpy as np
import shutil
from src.distillation import SemanticDistillation
from src.preprocessing import *
from src.io import load_file
import sys
import re


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

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

def preprocess_point_cloud_data(df):
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
    return df

if __name__ == "__main__":
    print('\n\n=== PointsToWood DISTILLATION ===\n')
    print(f'Using PyTorch device: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0:.2f} GB RAM')

    if len(sys.argv) == 1:
        print("No arguments provided. Use --help for usage.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Distill biome-specific models from EU teacher')

    # Core arguments (matching train.py)
    parser.add_argument('--device', type=str, default='cuda', help='Insert either "cuda" or "cpu"')
    parser.add_argument('--region', type=str, default='spain', help='Data region (e.g. spain, germany, poland)')
    parser.add_argument('--model', type=str, default=None, help='Student model name')
    parser.add_argument('--teacher-model', type=str, default='fbeta-eu.pth', help='Teacher model path (default: fbeta-eu.pth)')

    # Data preprocessing (matching train.py)
    parser.add_argument('--resolution', type=float, default=0.02, help='Resolution to which point cloud is downsampled [m]')
    parser.add_argument('--grid-size', type=float, nargs='+', default=[2.0], help='Grid sizes for voxelization')
    parser.add_argument('--min-pts', type=int, default=512, help='Minimum number of points in voxel')
    parser.add_argument('--max-pts', type=int, default=16384, help='Maximum number of points in voxel')
    parser.add_argument('--grid-method', type=str, default='max', choices=['mean', 'max'], help="Voxel representative method")
    parser.add_argument('--collect-grid-size', type=float, default=0.0, help='Optional post-aggregation voxel size (m)')
    parser.add_argument('--preprocess', action='store_true', help="Preprocess point clouds into voxels")

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-epochs', default=60, type=int, help='Number of training epochs (default: 60 for distillation)')
    parser.add_argument('--max-lr', type=float, default=1e-3, help='Maximum learning rate for OneCycleLR (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay for optimizer (default: 1e-2)')
    parser.add_argument('--augmentation', action='store_true', default=True, help='Enable data augmentation')
    parser.add_argument('--test', action='store_true', default=True, help='Run testing after training')

    # Distillation parameters
    parser.add_argument('--alpha', type=float, default=0.8, help='Distillation loss weight (default: 0.8)')
    parser.add_argument('--temperature', type=float, default=4.0, help='Distillation temperature (default: 4.0)')

    # Augmentation options
    parser.add_argument('--pointcutmix', action='store_true', default=False, help='Enable PointCutMix data augmentation (spatial method)')
    parser.add_argument('--pointcutmix-prob', type=float, default=0.25, help='Probability of applying PointCutMix to each batch')
    parser.add_argument('--pointcutmix-beta', type=float, default=1.0, help='Beta parameter for PointCutMix mixing ratio')

    # Other parameters (matching train.py)
    parser.add_argument('--balance-mode', dest='balance_mode', type=str, default='downsampling', choices=['downsampling', 'upsampling'], help='Class balancing mode')
    parser.add_argument('--wandb', action='store_true', help="Use wandb for logging")
    parser.add_argument('--verbose', action='store_true', help="Print detailed information")

    args = parser.parse_args()
    args.wdir = get_path()
    args.mode = 'distill'

    if args.model is None:
        args.model = f'fbeta-{args.region}.pth'
        print(f"No model name provided. Using: {args.model}")

    # Set up data paths (matching train.py)
    args.train_dir = os.path.join(args.wdir, f'data/{args.region}_train')
    args.test_dir = os.path.join(args.wdir, f'data/{args.region}_test')
    args.trfile = os.path.join(args.train_dir, "voxels")
    args.tefile = os.path.join(args.test_dir, "voxels")

    train_files = glob.glob(os.path.join(args.train_dir, '*.ply'))
    test_files = glob.glob(os.path.join(args.test_dir, '*.ply'))

    if args.preprocess:
        if os.path.exists(args.trfile):
            shutil.rmtree(args.trfile)

        if args.verbose:
            print('\n----- Preprocessing started -----')

        for i, p in enumerate(train_files):
            os.makedirs(args.trfile, exist_ok=True)
            args.pc, args.headers = load_file(filename=p, additional_headers=True, verbose=True)
            args.pc = preprocess_point_cloud_data(args.pc)
            args.vxfile = args.trfile

            if args.verbose:
                print(f'Voxelising to {args.grid_size} grid sizes')
            preprocess(args)

        if args.test:
            if os.path.exists(args.tefile):
                shutil.rmtree(args.tefile)

            for i, p in enumerate(test_files):
                if args.verbose:
                    print(f'Processing test file {i+1}/{len(test_files)}: {p}')

                os.makedirs(args.tefile, exist_ok=True)
                args.pc, args.headers = load_file(filename=p, additional_headers=True, verbose=True)
                args.pc = preprocess_point_cloud_data(args.pc)
                args.vxfile = args.tefile
                preprocess(args)

        if args.verbose:
            print('----- Preprocessing completed -----\n')

    # Check data exists
    if not os.path.exists(args.trfile):
        raise ValueError(f'Training data not found at {args.trfile}. Run with --preprocess first.')

    if args.test and not os.path.exists(args.tefile):
        raise ValueError(f'Test data not found at {args.tefile}. Run with --preprocess first.')

    print(f'Training data: {args.trfile}')
    if args.test:
        print(f'Test data: {args.tefile}')

    # Check teacher model exists
    teacher_path = os.path.join(args.wdir, 'model', args.teacher_model)
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f'Teacher model not found: {teacher_path}')

    print(f'Using teacher model: {args.teacher_model}')
    print(f'Student model will be saved as: {args.model}')

    # Start distillation training
    SemanticDistillation(args)

    elapsed_time = datetime.datetime.now() - start
    print('\n\n=== DISTILLATION COMPLETED ===')
    print(f'Total time: {elapsed_time}')
    print(f'RAM usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0:.2f} GB')