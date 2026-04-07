import datetime
start = datetime.datetime.now()
import resource

import argparse, glob, os
import numpy as np
import shutil
from src.distillation import SemanticDistillation
from src.preprocessing import preprocess
from src.io import load_file
from src.utils import preprocess_point_cloud_data
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

if __name__ == "__main__":
    print('\n\n=== PointsToWood DISTILLATION ===\n')
    print(f'Using PyTorch device: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0:.2f} GB RAM')

    parser = argparse.ArgumentParser(description='Distill biome-specific models from EU teacher')

    # Core arguments (matching train.py)
    parser.add_argument('--device', type=str, default='cuda', help='Insert either "cuda" or "cpu"')
    parser.add_argument('--region', type=str, default='fin', help='Region prefix (e.g. fin, spa, cam, uk). Filters data/train/*.ply by this prefix. Use "eu" or "global" for all files.')
    parser.add_argument('--model', type=str, default=None, help='Student model name')
    parser.add_argument('--teacher-model', type=str, default='mcc-eu.pth', help='Teacher model path (default: mcc-eu.pth)')

    # Data preprocessing (matching train.py)
    parser.add_argument('--resolution', type=float, default=0.0, help='Pre-voxel downsampling spacing [m]. Default 0 (adaptive: spacing=grid_size/100). Set e.g. 0.02 for fixed 2cm.')
    parser.add_argument('--grid-size', type=float, nargs='+', default=[2.0], help='Grid sizes for voxelization (default: 2.0m)')
    parser.add_argument('--min-pts', type=int, default=2048, help='Minimum number of points in voxel')
    parser.add_argument('--max-pts', type=int, default=32768, help='Max points per voxel; random subsample if exceeded. 0 = no subsampling.')
    parser.add_argument('--grid-method', type=str, default='max', choices=['mean', 'max'], help="Voxel representative method")
    parser.add_argument('--collect-grid-size', type=float, default=0.0, help='Optional post-aggregation voxel size (m)')
    parser.add_argument('--preprocess', action='store_true', help="Preprocess point clouds into voxels")
    parser.add_argument('--sor', action='store_true', help='Per-voxel SOR before subsampling when writing voxels (k=10, 1 std)')
    parser.add_argument('--sor-k', type=int, default=10, help='SOR neighbours per voxel (default 10)')
    parser.add_argument('--sor-std', type=float, default=1.0, help='SOR threshold = mean + sor_std*std (default 1.0)')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (fixed-batch path; ignored when point-budget batching is active)')
    parser.add_argument('--max-points-per-batch', type=int, default=32000,
                        help='Target max points per batch. Enables point-budget-aware batching when >0.')
    parser.add_argument('--min-points-per-batch', type=int, default=16000,
                        help='Avoid tiny underutilized batches in point-budget mode (default 16000).')
    parser.add_argument('--packing-mode', type=str, default='balanced_bfd', choices=['ffd', 'bfd', 'balanced', 'balanced_bfd'],
                        help='Batch packing mode in point-budget sampler (default: balanced_bfd).')
    parser.add_argument('--accumulation-steps', type=int, default=4, help='Gradient accumulation steps (default 4).')
    parser.add_argument('--epoch-steps', type=int, default=600, help='Max training steps per epoch (0 = use all batches).')
    parser.add_argument('--val-steps', type=int, default=200, help='Max validation steps per epoch (0 = use all batches).')
    parser.add_argument('--num-epochs', default=90, type=int, help='Number of training epochs (default: 90 for distillation)')
    parser.add_argument('--max-lr', type=float, default=1e-3, help='Maximum learning rate for OneCycleLR (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay for optimizer (default: 1e-2)')
    parser.add_argument('--augmentation', action='store_true', default=True, help='Enable data augmentation')
    parser.add_argument('--test', action='store_true', default=True, help='Run testing after training')

    # Distillation parameters
    parser.add_argument('--alpha', type=float, default=0.5, help='Initial distillation loss weight — starts GT-guided, increases to alpha-final (default: 0.5)')
    parser.add_argument('--alpha-final', type=float, default=0.65, help='Final distillation weight — student leans more on teacher once representations stabilise (default: 0.65)')
    parser.add_argument('--temperature', type=float, default=3.0, help='Initial teacher logit temperature — anneals to temperature-floor over first 50%% of epochs (default: 3.0)')
    parser.add_argument('--temperature-floor', type=float, default=1.5, help='Minimum temperature — prevents raw overconfident teacher logits dominating late training (default: 1.5)')
    parser.add_argument('--paced', action='store_true', default=False,
                        help='PACED frontier-weighted KD: concentrate soft loss on uncertain student points '
                             'w(p)=4p(1-p), suppressed at confident predictions. Run B=bare KD, Run C=KD+PACED.')
    parser.add_argument('--rel-kd-weight', type=float, default=0.0, help='Weight for relation distillation on encoder features (default: 0 — enable with e.g. 0.15)')
    parser.add_argument('--rel-kd-anchors', type=int, default=64, help='Relation KD anchors per sample and stage (default: 64)')
    parser.add_argument('--rel-kd-stages', type=int, default=3, help='Number of encoder stages for relation KD (default: 3)')
    parser.add_argument('--gate-kd-weight', type=float, default=0.0, help='Weight for reflectance gate distillation (default: 0)')
    parser.add_argument('--feat-kd-weight', type=float, default=0.0, help='Weight for feature alignment KD at SA2+SA3 (default: 0)')
    parser.add_argument('--proto-kd-weight', type=float, default=0.0, help='Weight for DINOv2-style prototype distillation at SA2 (default: 0)')
    parser.add_argument('--koleo-weight', type=float, default=0.0, help='Weight for KoLeo feature entropy regularisation (default: 0)')

    # Augmentation options
    parser.add_argument('--pointcutmix', action='store_true', default=False, help='Enable PointCutMix data augmentation (spatial method)')
    parser.add_argument('--pointcutmix-prob', type=float, default=0.25, help='Probability of applying PointCutMix to each batch')
    parser.add_argument('--pointcutmix-beta', type=float, default=1.0, help='Beta parameter for PointCutMix mixing ratio')

    # Other parameters (matching train.py)
    parser.add_argument('--drop-path-rate', type=float, default=0.0, help='KPConvX-style stochastic depth on student SA2 (default 0)')
    parser.add_argument('--learnable-kernels', action='store_true', help='Teacher (NetFull) was trained with learnable kernel directions; must match checkpoint')
    parser.add_argument('--teacher-kernels', type=int, default=16, help='Teacher kernel count (must match checkpoint, default 16)')
    parser.add_argument('--student-c', type=int, default=16, help='Student base channels C (default 16; ~8x smaller than NetFull C=128)')
    parser.add_argument('--student-kernels', type=int, default=16, help='Student kernel count (default 16 — matches teacher geometric resolution; compress via C and blocks instead)')
    parser.add_argument('--student-learnable-kernels', dest='student_learnable_kernels', action='store_true', help='Use learnable kernel directions in student')
    parser.add_argument('--student-fixed-kernels', dest='student_learnable_kernels', action='store_false', help='Use fixed kernel directions in student (default)')
    parser.add_argument('--student-blocks', type=int, nargs=3, default=[1, 2, 1], metavar=('SA1', 'SA2', 'SA3'),
                        help='Residual blocks per SA stage (default 1 2 1). Use 2 3 1 for more capacity at fine/mid scale.')
    parser.add_argument('--dualnorm-lite', action='store_true', default=True, help='Enable DualNorm-lite in anisotropic conv')
    parser.add_argument('--spatial-mix-lite', action='store_true', default=True, help='Enable low-cost spatial mixing in SA2 residual blocks')
    parser.add_argument('--ema', action='store_true', default=False, help='Use EMA model weights for validation')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay per optimizer step when --ema is set (default 0.999)')
    parser.add_argument('--amp-dtype', type=str, default='auto', choices=['auto', 'fp16', 'bf16'],
                        help='AMP precision: auto (prefer bf16 if supported), fp16, or bf16 (default auto).')
    parser.add_argument('--balance-mode', dest='balance_mode', type=str, default='downsampling', choices=['downsampling', 'upsampling'], help='Class balancing mode')
    parser.add_argument('--wandb', action='store_true', default=True, help="Use wandb for logging")
    parser.add_argument('--verbose', action='store_true', default=True, help="Print detailed information")
    parser.add_argument('--check-teacher', action='store_true', default=False,
                        help='Evaluate teacher baseline on test set then exit — use to verify teacher loads correctly before a full run.')
    parser.add_argument('--scratch', action='store_true', default=False,
                        help='Train NetLight purely supervised on biome data — no teacher, no KD losses. '
                             'Critical baseline: same architecture as distilled student, no knowledge transfer. '
                             'Teacher model is still loaded to log its biome performance for comparison.')
    parser.set_defaults(student_learnable_kernels=False)  # fixed kernels by default — matches teacher, preserves flashlight semantics

    args = parser.parse_args()
    args.wdir = get_path()
    args.mode = 'distill'

    if args.model is None:
        args.model = f'mcc-{args.region}.pth'
        print(f"No model name provided. Using: {args.model}")

    # Global pool: all raw .ply files live in data/train/, data/test/
    # Voxels are per-region: data/train/voxels_{region}/ so regions never mix.
    # --region eu/global → preprocess all files; any other region → filter by prefix.
    _data_root = os.path.join(args.wdir, 'data')
    args.train_dir = os.path.join(_data_root, 'train')
    args.test_dir  = os.path.join(_data_root, 'test')
    args.trfile = os.path.join(args.train_dir, f'voxels_{args.region}')
    args.tefile = os.path.join(args.test_dir,  f'voxels_{args.region}')

    from src.regions import filter_ply_files, get_prefixes
    _prefixes = get_prefixes(args.region)

    def _collect_ply(directory):
        return filter_ply_files(sorted(glob.glob(os.path.join(directory, '*.ply'))), args.region)

    train_files = _collect_ply(args.train_dir)
    test_files  = _collect_ply(args.test_dir)

    _pstr = 'all' if _prefixes is None else ', '.join(_prefixes)
    print(f'Region: {args.region} | prefixes: {_pstr}')
    print(f'Train files: {len(train_files)} | Test files: {len(test_files)}')

    if args.preprocess:
        if os.path.exists(args.trfile):
            shutil.rmtree(args.trfile)

        if args.verbose:
            print('\n----- Preprocessing started -----')

        for i, p in enumerate(train_files):
            os.makedirs(args.trfile, exist_ok=True)
            args.pc, args.headers = load_file(filename=p, additional_headers=True, verbose=True)
            args.pc, _, args.reflectance = preprocess_point_cloud_data(args.pc)
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
                args.pc, _, args.reflectance = preprocess_point_cloud_data(args.pc)
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
