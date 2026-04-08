import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import datetime
start = datetime.datetime.now()
import resource

import argparse, glob, os, random
import numpy as np
import shutil
from src.trainer import SemanticTraining
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

if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        

        parser.add_argument('--device', type=str, default='cuda', help='Insert either "cuda" or "cpu"')
        parser.add_argument('--region', type=str, default='eu', help='Data region (e.g. eu, spain, germany, global)')
        parser.add_argument('--num-procs', type=int, default=1, help='Number of cpu cores to use')
        parser.add_argument('--num-epochs', default=180, type=int, metavar='N', help='number of total epochs to run')
        parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default 1e-3). Used as max_lr in OneCycleLR.')
        parser.add_argument('--checkpoint-saves', default=1, type=int, metavar='N', help='number of times to save model')
        parser.add_argument('--model', type=str, default=None, help='Name of global model [e.g. model.pth]')
        parser.add_argument('--resolution', type=float, default=0.0,
                            help='Pre-voxel downsampling spacing [m]. Default 0 (adaptive: spacing=grid_size/100). Set e.g. 0.02 for fixed 2cm.')
        parser.add_argument('--grid-size', type=float, nargs='+', default=[2.0, 4.0], help='Grid sizes for voxelization (default: 2.0m 4.0m)')
        parser.add_argument('--overlap', type=int, default=0, choices=[0, 4, 8],
                            help='Overlapping grid offsets for training preprocessing. 0=disabled (multi-resolution), 4=50%% overlap, 8=denser coverage. Recommended: 4 or 8 with single grid-size.')
        parser.add_argument('--min-pts', type=int, default=2048, help='Minimum number of points in voxel (default: 2048 for quality samples)')
        parser.add_argument('--max-pts', type=int, default=32768,
                            help='Max points per voxel; random subsample if exceeded. 0 = no subsampling (use full voxel). Use e.g. 32768 for subsampling.')
        parser.add_argument('--batch-size', type=int, default=4, help='Batch size for cuda processing [Lower less memory usage]')
        parser.add_argument('--target-points-per-batch', type=int, default=16384, help='Max points per batch - never exceeds largest sample size')
        parser.add_argument('--grid-method', type=str, default='max', choices=['mean', 'max'],
                            help="Voxel representative method: 'max' (select point with max reflectance) or 'mean' (mean xyz/refl)")
        parser.add_argument('--collect-grid-size', type=float, default=0.0,
                            help='Optional post-aggregation voxel size (m). If >0, assign labels per voxel during training')
        parser.add_argument('--augmentation', action='store_true', default=True, help="Perform data augmentation")
        parser.add_argument('--preprocess', action='store_true', help="Preprocess point clouds into voxels")
        parser.add_argument('--test', action='store_true', default=True, help="Perform model testing during training")
        parser.add_argument('--tune', action='store_true', help="Tune model hyperparameters with lower learning rate schedule")
        parser.add_argument('--early_stop', action='store_true', help='Enable early stopping on validation harmonic Fbeta')
        parser.add_argument('--wandb', action='store_true', default=True, help="Use wandb for logging")
        parser.add_argument('--verbose', action='store_true', default=True, help="print stuff")
        parser.add_argument('--balance-mode', dest='balance_mode', type=str, default='downsampling', choices=['downsampling', 'upsampling'], help='Class balancing mode: downsampling or upsampling')
        parser.add_argument('--pointcutmix', dest='pointcutmix', action='store_true', default=False, help='Enable PointCutMix (insert: wood inside leaf). Default: off; use --pointcutmix to enable.')
        parser.add_argument('--no-pointcutmix', dest='pointcutmix', action='store_false', help='Disable PointCutMix (default)')
        parser.add_argument('--dualnorm-lite', action='store_true', default=True,
                            help='Enable lightweight dual normalization in AnisotropicConv message features.')
        parser.add_argument('--spatial-mix-lite', action='store_true', default=True,
                            help='Enable lightweight spatial mixing inside SA2 residual blocks (reuse local graph; low-memory).')
        parser.add_argument('--learnable-kernels', action='store_true', help='Use learnable kernel directions in AnisotropicConv (NetFull only; NetLight uses True by default)')
        parser.add_argument('--num-kernel-points', type=int, default=16,
                            help='Kernel points for NetFull anisotropic convs (default 16; lower saves memory, 32 is heavier).')
        parser.add_argument('--max-points-per-batch', type=int, default=32000,
                            help='Target max points per batch. Enables point-budget-aware batching instead of fixed batch size. '
                                 'Set to 0 to use fixed --batch-size instead. Recommended: 40000-60000 for 24GB GPU.')
        parser.add_argument('--min-points-per-batch', type=int, default=16000,
                            help='Avoid batches smaller than this (e.g., single 4K voxel). Packs small samples with next even if over target; collate downsamples.')
        parser.add_argument('--packing-mode', type=str, default='balanced_bfd', choices=['ffd', 'bfd', 'balanced', 'balanced_bfd'],
                            help='Batch packing: bfd=Best-Fit Decreasing (flattest), ffd=First-Fit, balanced=class round-robin, balanced_bfd=class-aware BFD + smoothing (default: balanced_bfd)')
        parser.add_argument('--edge-weight-loss', type=float, default=0.0,
                            help='Max edge weighting in focal loss (0=off). Ramps in after 10%% of training.')
        parser.add_argument('--gamma-max', type=float, default=1.0,
                            help='Focal loss max gamma (0=plain BCE, no focal weighting). Cyclical: 0→gamma_max→0 over training.')
        parser.add_argument('--gamma-peak-pct', type=float, default=0.5,
                            help='Fraction of training where cyclical focal gamma reaches gamma_max (default 0.5 = mid-training peak).')
        parser.add_argument('--cbl-weight', type=float, default=0.25,
                            help='Weight for Contrastive Boundary Loss (default 0.25). Lower = main segmentation loss dominates more.')
        parser.add_argument('--no-cbl-ramp', dest='cbl_ramp', action='store_false', default=True,
                            help='Disable CBL warmup ramp (use full CBL weight from epoch 1). Default: enabled.')
        parser.add_argument('--cbl-ramp-pct', type=float, default=0.33,
                            help='Fraction of training used to ramp CBL weight from 0 to full (default 0.33, gamma-style).')
        parser.add_argument('--focal-alpha', type=float, default=None,
                            help='Focal loss class weight for wood (positive class). <0.5 penalises leaf errors more → lower FPR. Default None (no weighting; class imbalance handles this naturally).')
        parser.add_argument('--label-smoothing', type=float, default=0.1,
                            help='Focal/BCE label smoothing (0=none, 0.1=default).')
        parser.add_argument('--refl-fp-penalty', type=float, default=0.0,
                            help='Weight for FP penalty on leaf points (0=off). Penalizes leaf predicted as wood. Try 0.25. Flat by default (all leaf points).')
        parser.add_argument('--refl-fp-weighted', dest='refl_fp_flat', action='store_false', default=True,
                            help='Weight FP penalty by reflectance (higher penalty on high-refl leaf). Default: flat (all leaf points).')
        parser.add_argument('--no-refl-fp-ramp', dest='refl_fp_ramp', action='store_false', default=True,
                            help='Disable FP penalty ramp (use full weight from epoch 1). Default: ramp 0→1 over training.')
        parser.add_argument('--eval', nargs='?', default=True, const=True, metavar='PLY',
                            help='Eval visualization: optional PLY filename in data/eval/ filtered by region prefix (if omitted, use all matching files)')
        parser.add_argument('--no-eval', dest='eval', action='store_const', const=None,
                            help='Disable eval visualization entirely.')
        parser.add_argument('--eval-grid-size', type=float, default=2.0,
                            help='Grid size (m) used only for eval visualization voxelization (default 2.0).')
        parser.add_argument('--sor', action='store_true', help='Per-voxel SOR before subsampling when writing voxels (k=10, 1 std)')
        parser.add_argument('--sor-k', type=int, default=10, help='SOR neighbours per voxel (default 10)')
        parser.add_argument('--sor-std', type=float, default=1.0, help='SOR threshold = mean + sor_std*std (default 1.0)')
        parser.add_argument('--accumulation-steps', type=int, default=4, help='Gradient accumulation steps (default 4). Effective batch = batches × this.')
        parser.add_argument('--drop-path-rate', type=float, default=0.0, help='KPConvX-style stochastic depth on SA2 (default 0). Try 0.05-0.1 if overfitting.')
        parser.add_argument('--ema', action='store_true', default=True, help='Use EMA model weights for validation')
        parser.add_argument('--ema-decay', type=float, default=0.99, help='EMA decay per step when --ema is set (default 0.99 ~100 steps)')
        parser.add_argument('--amp-dtype', type=str, default='auto', choices=['auto', 'fp16', 'bf16'],
                            help='AMP precision: auto (prefer bf16 if supported), fp16, or bf16 (default auto).')
        parser.add_argument('--no-stem', action='store_true', help='Deprecated no-op: stem has been removed; SA1 always starts directly from pos+reflectance.')
        parser.add_argument('--density-aug', action='store_true', default=True, help='Random voxel-grid downsampling (1-4 cm by default) at train time for density robustness')
        parser.add_argument('--density-aug-prob', type=float, default=0.5, help='Probability of applying density downsampling per sample (default 0.5)')
        parser.add_argument('--density-aug-spacing', type=float, nargs=2, default=[0.01, 0.04], metavar=('MIN', 'MAX'), help='Density aug spacing range in m (default 0.01 0.04 = 1-4 cm, continuous random draw; per-voxel rep = max reflectance)')
        parser.add_argument('--epoch-steps', type=int, default=600, help='Max training steps per epoch (0 = use all data)')
        parser.add_argument('--val-steps', type=int, default=200, help='Max validation steps per epoch (0 = use all data)')
        parser.add_argument('--group-dro', action='store_true', default=False,
                            help='Enable GroupDRO: biome-aware loss reweighting. Upweights underperforming groups (fin, spa, cam etc). Implies gamma=0.')
        parser.add_argument('--group-dro-eta', type=float, default=0.01,
                            help='GroupDRO weight update step size (default 0.01). Higher = faster reweighting but less stable.')
        args = parser.parse_args()
        args.wdir = get_path()
        args.mode = 'predict' if 'predict' in sys.argv[0] else 'train'

        if args.model is None:
            args.model = f"{args.region}.pth"
        
        if args.verbose: print('Mode: {}'.format(args.mode))

        args.checkpoints = np.arange(0, args.num_epochs+1, int(args.num_epochs / args.checkpoint_saves))

        old_checkpoints = glob.glob(os.path.join(args.wdir,'checkpoints/*.pth'))
        if len(old_checkpoints) > 0:
                shutil.make_archive(os.path.join(args.wdir,'checkpoints_backup'), 'zip', os.path.join(args.wdir,'checkpoints'))
        for f in old_checkpoints:
                os.remove(f)

        # Global pool: all raw .ply files live in data/train/, data/test/, data/eval/
        # Voxels are per-region: data/train/voxels_{region}/ so regions never mix.
        # --region eu/global → preprocess all files; any other region → filter by prefix.
        _data_root = os.path.join(args.wdir, 'data')
        args.train_dir = os.path.join(_data_root, 'train')
        args.test_dir  = os.path.join(_data_root, 'test')
        args.eval_dir  = os.path.join(_data_root, 'eval')

        args.trfile = os.path.join(args.train_dir, f'voxels_{args.region}')
        args.tefile = os.path.join(args.test_dir,  f'voxels_{args.region}')

        from src.regions import filter_ply_files, get_prefixes
        _prefixes = get_prefixes(args.region)

        def _collect_ply(directory):
            return filter_ply_files(sorted(glob.glob(os.path.join(directory, '*.ply'))), args.region)

        train_files = _collect_ply(args.train_dir)
        test_files  = _collect_ply(args.test_dir)

        if args.verbose:
            _pstr = 'all' if _prefixes is None else ', '.join(_prefixes)
            print(f'Region: {args.region} | prefixes: {_pstr}')
            print(f'Train files: {len(train_files)} | Test files: {len(test_files)}')

        # Eval visualization: .ply files in data/eval/, filtered by region
        if args.eval is not None:
            eval_files = _collect_ply(args.eval_dir)
            if not eval_files:
                raise ValueError(f'--eval requires at least one .ply in {args.eval_dir} matching region "{args.region}"')
            if isinstance(args.eval, str) and args.eval.strip():
                name = os.path.basename(args.eval)
                candidate = os.path.join(args.eval_dir, name)
                if os.path.isfile(candidate):
                    args.eval_files = [candidate]
                elif os.path.isfile(args.eval):
                    args.eval_files = [args.eval]
                else:
                    raise FileNotFoundError(f'--eval file not found: {args.eval} (nor in {args.eval_dir})')
            else:
                args.eval_files = sorted(eval_files)
            args.eval_vxfile = os.path.join(args.eval_dir, f'eval_voxels_{args.region}')
            args.eval_collect_grid_size = 0.04
            args.eval = True

        if args.preprocess:

                if os.path.exists(args.trfile): shutil.rmtree(args.trfile)

                if args.verbose: print('\n----- Preprocessing started -----')

                for i, p in enumerate(train_files):
                        
                        os.makedirs(args.trfile, exist_ok=True)
                        args.pc, args.headers = load_file(filename=p, additional_headers=True, verbose=True)
                        args.pc, _, args.reflectance = preprocess_point_cloud_data(args.pc)
                        args.vxfile = args.trfile
                        
                        if args.verbose: print(f'Voxelising to {args.grid_size} grid sizes')
                        preprocess(args)

                if args.test:

                        if os.path.exists(args.tefile): shutil.rmtree(args.tefile)
                        
                        if args.verbose: print("\nTesting")

                        args.mode = 'test'

                        for i, p in enumerate(test_files):

                                os.makedirs(args.tefile, exist_ok=True)
                                args.pc, args.headers = load_file(filename=p, additional_headers=True, verbose=True)
                                args.pc, _, args.reflectance = preprocess_point_cloud_data(args.pc)
                                args.vxfile = args.tefile

                                if args.verbose: print(f'Voxelising to {args.grid_size} grid sizes')
                                preprocess(args)

                if getattr(args, 'eval', False):
                        if os.path.exists(args.eval_vxfile):
                                shutil.rmtree(args.eval_vxfile)
                        if args.verbose:
                                print(f'\nEval preprocessing: {len(args.eval_files)} file(s)')
                                print(f'  Eval voxel grid size: {args.eval_grid_size}m')
                        os.makedirs(args.eval_vxfile, exist_ok=True)
                        original_grid_sizes = args.grid_size
                        args.grid_size = [float(args.eval_grid_size)]
                        for eval_file in args.eval_files:
                                if args.verbose:
                                        print(f'  Processing: {os.path.basename(eval_file)}')
                                args.pc, args.headers = load_file(filename=eval_file, additional_headers=True, verbose=True)
                                args.pc, _, args.reflectance = preprocess_point_cloud_data(args.pc)
                                if 'label' not in args.pc.columns:
                                        args.pc['label'] = np.zeros(len(args.pc), dtype=np.int64)
                                # Store source filename in voxel metadata for separate output
                                args.eval_source_file = os.path.splitext(os.path.basename(eval_file))[0]
                                args.vxfile = args.eval_vxfile
                                if args.verbose:
                                        print(f'  Voxelising to {args.grid_size} grid sizes')
                                preprocess(args)
                        args.grid_size = original_grid_sizes

                if args.verbose:
                        print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
                        print(f'runtime: {(datetime.datetime.now() - start).seconds}')

        if args.augmentation:
                if args.verbose: print('Training with data augmentation')


        if len(args.checkpoints) == 0:
                args.checkpoints = np.asarray([args.num_epochs-1])
        if args.verbose: print('\n----- Semantic segmenation started -----')

        # Clear eval visualisations folder at training start so we don't accumulate files
        if getattr(args, 'eval', False):
            vis_dir = os.path.join(args.eval_dir, f'visualisations_{args.region}')
            if os.path.isdir(vis_dir):
                shutil.rmtree(vis_dir)
            os.makedirs(vis_dir, exist_ok=True)

        SemanticTraining(args)

        if args.verbose:
                print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
                print(f'runtime: {(datetime.datetime.now() - start).seconds}')
