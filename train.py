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


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def get_path(location: str = "") -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, location).replace("\\", "/") if location else base

if __name__ == '__main__':

        parser = argparse.ArgumentParser(description='PointsToWood training')

        # ── Core ──────────────────────────────────────────────────────────────
        parser.add_argument('--region', type=str, default='eu',
                            help='Training region: eu, global, or a country prefix (default: eu)')
        parser.add_argument('--num-epochs', type=int, default=300,
                            help='Training epochs (default: 300)')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='Peak learning rate for OneCycleLR (default: 1e-3)')
        parser.add_argument('--model', type=str, default=None,
                            help='Output model filename (default: <region>.pth)')

        # ── Data / memory ─────────────────────────────────────────────────────
        parser.add_argument('--grid-size', type=float, nargs='+', default=[1.0, 2.0],
                            help='Voxel grid sizes in metres (default: 1.0 2.0)')
        parser.add_argument('--max-points-per-batch', type=int, default=32000,
                            help='Point budget per batch — reduce if OOM (default: 32000)')
        parser.add_argument('--accumulation-steps', type=int, default=8,
                            help='Gradient accumulation steps (default: 8)')
        parser.add_argument('--min-points', dest='min_pts', type=int, default=2048,
                            help='Minimum points per voxel (default: 2048)')

        # ── Workflow ──────────────────────────────────────────────────────────
        parser.add_argument('--preprocess', action='store_true',
                            help='Preprocess raw PLY files into voxels before training')
        parser.add_argument('--tune', action='store_true',
                            help='Fine-tune an existing model with a lower LR schedule')
        parser.add_argument('--no-eval', dest='eval', action='store_const', const=None,
                            help='Skip eval visualisation after each epoch')
        parser.add_argument('--wandb', action='store_true', default=False,
                            help='Enable Weights & Biases logging')
        parser.add_argument('--no-augmentation', dest='augmentation', action='store_false', default=True,
                            help='Disable all data augmentation (for debugging)')
        parser.add_argument('--no-difficulty-mining', dest='difficulty_mining', action='store_false', default=True,
                            help='Disable per-voxel EMA difficulty mining and adaptive group sampling')
        parser.add_argument('--overlap', type=int, default=0,
                            help='Additional shifted grid origins when writing training voxels (0=none, max 8). Requires re-preprocessing.')

        # ── Loss / training quality ───────────────────────────────────────────
        parser.add_argument('--refl-fp-penalty', type=float, default=0.40,
                            help='False-positive penalty weight (leaf predicted as wood) (default: 0.40)')
        parser.add_argument('--contrastive-weight', type=float, default=0.1,
                            help='Supervised contrastive loss weight on FP1 features (default: 0.1, 0=off)')
        parser.add_argument('--fine-grid-threshold', type=float, default=2.0,
                            help='Voxels > this size (m) get flat BCE only — no focal gamma or difficulty mining (default: 2.0)')

        args = parser.parse_args()

        args.overlap = max(0, min(8, args.overlap))

        if len(args.num_kernel_points) not in (1, 3):
            parser.error('--num-kernel-points expects either one value or three values for SA1 SA2 SA3')
        args.num_kernel_points = args.num_kernel_points[0] if len(args.num_kernel_points) == 1 else [int(v) for v in args.num_kernel_points]

        # Hard defaults (previously exposed as CLI args)
        args.num_procs = 1
        args.checkpoint_saves = 1
        args.resolution = 0.0
        # args.min_pts is parsed above.
        # Default per-voxel cap to the batch budget so point-budget batches do
        # not need random within-batch cropping for single large voxels.
        if args.max_pts is None:
            args.max_pts = args.max_points_per_batch
        args.batch_size = 4
        args.target_points_per_batch = 16384
        args.grid_method = 'max'
        args.collect_grid_size = 0.0
        # args.augmentation is parsed above
        args.test = True
        args.verbose = True
        args.balance_mode = 'downsampling'
        args.pointcutmix = True
        args.pointcutmix_prob = 0.15
        args.spatial_mix_lite = True
        args.learnable_kernels = False
        args.k_neighbors = 32
        args.compressed_head = True
        args.compressed_head_dim = 128
        args.min_points_per_batch = 16000
        args.packing_mode = 'balanced_bfd'
        args.boundary_weight = 0.0
        args.boundary_ramp_start = 0.1
        args.gamma_max = 4.0
        args.gamma_peak_pct = 0.33
        args.difficulty_sampling = False
        args.difficulty_alpha = 2.0
        args.difficulty_ramp_pct = 0.5
        args.focal_alpha = None
        args.label_smoothing = 0.05
        # Reflectance FP penalty controls are parsed above.
        args.eval_grid_size = 2.0
        args.sor = False
        args.sor_k = 10
        args.sor_std = 1.0
        # args.accumulation_steps is parsed above
        args.drop_path_rate = 0.1
        # args.ema / args.ema_decay are parsed above.
        args.amp_dtype = 'auto'
        # Density augmentation controls are parsed above
        args.val_steps = max(1, args.epoch_steps // 3) if args.epoch_steps > 0 else 0
        # --difficulty-mining enables both biome-level adaptive group sampling
        # and per-voxel boundary-weighted loss EMA hard example mining
        args.adaptive_group_sampling = args.difficulty_mining
        args.adaptive_sampling_warmup = None  # defaults to 10% of epochs in trainer
        args.adaptive_sampling_ramp = None    # defaults to 10% of epochs in trainer
        # adaptive_sampling_alpha/min/max are now proper flags — already on args
        args.per_voxel_difficulty = args.difficulty_mining
        args.per_voxel_alpha = 3.0
        args.per_voxel_warmup_epochs = max(3, int(round(0.10 * args.num_epochs)))
        args.voxel_ema_alpha = 0.9
        args.early_stop = False
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
                        args.source_file_prefix = os.path.splitext(os.path.basename(p))[0]
                        
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
                                args.source_file_prefix = os.path.splitext(os.path.basename(p))[0]

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
                                # Store source filename in voxel metadata for separate output.
                                args.source_file_prefix = os.path.splitext(os.path.basename(eval_file))[0]
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
