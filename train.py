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

        parser = argparse.ArgumentParser()

        parser.add_argument('--device', type=str, default='cuda', help='"cuda" or "cpu"')
        parser.add_argument('--region', type=str, default='eu', help='Data region (e.g. eu, spain, germany, global)')
        parser.add_argument('--num-epochs', default=180, type=int, metavar='N', help='Number of total epochs to run')
        parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (max_lr in OneCycleLR)')
        parser.add_argument('--model', type=str, default=None, help='Model filename (e.g. model.pth)')
        parser.add_argument('--max-points-per-batch', type=int, default=32000,
                            help='Target max points per batch (default 32000). Adjust for GPU memory.')
        parser.add_argument('--min-points', dest='min_pts', type=int, default=2048,
                            help='Minimum points required per preprocessed voxel (default 2048).')
        parser.add_argument('--max-points', dest='max_pts', type=int, default=None,
                            help='Maximum points kept per voxel before training. Default: --max-points-per-batch. Use 0 for no per-voxel subsampling.')
        parser.add_argument('--grid-size', type=float, nargs='+', default=[2.0],
                            help='Final model voxel block sizes for voxelization (default: 2.0)')
        parser.add_argument('--num-kernel-points', type=int, nargs='+', default=[16, 16, 16],
                            help='Kernel points per SA stage: one value for all stages, or three values for SA1 SA2 SA3 (default: 16 16 16)')
        parser.add_argument('--preprocess', action='store_true', help='Preprocess point clouds into voxels')
        parser.add_argument('--tune', action='store_true', help='Tune with lower learning rate schedule')
        parser.add_argument('--eval', nargs='?', default=True, const=True, metavar='PLY',
                            help='Eval visualization (optional PLY filename). Use --no-eval to disable.')
        parser.add_argument('--no-eval', dest='eval', action='store_const', const=None,
                            help='Disable eval visualization.')
        parser.add_argument('--wandb', dest='wandb', action='store_true', default=False,
                            help='Enable wandb logging (default: disabled)')
        parser.add_argument('--epoch-steps', type=int, default=0,
                            help='Max training steps per epoch (0 = full pass). Val steps = epoch_steps // 3, or full val if 0.')
        parser.add_argument('--overlap', type=int, default=0,
                            help='Number of additional shifted grid origins when writing training voxels (0=none, 2=3x voxels, max 8). Requires re-preprocessing.')
        parser.add_argument('--no-augmentation', dest='augmentation', action='store_false', default=True,
                            help='Disable per-sample geometry/reflectance augmentation.')
        parser.add_argument('--no-density-aug', dest='density_aug', action='store_false', default=True,
                            help='Disable batch-level density downsampling augmentation.')
        parser.add_argument('--density-aug-prob', type=float, default=0.20,
                            help='Probability of applying density augmentation to a batch (default 0.20).')
        parser.add_argument('--density-aug-spacing', type=float, nargs=2, default=[0.01, 0.04],
                            metavar=('MIN', 'MAX'),
                            help='Density augmentation spacing range in metres; only coarsens native spacing.')
        parser.add_argument('--density-aug-hard-threshold', type=float, default=0.75,
                            help='Skip batch-level density augmentation when hardest sample difficulty is at or above this value.')
        parser.add_argument('--density-aug-difficulty-power', type=float, default=2.0,
                            help='Exponent for reducing density augmentation probability as batch difficulty rises.')
        parser.add_argument('--refl-diagnostics', action='store_true', default=False,
                            help='Run extra reflectance sensitivity diagnostics. Uses an additional validation backward pass.')
        parser.add_argument('--accumulation-steps', type=int, default=8,
                            help='Gradient accumulation steps (default 8).')
        parser.add_argument('--memory-efficient-conv', action='store_true', default=False,
                            help='Use lower-memory, slower AnisotropicConv aggregation.')
        parser.add_argument('--no-sparse-max', dest='sparse_max', action='store_false',
                            help='Use softmax kernel routing instead of sparsemax in AnisotropicConv.')
        parser.set_defaults(sparse_max=True)
        parser.add_argument('--compressed-head', dest='compressed_head', action='store_true',
                            help='Use compact compressed seg head instead of the default full-width FP head.')
        parser.add_argument('--no-compressed-head', dest='compressed_head', action='store_false',
                            help='Use the default full-width FP seg head.')
        parser.set_defaults(compressed_head=False)
        parser.add_argument('--compressed-head-dim', type=int, default=64,
                            help='Channel width of the compressed seg head (default 64). Use 32 for a tighter bottleneck.')
        parser.add_argument('--k-neighbors', dest='k_neighbors', type=int, default=16,
                            help='kNN neighbourhood size for each SA module (default 16). Higher values (e.g. 32) extend the reach along reflectance chains at the cost of more graph edges.')
        parser.add_argument('--refl-fp-penalty', type=float, default=0.05,
                            help='Ramped penalty weight for leaf points predicted as wood, weighted by reflectance when --refl-fp-flat is not set (default 0.05).')
        parser.add_argument('--refl-fp-flat', action='store_true', default=False,
                            help='Apply reflectance false-positive penalty equally to all leaf points instead of weighting bright leaves more.')
        parser.add_argument('--no-refl-fp-ramp', dest='refl_fp_ramp', action='store_false', default=True,
                            help='Disable ramp-up for reflectance false-positive penalty.')
        parser.add_argument('--adaptive-sampling-metric', type=str, default='balanced_acc',
                            choices=['balanced_acc', 'wood_f1', 'mcc', 'wood_recall'],
                            help='Validation group metric used by --difficulty-mining (default balanced_acc).')
        parser.add_argument('--adaptive-sampling-alpha', type=float, default=5.0,
                            help='Steepness of group sampling curve: higher = stronger contrast between easy/hard sites (default 5.0).')
        parser.add_argument('--adaptive-sampling-min', type=float, default=0.3,
                            help='Minimum sampling multiplier for easy sites (default 0.3 — easy sites get ~30%% of baseline).')
        parser.add_argument('--adaptive-sampling-max', type=float, default=3.0,
                            help='Maximum sampling multiplier for hard sites (default 3.0).')
        parser.add_argument('--ema-decay', type=float, default=0.995,
                            help='EMA decay per optimizer step (default 0.995). Try 0.98 or 0.90 for faster EMA.')
        parser.add_argument('--no-ema', dest='ema', action='store_false', default=True,
                            help='Disable EMA validation/checkpoint weights.')
        parser.add_argument('--difficulty-mining', action='store_true', default=True,
                            help='Track per-voxel boundary-weighted loss EMA and progressively up-weight hard training voxels.')
        parser.add_argument('--no-difficulty-mining', dest='difficulty_mining', action='store_false',
                            help='Disable difficulty mining.')
        parser.add_argument('--unseen-voxel-boost', type=float, default=8.0,
                            help='During per-voxel warmup, multiply sampler weight for voxels not yet seen by the EMA tracker.')
        parser.add_argument('--coverage-warmup-replacement', action='store_true', default=False,
                            help='Allow weighted replacement during per-voxel coverage warmup. Default prevents replacement where possible.')
        parser.add_argument('--contrastive-weight', type=float, default=0.1,
                            help='Weight for supervised contrastive loss on FP1 features (default 0.1, set 0 to disable).')

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
        args.pointcutmix = False
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
        args.drop_path_rate = 0.0
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
