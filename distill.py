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


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def get_path(location: str = "") -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, location).replace("\\", "/") if location else base

if __name__ == "__main__":
    print('\n=== PointsToWood Distillation ===')

    parser = argparse.ArgumentParser(description='Distill biome-specific models from EU teacher')

    # Core arguments (matching train.py)
    parser.add_argument('--device', type=str, default='cuda', help='Insert either "cuda" or "cpu"')
    parser.add_argument('--region', type=str, default='fin', help='Region prefix (e.g. fin, spa, cam, uk). Filters data/train/*.ply by this prefix. Use "eu" or "global" for all files.')
    parser.add_argument('--model', type=str, default=None, help='Student model name')
    parser.add_argument('--teacher-model', type=str, default='eu.pth', help='Teacher model path (default: eu.pth)')

    # Data preprocessing (matching train.py)
    parser.add_argument('--resolution', type=float, default=0.0, help='Representative spacing before final voxel blocks [m]. Default 0 = adaptive per scale (spacing = grid_size / 100). Set e.g. 0.02 for fixed 2cm.')
    parser.add_argument('--grid-size', type=float, nargs='+', default=[1.0], help='Final model voxel block sizes for voxelization (default: 1.0m, matching eu.pth)')
    parser.add_argument('--min-pts', '--min-points', dest='min_pts', type=int, default=1024, help='Minimum points required per preprocessed voxel (default: 1024, matching eu.pth run)')
    parser.add_argument('--max-pts', '--max-points', dest='max_pts', type=int, default=None, help='Maximum points kept per voxel before training. Default: --max-points-per-batch. Use 0 for no per-voxel subsampling.')
    parser.add_argument('--grid-method', type=str, default='max', choices=['mean', 'max'], help="Pre-downsampling representative method inside the small spacing voxels")
    parser.add_argument('--collect-grid-size', type=float, default=0.0, help='Optional post-aggregation voxel size (m)')
    parser.add_argument('--preprocess', action='store_true', help="Preprocess point clouds into voxels")
    parser.add_argument('--sor', action='store_true', help='Per-voxel SOR before subsampling when writing voxels (k=10, 1 std)')
    parser.add_argument('--sor-k', type=int, default=10, help='SOR neighbours per voxel (default 10)')
    parser.add_argument('--sor-std', type=float, default=1.0, help='SOR threshold = mean + sor_std*std (default 1.0)')
    parser.add_argument('--eval', nargs='?', default=True, const=True, metavar='PLY',
                        help='Run deployment-style eval on region-filtered data/eval files. Optionally pass one PLY filename.')
    parser.add_argument('--no-eval', dest='eval', action='store_const', const=None,
                        help='Disable eval-folder deployment evaluation.')
    parser.add_argument('--eval-grid-size', type=float, default=2.0,
                        help='Model voxel grid size for eval-folder deployment evaluation (default 2.0m).')
    parser.add_argument('--eval-collect-grid-size', type=float, default=0.04,
                        help='Prediction aggregation grid for eval-folder deployment evaluation (default 0.04m).')
    parser.add_argument('--eval-any-wood', type=float, default=0.5,
                        help='Any-wood threshold at eval collection grid; set negative to use argmax confidence instead.')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='Run eval-folder deployment evaluation every N epochs (default 10; 0 disables during training).')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (fixed-batch path; ignored when point-budget batching is active)')
    parser.add_argument('--max-points-per-batch', type=int, default=16384,
                        help='Target max points per batch. Enables point-budget-aware batching when >0. Default: 16384, matching eu.pth run.')
    parser.add_argument('--min-points-per-batch', type=int, default=16000,
                        help='Avoid tiny underutilized batches in point-budget mode (default 16000).')
    parser.add_argument('--packing-mode', type=str, default='balanced_bfd', choices=['ffd', 'bfd', 'balanced', 'balanced_bfd'],
                        help='Batch packing mode in point-budget sampler (default: balanced_bfd).')
    parser.add_argument('--accumulation-steps', type=int, default=8, help='Gradient accumulation steps (default 8, matching eu.pth run).')
    parser.add_argument('--epoch-steps', type=int, default=300, help='Max training steps per epoch (0 = use all batches; default 300 for distillation).')
    parser.add_argument('--val-steps', type=int, default=100, help='Max validation steps per epoch (0 = use all batches; default 100 = epoch_steps / 3).')
    parser.add_argument('--num-epochs', default=90, type=int, help='Number of training epochs (default: 90 for distillation)')
    parser.add_argument('--max-lr', type=float, default=1e-3, help='Maximum learning rate for OneCycleLR (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay for optimizer (default: 1e-2)')
    parser.add_argument('--augmentation', action='store_true', default=True, help='Enable data augmentation')
    parser.add_argument('--test', action='store_true', default=True, help='Run testing after training')
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
    parser.add_argument('--gamma-max', type=float, default=4.0,
                        help='Maximum cyclical focal gamma for hard GT loss (default 4.0, matches main run).')
    parser.add_argument('--gamma-peak-pct', type=float, default=0.33,
                        help='Fraction of training where cyclical focal gamma peaks (default 0.33).')
    parser.add_argument('--label-smoothing', type=float, default=0.05,
                        help='Label smoothing for hard GT loss (default 0.05, matches main run).')
    parser.add_argument('--boundary-weight', type=float, default=0.0,
                        help='Optional boundary edge-score hard-loss upweighting max extra weight (default 0).')
    parser.add_argument('--boundary-ramp-start', type=float, default=0.1,
                        help='Fraction of training before boundary hard-loss ramp begins (default 0.1).')
    parser.add_argument('--refl-fp-penalty', type=float, default=0.05,
                        help='Ramped penalty for leaf points predicted as wood (default 0.05, matches main run).')
    parser.add_argument('--refl-fp-flat', action='store_true', default=False,
                        help='Apply reflectance false-positive penalty equally to all leaf points instead of weighting bright leaves more.')
    parser.add_argument('--no-refl-fp-ramp', dest='refl_fp_ramp', action='store_false', default=True,
                        help='Disable ramp-up for reflectance false-positive penalty.')
    parser.add_argument('--contrastive-weight', type=float, default=0.05,
                        help='Boundary-focused SupCon auxiliary weight on student projection head (default 0.05; set 0 to disable).')
    parser.add_argument('--difficulty-mining', action='store_true', default=True,
                        help='Track per-voxel boundary-weighted loss EMA and progressively up-weight hard training voxels (default: on, matching train.py).')
    parser.add_argument('--no-difficulty-mining', dest='difficulty_mining', action='store_false',
                        help='Disable per-voxel hard sample mining.')
    parser.add_argument('--per-voxel-alpha', type=float, default=3.0,
                        help='Strength of per-voxel hard sample mining when --difficulty-mining is set (default 3.0, matching train.py).')
    parser.add_argument('--per-voxel-warmup-epochs', type=int, default=None,
                        help='Epochs before per-voxel loss EMA affects sampling (default 10%% of epochs, min 3).')
    parser.add_argument('--voxel-ema-alpha', type=float, default=0.9,
                        help='EMA decay for per-voxel hard sample mining (default 0.9).')
    parser.add_argument('--unseen-voxel-boost', type=float, default=8.0,
                        help='During per-voxel warmup, multiply sampler weight for unseen voxels (default 8).')
    parser.add_argument('--coverage-warmup-replacement', action='store_true', default=False,
                        help='Allow weighted replacement during per-voxel coverage warmup.')

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
    parser.add_argument('--student-c', type=int, default=32, help='Student base channels C (default 32; compact but expressive for NetFull C=128)')
    parser.add_argument('--student-kernels', type=int, nargs='+', default=[16, 16, 16],
                        metavar='K',
                        help='Student kernel counts per SA stage (default 16 16 16, matching eu.pth teacher geometry). '
                             'A single value is broadcast to all stages.')
    parser.add_argument('--student-k-neighbors', type=int, default=32,
                        help='Student kNN neighbourhood size per SA stage (default 32; matches eu.pth teacher reach).')
    parser.add_argument('--student-learnable-kernels', dest='student_learnable_kernels', action='store_true', help='Use learnable kernel directions in student')
    parser.add_argument('--student-fixed-kernels', dest='student_learnable_kernels', action='store_false', help='Use fixed kernel directions in student (default)')
    parser.add_argument('--student-blocks', type=int, nargs=3, default=[2, 3, 1], metavar=('SA1', 'SA2', 'SA3'),
                        help='Residual blocks per SA stage (default 2 3 1). Use 1 2 1 for a tiny baseline.')
    parser.add_argument('--student-compressed-head', dest='student_compressed_head', action='store_true',
                        help='Use compact student segmentation head (default).')
    parser.add_argument('--student-full-head', dest='student_compressed_head', action='store_false',
                        help='Use full-width student segmentation head.')
    parser.add_argument('--student-head-dim', type=int, default=64,
                        help='Compressed student head width when --student-compressed-head is active (default 64).')
    parser.add_argument('--spatial-mix-lite', action='store_true', default=True, help='Enable low-cost spatial mixing in SA2 residual blocks')
    parser.add_argument('--ema', action='store_true', default=False, help='Use EMA model weights for validation')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay per optimizer step when --ema is set (default 0.999)')
    parser.add_argument('--amp-dtype', type=str, default='auto', choices=['auto', 'fp16', 'bf16'],
                        help='AMP precision: auto (prefer bf16 if supported), fp16, or bf16 (default auto).')
    parser.add_argument('--balance-mode', dest='balance_mode', type=str, default='downsampling', choices=['downsampling', 'upsampling'], help='Class balancing mode')
    parser.add_argument('--wandb', action='store_true', default=False, help="Enable wandb logging (default: disabled)")
    parser.add_argument('--verbose', action='store_true', default=True, help="Print detailed information")
    parser.add_argument('--check-teacher', action='store_true', default=False,
                        help='Evaluate teacher baseline on test set then exit — use to verify teacher loads correctly before a full run.')
    parser.add_argument('--scratch', action='store_true', default=False,
                        help='Train NetLight purely supervised on biome data — no teacher, no KD losses. '
                             'Critical baseline: same architecture as distilled student, no knowledge transfer. '
                             'Teacher model is still loaded to log its biome performance for comparison.')
    parser.add_argument('--fast-distil', action='store_true', default=False,
                        help='10-epoch rapid distillation mode: skips Phase 1 warm-up, full KD from epoch 1, '
                             'eval every epoch. Designed for domain-specific compression from a strong teacher.')
    parser.set_defaults(student_learnable_kernels=False, student_compressed_head=True)  # fixed kernels by default — matches teacher, preserves flashlight semantics

    args = parser.parse_args()
    args.wdir = get_path()
    args.mode = 'distill'
    if args.max_pts is None:
        args.max_pts = args.max_points_per_batch
    args.per_voxel_difficulty = bool(args.difficulty_mining)
    if args.per_voxel_warmup_epochs is None:
        args.per_voxel_warmup_epochs = max(3, int(round(0.10 * args.num_epochs)))

    if args.model is None:
        args.model = f'mcc-{args.region}.pth'

    # Global pool: all raw .ply files live in data/train/, data/test/
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
    eval_files = _collect_ply(args.eval_dir)

    _pstr = 'all' if _prefixes is None else ', '.join(_prefixes)
    args.region_prefixes = _pstr
    args.train_file_count = len(train_files)
    args.test_file_count = len(test_files)
    args.eval_file_count = 0
    args.startup_ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    args.eval_vxfile = os.path.join(args.eval_dir, f'eval_voxels_{args.region}')
    if args.eval is not None and eval_files:
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
        args.eval = True
        args.eval_file_count = len(args.eval_files)
    else:
        args.eval = False
        args.eval_files = []
    if getattr(args, 'eval_any_wood', None) is not None and args.eval_any_wood < 0:
        args.eval_any_wood = None

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
                args.source_file_prefix = os.path.splitext(os.path.basename(eval_file))[0]
                args.vxfile = args.eval_vxfile
                if args.verbose:
                    print(f'  Voxelising to {args.grid_size} grid sizes')
                preprocess(args)
            args.grid_size = original_grid_sizes

        if args.verbose:
            print('----- Preprocessing completed -----\n')

    # Check data exists
    if not os.path.exists(args.trfile):
        raise ValueError(f'Training data not found at {args.trfile}. Run with --preprocess first.')

    if args.test and not os.path.exists(args.tefile):
        raise ValueError(f'Test data not found at {args.tefile}. Run with --preprocess first.')

    if getattr(args, 'eval', False) and not args.preprocess and not os.path.isdir(args.eval_vxfile):
        print(f'Eval voxels not found at {args.eval_vxfile}; run with --preprocess to enable eval-folder checks.')
        args.eval = False

    # Check teacher model exists
    teacher_path = os.path.join(args.wdir, 'model', args.teacher_model)
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f'Teacher model not found: {teacher_path}')

    # Start distillation training
    SemanticDistillation(args)

    elapsed_time = datetime.datetime.now() - start
    print('\n\n=== DISTILLATION COMPLETED ===')
    print(f'Total time: {elapsed_time}')
    print(f'RAM usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0:.2f} GB')
