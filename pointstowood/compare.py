"""
compare.py — run scratch baseline and distilled student for one biome, then write a report.

Usage:
    python compare.py --region spain
    python compare.py --region poland --num-epochs 90
    python compare.py --region finland --all-biomes   # runs all three sequentially

Produces: reports/<region>_comparison.md
"""
import datetime
import os
import re
import argparse
import glob
import shutil
import copy

from src.distillation import SemanticDistillation
from src.preprocessing import preprocess
from src.io import load_file
from src.utils import preprocess_point_cloud_data


def dir_path(string):
    if os.path.isdir(string):
        return string
    raise NotADirectoryError(string)


def get_path(location_in_pointstowood: str = '') -> str:
    current_wdir = os.getcwd()
    match = re.search(r'PointsToWood.*?pointstowood', current_wdir, re.IGNORECASE)
    if not match:
        raise ValueError('"PointsToWood/pointstowood" not found in cwd')
    last_index = match.end()
    output_path = current_wdir[:last_index]
    if location_in_pointstowood:
        output_path = os.path.join(output_path, location_in_pointstowood)
    return output_path.replace('\\', '/')


def build_args(base_args, region, scratch, model_name):
    """Clone base args and set region/scratch/model fields."""
    args = copy.deepcopy(base_args)
    args.region = region
    args.scratch = scratch
    args.model = model_name
    args.mode = 'distill'
    args.wdir = get_path()
    args.train_dir = os.path.join(args.wdir, f'data/{region}_train')
    args.test_dir = os.path.join(args.wdir, f'data/{region}_test')
    args.trfile = os.path.join(args.train_dir, 'voxels')
    args.tefile = os.path.join(args.test_dir, 'voxels')
    return args


def _fmt(v, decimals=4):
    if v != v:  # nan
        return 'N/A'
    return f'{v:.{decimals}f}'


def _delta(a, b):
    """Return formatted signed delta (a - b)."""
    if a != a or b != b:
        return 'N/A'
    d = a - b
    sign = '+' if d >= 0 else ''
    return f'{sign}{d:.4f}'


def write_report(region, teacher, scratch, paced, output_dir):
    """Write a markdown comparison report for one biome: scratch vs PACED KD."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{region}_comparison.md')

    comp = paced.get('compression_ratio', float('nan'))

    t_hmcc   = teacher.get('hmcc', float('nan'))
    sc_hmcc  = scratch.get('best_mcc_harmonic', float('nan'))
    pa_hmcc  = paced.get('best_mcc_harmonic', float('nan'))

    t_refl   = teacher.get('mcc_refl', float('nan'))
    sc_refl  = scratch.get('best_mcc_refl', float('nan'))
    pa_refl  = paced.get('best_mcc_refl', float('nan'))

    t_norefl  = teacher.get('mcc_norefl', float('nan'))
    sc_norefl = scratch.get('best_mcc_no_refl', float('nan'))
    pa_norefl = paced.get('best_mcc_no_refl', float('nan'))

    t_hauprc  = teacher.get('hauprc', float('nan'))
    sc_hauprc = scratch.get('best_auprc_harmonic', float('nan'))
    pa_hauprc = paced.get('best_auprc_harmonic', float('nan'))

    t_auprc_refl   = teacher.get('auprc_refl', float('nan'))
    sc_auprc_refl  = scratch.get('best_auprc_refl', float('nan'))
    pa_auprc_refl  = paced.get('best_auprc_refl', float('nan'))

    t_auprc_norefl  = teacher.get('auprc_norefl', float('nan'))
    sc_auprc_norefl = scratch.get('best_auprc_no_refl', float('nan'))
    pa_auprc_norefl = paced.get('best_auprc_no_refl', float('nan'))

    t_fb_refl   = teacher.get('fbeta_refl', float('nan'))
    sc_fb_refl  = scratch.get('best_fbeta_refl', float('nan'))
    pa_fb_refl  = paced.get('best_fbeta_refl', float('nan'))

    t_fb_norefl  = teacher.get('fbeta_norefl', float('nan'))
    sc_fb_norefl = scratch.get('best_fbeta_no_refl', float('nan'))
    pa_fb_norefl = paced.get('best_fbeta_no_refl', float('nan'))

    sc_fbe_refl   = scratch.get('best_fbeta_edge_refl', float('nan'))
    pa_fbe_refl   = paced.get('best_fbeta_edge_refl', float('nan'))
    sc_fbe_norefl = scratch.get('best_fbeta_edge_no_refl', float('nan'))
    pa_fbe_norefl = paced.get('best_fbeta_edge_no_refl', float('nan'))

    t_brier_refl    = teacher.get('brier_refl',           float('nan'))
    t_brier_norefl  = teacher.get('brier_norefl',         float('nan'))
    sc_brier_refl   = scratch.get('best_brier_refl',      float('nan'))
    sc_brier_norefl = scratch.get('best_brier_no_refl',   float('nan'))
    pa_brier_refl   = paced.get('best_brier_refl',        float('nan'))
    pa_brier_norefl = paced.get('best_brier_no_refl',     float('nan'))

    best_hmcc  = max(v for v in [sc_hmcc, pa_hmcc] if v == v)
    best_label = 'PACED KD' if pa_hmcc == best_hmcc else 'Scratch'
    retention_paced = (pa_hmcc / t_hmcc * 100) if (t_hmcc == t_hmcc and t_hmcc > 0) else float('nan')

    lines = [
        f'# Biome Distillation Report — {region.capitalize()}',
        f'',
        f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}',
        f'',
        f'## Summary',
        f'',
        f'| | Value |',
        f'|---|---|',
        f'| Biome | {region.capitalize()} |',
        f'| Teacher model | {paced.get("teacher_model", "mcc-eu.pth")} |',
        f'| Compression ratio | {_fmt(comp, 1)}x |',
        f'| Teacher params | {paced.get("teacher_params", 0):,} |',
        f'| Student params | {paced.get("student_params", 0):,} |',
        f'| Best student HMCC | {_fmt(best_hmcc)} ({best_label}) |',
        f'| Teacher HMCC | {_fmt(t_hmcc)} |',
        f'| PACED retention of teacher HMCC | {_fmt(retention_paced, 1)}% |',
        f'| PACED − Scratch HMCC | {_delta(pa_hmcc, sc_hmcc)} |',
        f'',
        f'---',
        f'',
        f'## Primary Metric: Harmonic MCC',
        f'',
        f'HMCC = harmonic mean of MCC with-reflectance and without-reflectance.',
        f'Higher is better. Penalises models that rely on reflectance to function.',
        f'',
        f'| Model | HMCC | MCC (with refl) | MCC (no refl) |',
        f'|---|---|---|---|',
        f'| **Teacher** (NetFull, pan-EU) | {_fmt(t_hmcc)} | {_fmt(t_refl)} | {_fmt(t_norefl)} |',
        f'| **Scratch** (NetLight, no KD) | {_fmt(sc_hmcc)} | {_fmt(sc_refl)} | {_fmt(sc_norefl)} |',
        f'| **PACED KD** (frontier-weighted) | {_fmt(pa_hmcc)} | {_fmt(pa_refl)} | {_fmt(pa_norefl)} |',
        f'| PACED − Scratch | {_delta(pa_hmcc, sc_hmcc)} | {_delta(pa_refl, sc_refl)} | {_delta(pa_norefl, sc_norefl)} |',
        f'| Teacher − PACED | {_delta(t_hmcc, pa_hmcc)} | {_delta(t_refl, pa_refl)} | {_delta(t_norefl, pa_norefl)} |',
        f'',
        f'---',
        f'',
        f'## Probability Ranking Quality: Harmonic AUPRC',
        f'',
        f'AUPRC is threshold-agnostic — it measures how well the model ranks points regardless of',
        f'where the decision boundary sits. This is a fairer comparison for the teacher, which is a',
        f'pan-EU generalist whose optimal threshold on a single biome is not necessarily 0.5.',
        f'',
        f'| Model | H-AUPRC | AUPRC (with refl) | AUPRC (no refl) |',
        f'|---|---|---|---|',
        f'| **Teacher** (NetFull, pan-EU) | {_fmt(t_hauprc)} | {_fmt(t_auprc_refl)} | {_fmt(t_auprc_norefl)} |',
        f'| **Scratch** (NetLight, no KD) | {_fmt(sc_hauprc)} | {_fmt(sc_auprc_refl)} | {_fmt(sc_auprc_norefl)} |',
        f'| **PACED KD** (frontier-weighted) | {_fmt(pa_hauprc)} | {_fmt(pa_auprc_refl)} | {_fmt(pa_auprc_norefl)} |',
        f'| PACED − Scratch | {_delta(pa_hauprc, sc_hauprc)} | {_delta(pa_auprc_refl, sc_auprc_refl)} | {_delta(pa_auprc_norefl, sc_auprc_norefl)} |',
        f'| Teacher − PACED | {_delta(t_hauprc, pa_hauprc)} | {_delta(t_auprc_refl, pa_auprc_refl)} | {_delta(t_auprc_norefl, pa_auprc_norefl)} |',
        f'',
        f'---',
        f'',
        f'## Probability Calibration: Brier Score',
        f'',
        f'Brier Score = mean((p_wood - GT)²). Lower is better (0=perfect, 0.25=no skill).',
        f'Directly measures calibration — exposes collapsed distributions that AUPRC misses.',
        f'',
        f'| Model | Brier (with refl) | Brier (no refl) |',
        f'|---|---|---|',
        f'| **Teacher** | {_fmt(t_brier_refl)} | {_fmt(t_brier_norefl)} |',
        f'| **Scratch** | {_fmt(sc_brier_refl)} | {_fmt(sc_brier_norefl)} |',
        f'| **PACED KD** | {_fmt(pa_brier_refl)} | {_fmt(pa_brier_norefl)} |',
        f'| PACED − Scratch | {_delta(pa_brier_refl, sc_brier_refl)} | {_delta(pa_brier_norefl, sc_brier_norefl)} |',
        f'',
        f'---',
        f'',
        f'## Classification Quality: Fbeta (β=0.5, precision-weighted)',
        f'',
        f'| Model | Fbeta (with refl) | Fbeta (no refl) |',
        f'|---|---|---|',
        f'| **Teacher** | {_fmt(t_fb_refl)} | {_fmt(t_fb_norefl)} |',
        f'| **Scratch** | {_fmt(sc_fb_refl)} | {_fmt(sc_fb_norefl)} |',
        f'| **PACED KD** | {_fmt(pa_fb_refl)} | {_fmt(pa_fb_norefl)} |',
        f'| PACED − Scratch | {_delta(pa_fb_refl, sc_fb_refl)} | {_delta(pa_fb_norefl, sc_fb_norefl)} |',
        f'',
        f'---',
        f'',
        f'## Boundary Quality: Edge Fbeta',
        f'',
        f'Evaluated only on transition points (wood/leaf boundaries).',
        f'Key metric for complex branching geometry (e.g. Mediterranean Spain).',
        f'',
        f'| Model | Edge Fbeta (with refl) | Edge Fbeta (no refl) |',
        f'|---|---|---|',
        f'| **Scratch** | {_fmt(sc_fbe_refl)} | {_fmt(sc_fbe_norefl)} |',
        f'| **PACED KD** | {_fmt(pa_fbe_refl)} | {_fmt(pa_fbe_norefl)} |',
        f'| PACED − Scratch | {_delta(pa_fbe_refl, sc_fbe_refl)} | {_delta(pa_fbe_norefl, sc_fbe_norefl)} |',
        f'',
        f'---',
        f'',
        f'## Interpretation',
        f'',
        _interpret(region, pa_hmcc, sc_hmcc, t_hmcc, retention_paced, pa_norefl, sc_norefl),
        f'',
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    print(f'\nReport written: {path}')
    return path


def _interpret(region, pa_hmcc, sc_hmcc, t_hmcc, retention, pa_norefl, sc_norefl):
    lines = []
    if pa_hmcc != pa_hmcc:
        return '*(metrics not available — run with --test)*'

    paced_vs_scratch = pa_hmcc - sc_hmcc if (pa_hmcc == pa_hmcc and sc_hmcc == sc_hmcc) else float('nan')

    # PACED vs scratch verdict
    if paced_vs_scratch != paced_vs_scratch:
        pass
    elif paced_vs_scratch > 0.01:
        lines.append(f'**PACED KD beats scratch** (+{paced_vs_scratch:.4f} HMCC on {region.capitalize()}): '
                     f'frontier-weighted distillation successfully transfers pan-European teacher knowledge '
                     f'that biome-local training alone cannot recover. The specialist advantage hypothesis holds.')
    elif paced_vs_scratch > -0.01:
        lines.append(f'**PACED KD matches scratch** (ΔHMCC = {paced_vs_scratch:+.4f} on {region.capitalize()}): '
                     f'distillation neither helps nor hurts at this compression. '
                     f'Local biome data is sufficient for this student capacity.')
    else:
        lines.append(f'**Scratch leads** (ΔHMCC = {paced_vs_scratch:+.4f} on {region.capitalize()}): '
                     f'teacher domain mismatch outweighs KD benefit. '
                     f'Consider adjusting alpha schedule or temperature.')

    # Teacher retention
    if retention == retention:
        quality = ('strong.' if retention >= 90 else 'moderate.' if retention >= 75 else
                   'partial — further tuning may help.')
        lines.append(f'PACED student retains {retention:.1f}% of teacher HMCC — {quality}')

    # No-reflectance gap
    norefl_gain = pa_norefl - sc_norefl if (pa_norefl == pa_norefl and sc_norefl == sc_norefl) else float('nan')
    if norefl_gain == norefl_gain:
        if norefl_gain > 0.02:
            lines.append(f'No-reflectance: PACED gains +{norefl_gain:.4f} MCC vs scratch — '
                         f'teacher geometry-first representations transferring at wood/leaf boundaries.')
        elif norefl_gain > -0.02:
            lines.append(f'No-reflectance: PACED matches scratch ({norefl_gain:+.4f} MCC) — '
                         f'geometry-only performance preserved at 82x compression.')
        else:
            lines.append(f'No-reflectance: scratch leads ({norefl_gain:+.4f} MCC). '
                         f'Consider adjusting alpha schedule or temperature.')

    return '\n\n'.join(lines)


def run_biome(base_args, region):
    print(f'\n{"#"*80}')
    print(f'# BIOME: {region.upper()}')
    print(f'{"#"*80}\n')

    scratch_args = build_args(base_args, region, scratch=True,
                              model_name=f'scratch-{region}.pth')
    paced_args = build_args(base_args, region, scratch=False,
                            model_name=f'paced-{region}.pth')
    paced_args.paced = True

    print(f'\n--- Run A: Scratch baseline ({region}) ---\n')
    scratch_results = SemanticDistillation(scratch_args)

    print(f'\n--- Run B: PACED KD ({region}) ---\n')
    paced_results = SemanticDistillation(paced_args)

    teacher_baseline = paced_results.pop('teacher_baseline', {})
    paced_results['teacher_model'] = base_args.teacher_model

    report_dir = os.path.join(get_path(), 'reports')
    report_path = write_report(region, teacher_baseline, scratch_results, paced_results, report_dir)
    return report_path


if __name__ == '__main__':
    start = datetime.datetime.now()
    print('\n\n=== PointsToWood BIOME COMPARISON ===\n')

    parser = argparse.ArgumentParser(description='Run scratch + distilled training per biome and generate comparison report')

    parser.add_argument('--region', type=str, default='spain', help='Biome region (spain, poland, finland)')
    parser.add_argument('--all-biomes', action='store_true', default=False, help='Run all three biomes sequentially')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--teacher-model', type=str, default='mcc-eu.pth')
    parser.add_argument('--resolution', type=float, default=0.0)
    parser.add_argument('--grid-size', type=float, nargs='+', default=[2.0, 4.0])
    parser.add_argument('--min-pts', type=int, default=2048)
    parser.add_argument('--max-pts', type=int, default=32768)
    parser.add_argument('--grid-method', type=str, default='max', choices=['mean', 'max'])
    parser.add_argument('--collect-grid-size', type=float, default=0.0)
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--sor', action='store_true')
    parser.add_argument('--sor-k', type=int, default=10)
    parser.add_argument('--sor-std', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-points-per-batch', type=int, default=32000)
    parser.add_argument('--min-points-per-batch', type=int, default=16000)
    parser.add_argument('--packing-mode', type=str, default='balanced_bfd',
                        choices=['ffd', 'bfd', 'balanced', 'balanced_bfd'])
    parser.add_argument('--accumulation-steps', type=int, default=4)
    parser.add_argument('--epoch-steps', type=int, default=600)
    parser.add_argument('--val-steps', type=int, default=200)
    parser.add_argument('--num-epochs', default=90, type=int)
    parser.add_argument('--max-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--test', action='store_true', default=True)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--alpha-final', type=float, default=0.65)
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--temperature-floor', type=float, default=1.5)
    parser.add_argument('--paced', action='store_true', default=False,
                        help='Use PACED frontier-weighted soft loss in distilled run')
    parser.add_argument('--rel-kd-weight', type=float, default=0.0)
    parser.add_argument('--rel-kd-anchors', type=int, default=64)
    parser.add_argument('--rel-kd-stages', type=int, default=3)
    parser.add_argument('--gate-kd-weight', type=float, default=0.0)
    parser.add_argument('--feat-kd-weight', type=float, default=0.0)
    parser.add_argument('--proto-kd-weight', type=float, default=0.0)
    parser.add_argument('--koleo-weight', type=float, default=0.0)
    parser.add_argument('--pointcutmix', action='store_true', default=False)
    parser.add_argument('--pointcutmix-prob', type=float, default=0.25)
    parser.add_argument('--pointcutmix-beta', type=float, default=1.0)
    parser.add_argument('--drop-path-rate', type=float, default=0.0)
    parser.add_argument('--learnable-kernels', action='store_true')
    parser.add_argument('--teacher-kernels', type=int, default=16)
    parser.add_argument('--student-c', type=int, default=16)
    parser.add_argument('--student-kernels', type=int, default=8)
    parser.add_argument('--student-learnable-kernels', dest='student_learnable_kernels', action='store_true')
    parser.add_argument('--student-fixed-kernels', dest='student_learnable_kernels', action='store_false')
    parser.add_argument('--dualnorm-lite', action='store_true', default=True)
    parser.add_argument('--spatial-mix-lite', action='store_true', default=True)
    parser.add_argument('--ema', action='store_true', default=False)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--amp-dtype', type=str, default='auto', choices=['auto', 'fp16', 'bf16'])
    parser.add_argument('--balance-mode', dest='balance_mode', type=str, default='downsampling',
                        choices=['downsampling', 'upsampling'])
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Disable wandb by default for comparison runs (two concurrent runs conflict)')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.set_defaults(student_learnable_kernels=True)

    args = parser.parse_args()

    biomes = ['spain', 'poland', 'finland'] if args.all_biomes else [args.region]
    reports = []
    for biome in biomes:
        reports.append(run_biome(args, biome))

    elapsed = datetime.datetime.now() - start
    print(f'\n\n=== COMPARISON COMPLETE ===')
    print(f'Total time: {elapsed}')
    print(f'Reports written to:')
    for r in reports:
        print(f'  {r}')
