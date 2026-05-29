"""
End-to-end evaluation pipeline for the PointsToWood paper results section.

Runs all three models in forward mode on every cloud in data/eval/, then
evaluates predictions against ground-truth labels.

  Stage 1  PointsToWood  xyz + reflectance    (conda env: ptw)
  Stage 2  FSCT          xyz only             (conda env: fsct)
  Stage 3  KPConv        xyz + reflectance    (conda env: kpconv)
  Stage 4  Evaluation    PTW / KPConv / FSCT  (conda env: ptw)

Each model writes predictions back to data/eval/ as *-<model>.ply.
Existing prediction files are overwritten on each run.

Usage:
    conda run -n ptw python run_eval.py
"""
import subprocess
import sys
from pathlib import Path

REPO_DIR      = Path(__file__).parent
EVAL_DIR      = REPO_DIR / 'data' / 'eval'
FSCT_SCRIPT   = Path('/home/harryjfowen/Software/FSCT/eval_ptw.py')
KPCONV_SCRIPT = Path('/home/harryjfowen/KPConv-PyTorch/eval_PointsToWood.py')
EVAL_SCRIPT   = REPO_DIR / 'eval_comparison.py'

# Suffixes that mark prediction outputs — not inputs
_PRED_SUFFIXES = ('-ptw', '-fsct', '-kpconv', '-p2w', '_p2w')


def eval_clouds():
    """Return sorted list of raw eval cloud paths (no prediction files, no dirs)."""
    return sorted(
        p for p in EVAL_DIR.glob('*.ply')
        if p.is_file() and not any(p.stem.endswith(s) for s in _PRED_SUFFIXES)
    )


def run(label, conda_env, cmd, cwd=None):
    print(f'\n{"─" * 62}')
    print(f'  {label}')
    print(f'{"─" * 62}\n')
    result = subprocess.run(
        ['conda', 'run', '--no-capture-output', '-n', conda_env] + cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
    )
    if result.returncode != 0:
        print(f'\n  ERROR: {label} exited with code {result.returncode}', file=sys.stderr)
        sys.exit(result.returncode)


clouds = eval_clouds()
print(f'Found {len(clouds)} eval clouds in {EVAL_DIR}')

# Stage 1 — PointsToWood
run(
    'Stage 1 — PointsToWood inference (xyz + reflectance)',
    'ptw',
    ['python', 'predict.py', '--point-cloud'] + [str(p) for p in clouds],
    cwd=REPO_DIR,
)

# Stage 2 — FSCT
run(
    'Stage 2 — FSCT inference (xyz only)',
    'fsct',
    ['python', str(FSCT_SCRIPT)],
)

# Stage 3 — KPConv
run(
    'Stage 3 — KPConv inference (xyz + reflectance)',
    'kpconv',
    ['python', str(KPCONV_SCRIPT)],
    cwd=KPCONV_SCRIPT.parent,
)

# Stage 4 — Evaluate
run(
    'Stage 4 — Evaluation',
    'ptw',
    ['python', str(EVAL_SCRIPT)],
    cwd=REPO_DIR,
)
