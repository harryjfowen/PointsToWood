#!/usr/bin/env python3
"""
Biome Transfer Learning Evaluation Matrix
Evaluates EU, scratch-{biome}, and paced-{biome} models across all biomes.
Uses full predicter.py inference: overlapping voxels, point-level aggregation.
Reports HMCC = harmonic mean of MCC(with-refl) and MCC(no-refl).

Usage:
    python evaluation/biome_matrix_eval.py --eval_root data --output_dir reports/biome_matrix

Expects directories: data/spain_eval/, data/poland_eval/, data/finland_eval/
"""

import os
import gc
import glob
import datetime
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from sklearn.metrics import matthews_corrcoef
from sklearn.exceptions import UndefinedMetricWarning
import torch
from tqdm import tqdm

from src.io import load_file
from src.predicter import SemanticSegmentation
from src.utils import preprocess_point_cloud_data

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ── model / biome config ────────────────────────────────────────────────────

MODEL_ROWS = [
    'eu',
    'scratch-spain', 'paced-spain',
    'scratch-poland', 'paced-poland',
    'scratch-finland', 'paced-finland',
]

MODEL_FILES = {
    'eu':              'mcc-eu.pth',
    'scratch-spain':   'scratch-spain.pth',
    'paced-spain':     'paced-spain.pth',
    'scratch-poland':  'scratch-poland.pth',
    'paced-poland':    'paced-poland.pth',
    'scratch-finland': 'scratch-finland.pth',
    'paced-finland':   'paced-finland.pth',
}

TEST_BIOMES = ['spain', 'poland', 'finland', 'eu']

# ── helpers ─────────────────────────────────────────────────────────────────

def _fmt(v):
    return f"{v:.4f}" if np.isfinite(v) else "N/A"

def _delta(a, b):
    return f"{a - b:+.4f}" if (np.isfinite(a) and np.isfinite(b)) else "N/A"

def _harmonic_mean(a, b):
    if a > 0 and b > 0:
        return 2 * a * b / (a + b)
    return float('nan')

def _get_test_files(eval_root, biome):
    biome_dir = os.path.join(eval_root, f"{biome}_eval")
    files     = glob.glob(os.path.join(biome_dir, "*.ply"))
    return sorted(f for f in files if not f.lower().endswith('_p2w.ply'))

def _load_ground_truth(file_path):
    data = load_file(file_path)
    for col in ['truth', 'label'] + [c for c in data.columns
                if any(k in c.lower() for k in ('truth', 'label', 'class'))]:
        if col in data.columns:
            return (data[col].values > 0).astype(int)
    raise ValueError(f"No ground truth column in {file_path}. Columns: {list(data.columns)}")

# ── inference ────────────────────────────────────────────────────────────────

def _run_inference(file_path, model_name, models_dir, zero_reflectance=False):
    """Full predicter.py pipeline on one file. Returns binary prediction array or None."""
    model_file = MODEL_FILES[model_name]
    model_path = os.path.join(models_dir, model_file)
    if not os.path.exists(model_path):
        print(f"  [skip] model not found: {model_path}")
        return None

    tag      = 'norefl' if zero_reflectance else 'refl'
    base     = os.path.splitext(os.path.basename(file_path))[0]
    temp_dir = f"_bme_{model_name}_{base}_{tag}"

    try:
        class Args:
            def __init__(self):
                self.point_cloud     = [file_path]
                self.file            = file_path
                self.odir            = temp_dir
                self.model           = model_file
                self.wdir            = os.path.dirname(os.path.dirname(model_path))
                self.vxfile          = os.path.join(temp_dir, "voxels")
                self.device          = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.batch_size      = 0
                self.is_wood         = 0.5
                self.any_wood        = 0.5
                self.max_probability = False
                self.verbose         = False
                self.resolution      = 0.0
                self.grid_size       = [1.0, 2.0, 4.0]
                self.overlap         = 0.0
                self.min_pts         = 512
                self.max_pts         = 16384
                self.zero_reflectance = zero_reflectance
                self.boost_perspective = False
                self.denoise         = False
                self.denoise_k       = 16
                self.denoise_std     = 1.0
                self.mode            = 'predict'
                self.num_procs       = -1
                self.grid_method     = 'max'
                self.collect_grid_size = 0.05
                self.memory_fraction = 0.7

        args = Args()
        os.makedirs(args.vxfile, exist_ok=True)

        args.pc, args.headers = load_file(filename=file_path, additional_headers=True, verbose=False)
        args.pc, args.headers, args.reflectance = preprocess_point_cloud_data(args.pc, args.zero_reflectance)

        from src.preprocessing import preprocess
        preprocess(args)

        result = SemanticSegmentation(args)

        if hasattr(result, 'pc') and 'prediction' in result.pc.columns:
            return (result.pc['prediction'].values > 0).astype(int)
        print(f"  [warn] no prediction column for {model_name} on {base} ({tag})")
        return None

    except Exception as e:
        print(f"  [error] {model_name} | {base} | {tag}: {e}")
        return None
    finally:
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# ── per-model-biome evaluation ───────────────────────────────────────────────

def _evaluate(model_name, biome, eval_root, models_dir):
    """Returns dict: mcc_refl, mcc_norefl, hmcc — aggregated over all files in biome."""
    files = _get_test_files(eval_root, biome)
    if not files:
        print(f"  No test files found for {biome}")
        return dict(mcc_refl=float('nan'), mcc_norefl=float('nan'), hmcc=float('nan'))

    all_true, all_refl, all_norefl = [], [], []

    for f in tqdm(files, desc=f"{model_name}→{biome}"):
        try:
            y_true = _load_ground_truth(f)
        except Exception as e:
            print(f"  [skip GT] {e}")
            continue

        p_refl   = _run_inference(f, model_name, models_dir, zero_reflectance=False)
        p_norefl = _run_inference(f, model_name, models_dir, zero_reflectance=True)

        if p_refl is None or p_norefl is None:
            continue

        n = min(len(y_true), len(p_refl), len(p_norefl))
        all_true.append(y_true[:n])
        all_refl.append(p_refl[:n])
        all_norefl.append(p_norefl[:n])

    if not all_true:
        return dict(mcc_refl=float('nan'), mcc_norefl=float('nan'), hmcc=float('nan'))

    y  = np.concatenate(all_true)
    pr = np.concatenate(all_refl)
    pn = np.concatenate(all_norefl)

    mcc_r = matthews_corrcoef(y, pr)
    mcc_n = matthews_corrcoef(y, pn)
    return dict(mcc_refl=mcc_r, mcc_norefl=mcc_n, hmcc=_harmonic_mean(mcc_r, mcc_n))

# ── main evaluator class ─────────────────────────────────────────────────────

class BiomeMatrixEvaluator:
    def __init__(self, eval_root, models_dir):
        self.eval_root = eval_root
        self.models_dir    = models_dir
        self.results       = {}   # (model, biome) → dict

    def run_full_evaluation(self):
        for model in MODEL_ROWS:
            for biome in TEST_BIOMES:
                print(f"\n{'='*60}\n{model}  →  {biome}\n{'='*60}")
                r = _evaluate(model, biome, self.eval_root, self.models_dir)
                self.results[(model, biome)] = r
                print(f"  MCC refl={_fmt(r['mcc_refl'])}  norefl={_fmt(r['mcc_norefl'])}  HMCC={_fmt(r['hmcc'])}")

    def _df(self, metric):
        df = pd.DataFrame(index=MODEL_ROWS, columns=TEST_BIOMES, dtype=float)
        for m in MODEL_ROWS:
            for b in TEST_BIOMES:
                df.loc[m, b] = self.results.get((m, b), {}).get(metric, float('nan'))
        return df

    # ── text report ────────────────────────────────────────────────────────

    def write_paper_report(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'biome_matrix_report.txt')

        hmcc_df   = self._df('hmcc')
        refl_df   = self._df('mcc_refl')
        norefl_df = self._df('mcc_norefl')

        L = []
        L.append("=" * 70)
        L.append("BIOME TRANSFER LEARNING — FULL EVALUATION REPORT")
        L.append(f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        L.append("Primary metric : HMCC = harmonic_mean(MCC_refl, MCC_norefl)")
        L.append("Inference : predicter.py overlapping voxels, point-level aggregation")
        L.append("Compression : 82x  (teacher 23.5M params → student 287K params)")
        L.append("=" * 70)

        # ── matrices ──
        for label, df in [("HMCC", hmcc_df), ("MCC with reflectance", refl_df), ("MCC no reflectance", norefl_df)]:
            L.append(f"\n{'─'*70}\n{label}\n{'─'*70}")
            L.append(f"{'Model':<22}" + "".join(f"{b.capitalize():>12}" for b in TEST_BIOMES))
            L.append("-" * 58)
            prev_grp = None
            for m in MODEL_ROWS:
                grp = m.split('-')[0] if '-' in m else m
                if prev_grp and grp != prev_grp:
                    L.append("")
                prev_grp = grp
                L.append(f"{m:<22}" + "".join(f"{_fmt(df.loc[m, b]):>12}" for b in TEST_BIOMES))

        # ── analysis ──
        L.append(f"\n{'='*70}\nANALYSIS\n{'='*70}")

        L.append("\n── In-domain vs Transfer (HMCC) ──")
        for biome in TEST_BIOMES:
            sc = f"scratch-{biome}"
            pa = f"paced-{biome}"
            others = [b for b in TEST_BIOMES if b != biome]

            sc_in  = hmcc_df.loc[sc, biome]
            pa_in  = hmcc_df.loc[pa, biome]
            eu_in  = hmcc_df.loc['eu', biome]
            sc_out = float(np.nanmean([hmcc_df.loc[sc, b] for b in others]))
            pa_out = float(np.nanmean([hmcc_df.loc[pa, b] for b in others]))
            eu_out = float(np.nanmean([hmcc_df.loc['eu', b] for b in others]))

            L.append(f"\n  {biome.upper()}")
            L.append(f"    Scratch  in-domain={_fmt(sc_in)}  transfer={_fmt(sc_out)}  drop={_delta(sc_out, sc_in)}")
            L.append(f"    PACED    in-domain={_fmt(pa_in)}  transfer={_fmt(pa_out)}  drop={_delta(pa_out, pa_in)}")
            L.append(f"    EU       on {biome:<8}={_fmt(eu_in)}  transfer avg={_fmt(eu_out)}")
            L.append(f"    PACED transfer advantage over scratch: {_delta(pa_out, sc_out)}")

        L.append("\n── EU Generalisation ──")
        eu_vals = [hmcc_df.loc['eu', b] for b in TEST_BIOMES]
        L.append("  " + "  ".join(f"{b}={_fmt(v)}" for b, v in zip(TEST_BIOMES, eu_vals)))
        L.append(f"  EU avg HMCC across all biomes: {_fmt(float(np.nanmean(eu_vals)))}")

        L.append("\n── Cross-biome Transfer: PACED vs Scratch ──")
        for src in TEST_BIOMES:
            sc = f"scratch-{src}"
            pa = f"paced-{src}"
            for tgt in TEST_BIOMES:
                if tgt == src:
                    continue
                L.append(f"  {pa} on {tgt}: {_fmt(hmcc_df.loc[pa, tgt])}  "
                         f"vs  {sc} on {tgt}: {_fmt(hmcc_df.loc[sc, tgt])}  "
                         f"Δ={_delta(hmcc_df.loc[pa, tgt], hmcc_df.loc[sc, tgt])}")

        # ── key numbers for paper ──
        L.append("\n── Key Numbers for Paper ──")
        pa_in_all  = [hmcc_df.loc[f"paced-{b}", b] for b in TEST_BIOMES]
        sc_in_all  = [hmcc_df.loc[f"scratch-{b}", b] for b in TEST_BIOMES]
        sc_out_all = [hmcc_df.loc[f"scratch-{s}", t] for s in TEST_BIOMES for t in TEST_BIOMES if t != s]
        pa_out_all = [hmcc_df.loc[f"paced-{s}", t] for s in TEST_BIOMES for t in TEST_BIOMES if t != s]

        L.append(f"  PACED avg in-domain HMCC       : {_fmt(float(np.nanmean(pa_in_all)))}")
        L.append(f"  Scratch avg in-domain HMCC      : {_fmt(float(np.nanmean(sc_in_all)))}")
        L.append(f"  PACED avg transfer HMCC         : {_fmt(float(np.nanmean(pa_out_all)))}")
        L.append(f"  Scratch avg transfer HMCC       : {_fmt(float(np.nanmean(sc_out_all)))}")
        L.append(f"  PACED transfer advantage        : {_delta(float(np.nanmean(pa_out_all)), float(np.nanmean(sc_out_all)))}")
        L.append(f"  EU avg HMCC (all biomes)        : {_fmt(float(np.nanmean(eu_vals)))}")

        with open(path, 'w') as f:
            f.write('\n'.join(L))
        print(f"\nReport written: {path}")
        return path

    # ── viridis matrix figure ───────────────────────────────────────────────

    def plot_viridis_matrix(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        df     = self._df('hmcc')
        values = df.values.astype(float)

        finite = values[np.isfinite(values)]
        if not finite.size:
            print("No finite values — skipping figure")
            return

        vmin, vmax = finite.min(), finite.max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis

        n_rows, n_cols = values.shape
        fig, ax = plt.subplots(figsize=(7, 9), dpi=300)
        ax.imshow(values, cmap=cmap, norm=norm, aspect='auto')

        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels([b.capitalize() for b in TEST_BIOMES], fontsize=11)
        ax.set_yticklabels(MODEL_ROWS, fontsize=9)

        # Group separator lines (between EU / Spain pair / Poland pair / Finland pair)
        prev_grp = None
        for i, m in enumerate(MODEL_ROWS):
            grp = m.split('-')[0] if '-' in m else m
            if prev_grp and grp != prev_grp:
                ax.axhline(i - 0.5, color='white', linewidth=2.0)
            prev_grp = grp

        # Cell annotations
        col_max = np.nanargmax(values, axis=0)
        mid     = vmin + (vmax - vmin) * 0.55

        for i in range(n_rows):
            for j in range(n_cols):
                val   = values[i, j]
                label = 'N/A' if not np.isfinite(val) else f"{val:.3f}"

                m    = MODEL_ROWS[i]
                b    = TEST_BIOMES[j]
                grp  = m.split('-')[1] if '-' in m else None
                is_diagonal  = (grp == b)
                is_col_best  = (i == col_max[j])
                is_eu        = (m == 'eu')

                txt_color = 'white' if (np.isfinite(val) and val < mid) else 'black'
                weight    = 'bold' if (is_diagonal or (is_eu and is_col_best)) else 'normal'
                size      = 10 if (is_diagonal or (is_eu and is_col_best)) else 8

                if is_eu and is_col_best:
                    t = ax.text(j, i, label, ha='center', va='center',
                                color='#FFD700', fontsize=size, fontweight='bold')
                    t.set_path_effects([pe.withStroke(linewidth=1.5, foreground='black')])
                else:
                    ax.text(j, i, label, ha='center', va='center',
                            color=txt_color, fontsize=size, fontweight=weight)

        # Minor grid
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.8)
        ax.tick_params(which='minor', bottom=False, left=False)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label('HMCC', fontsize=10)

        ax.set_title('Biome Transfer Matrix — HMCC\n'
                     'harmonic mean of MCC(with-refl) and MCC(no-refl)',
                     fontweight='bold', fontsize=10, pad=10)
        ax.set_xlabel('Test Biome', fontsize=10)
        ax.set_ylabel('Model', fontsize=10)
        fig.tight_layout()

        path = os.path.join(output_dir, 'biome_matrix_hmcc.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved: {path}")
        return path

    # ── csv export ─────────────────────────────────────────────────────────

    def save_csvs(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for metric in ('hmcc', 'mcc_refl', 'mcc_norefl'):
            p = os.path.join(output_dir, f'biome_matrix_{metric}.csv')
            self._df(metric).to_csv(p)
            print(f"Saved: {p}")


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Biome transfer matrix evaluation')
    parser.add_argument('--eval_root', required=True,
                        help='Root data directory containing spain_eval/, poland_eval/, finland_eval/ subdirs')
    parser.add_argument('--models_dir', default='./model',
                        help='Directory containing trained .pth files')
    parser.add_argument('--output_dir', default='./reports/biome_matrix',
                        help='Output directory for report, CSVs, and figure')
    args = parser.parse_args()

    evaluator = BiomeMatrixEvaluator(args.eval_root, args.models_dir)
    evaluator.run_full_evaluation()
    evaluator.write_paper_report(args.output_dir)
    evaluator.save_csvs(args.output_dir)
    evaluator.plot_viridis_matrix(args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
