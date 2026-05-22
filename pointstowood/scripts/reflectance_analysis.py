#!/usr/bin/env python3
"""Statistical analysis of reflectance values across all PLY files in the data directories."""

import sys
import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.io import read_ply


def find_ply_files(base_dir):
    """Find all PLY files in data directories."""
    # Try pool first, then fall back to train/test/eval
    pool_dir = os.path.join(base_dir, "pool")
    if os.path.isdir(pool_dir):
        files = glob.glob(os.path.join(pool_dir, "**", "*.ply"), recursive=True)
        if files:
            print(f"Found {len(files)} PLY files in data/pool")
            return files

    files = []
    for subdir in ["train", "test", "eval", "validation"]:
        d = os.path.join(base_dir, subdir)
        if os.path.isdir(d):
            found = glob.glob(os.path.join(d, "**", "*.ply"), recursive=True)
            files.extend(found)
            if found:
                print(f"Found {len(found)} PLY files in data/{subdir}")

    if not files:
        # Try base dir itself
        files = glob.glob(os.path.join(base_dir, "**", "*.ply"), recursive=True)
        print(f"Found {len(files)} PLY files in data/ (recursive)")

    return sorted(files)


def get_reflectance_column(df):
    """Find reflectance column using the project's naming conventions."""
    canon_aliases = ['reflectance', 'refl', 'intensity']
    for col in df.columns:
        clean = col.lower().replace('scalar_', '')
        for alias in canon_aliases:
            if alias in clean:
                return col
    return None


def get_site_prefix(filepath):
    """Extract site prefix from filename (e.g., gbr, deu, pol, esp, fin)."""
    basename = os.path.basename(filepath).lower()
    # Try common 3-letter country prefixes
    known_prefixes = ['gbr', 'deu', 'pol', 'esp', 'fin', 'aus', 'bra', 'idn',
                      'mys', 'guf', 'per', 'cog', 'gab', 'cmr', 'guy', 'uga']
    for prefix in known_prefixes:
        if basename.startswith(prefix):
            return prefix.upper()

    # Try first 3 chars as fallback
    parts = basename.split('_')
    if parts:
        candidate = parts[0][:3]
        if candidate.isalpha():
            return candidate.upper()

    return os.path.basename(os.path.dirname(filepath)) or "UNKNOWN"


def analyze_reflectance(values):
    """Compute comprehensive statistics for a reflectance array."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return None

    percentiles = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    pvals = np.percentile(values, percentiles)

    q25, q75 = pvals[3], pvals[5]
    iqr = q75 - q25
    extreme_upper = q75 + 3 * iqr
    n_extreme = int(np.sum(values > extreme_upper))

    p99 = pvals[7]
    n_beyond_p99 = int(np.sum(values > p99))

    # Check if integer-scaled
    unique_sample = np.unique(values[:min(50000, len(values))])
    is_integer = np.all(unique_sample == np.floor(unique_sample))
    n_unique = len(np.unique(values[:min(100000, len(values))]))

    # Determine likely scale
    vmax = np.max(values)
    vmin = np.min(values)
    if is_integer and vmax <= 255:
        scale_guess = "uint8 (0-255)"
    elif is_integer and vmax <= 65535:
        scale_guess = "uint16 (0-65535)"
    elif is_integer and vmax > 65535:
        scale_guess = f"large int (max={vmax:.0f})"
    elif abs(vmax) <= 1.0 and abs(vmin) <= 1.0:
        scale_guess = "float [-1,1]"
    elif abs(vmax) <= 100 and vmin >= -10:
        scale_guess = "float (small range)"
    else:
        scale_guess = f"float (range {vmin:.1f} to {vmax:.1f})"

    # Skewness and kurtosis
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val > 0:
        skewness = float(np.mean(((values - mean_val) / std_val) ** 3))
        kurtosis = float(np.mean(((values - mean_val) / std_val) ** 4)) - 3  # excess kurtosis
    else:
        skewness = 0.0
        kurtosis = 0.0

    # Simple bimodality check: Ashman's D for two modes
    # Use histogram to detect multiple peaks
    hist, bin_edges = np.histogram(values, bins=100)
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.05 * np.max(hist):
            peaks.append((bin_edges[i] + bin_edges[i+1]) / 2)
    n_peaks = len(peaks)

    stats = {
        'count': len(values),
        'min': float(vmin),
        'max': float(vmax),
        'mean': float(mean_val),
        'median': float(np.median(values)),
        'std': float(std_val),
        'skewness': skewness,
        'kurtosis': kurtosis,
        'iqr': float(iqr),
        'n_extreme_outliers': n_extreme,
        'pct_extreme': 100.0 * n_extreme / len(values),
        'p99_value': float(p99),
        'n_beyond_p99': n_beyond_p99,
        'ratio_beyond_p99': float(p99) / float(mean_val) if mean_val != 0 else float('inf'),
        'is_integer': is_integer,
        'n_unique': n_unique,
        'scale_guess': scale_guess,
        'n_histogram_peaks': n_peaks,
        'peak_locations': peaks[:5],  # first 5
    }

    for p, v in zip(percentiles, pvals):
        stats[f'p{p}'] = float(v)

    return stats


def main():
    base_dir = "/home/harryjfowen/PointsToWood/pointstowood/data"

    if not os.path.isdir(base_dir):
        print(f"ERROR: {base_dir} does not exist")
        # Try alternative paths
        for alt in ["/home/harryjfowen/data", "data"]:
            if os.path.isdir(alt):
                base_dir = alt
                print(f"Using alternative: {base_dir}")
                break
        else:
            print("No data directory found. Listing /home/harryjfowen/PointsToWood/pointstowood/:")
            try:
                for item in os.listdir("/home/harryjfowen/PointsToWood/pointstowood/"):
                    print(f"  {item}")
            except Exception as e:
                print(f"  Error listing: {e}")
            return

    print(f"\nSearching for PLY files in: {base_dir}")
    print("=" * 80)

    ply_files = find_ply_files(base_dir)
    if not ply_files:
        print("No PLY files found!")
        print(f"Contents of {base_dir}:")
        for item in sorted(os.listdir(base_dir)):
            full = os.path.join(base_dir, item)
            if os.path.isdir(full):
                n = len(glob.glob(os.path.join(full, "*.ply")))
                print(f"  {item}/ ({n} ply files)")
            else:
                print(f"  {item}")
        return

    # Analyze each file
    all_stats = []
    site_values = defaultdict(list)  # site -> list of all reflectance values (sampled)

    for fpath in sorted(ply_files):
        fname = os.path.basename(fpath)
        site = get_site_prefix(fpath)

        try:
            df = read_ply(fpath)
        except Exception as e:
            print(f"  SKIP {fname}: read error: {e}")
            continue

        refl_col = get_reflectance_column(df)
        if refl_col is None:
            print(f"  SKIP {fname}: no reflectance column. Columns: {list(df.columns)}")
            continue

        values = df[refl_col].values.astype(np.float64)
        stats = analyze_reflectance(values)
        if stats is None:
            print(f"  SKIP {fname}: all NaN reflectance")
            continue

        stats['file'] = fname
        stats['site'] = site
        stats['refl_col_name'] = refl_col
        stats['parent_dir'] = os.path.basename(os.path.dirname(fpath))
        all_stats.append(stats)

        # Sample up to 50k points per file for cross-site comparison
        sample_size = min(50000, len(values))
        sampled = np.random.choice(values, sample_size, replace=False) if len(values) > sample_size else values
        site_values[site].append(sampled)

    if not all_stats:
        print("No files with reflectance data found!")
        return

    # ==================== PER-FILE TABLE ====================
    print("\n" + "=" * 120)
    print("PER-FILE REFLECTANCE STATISTICS")
    print("=" * 120)

    header = f"{'File':<45} {'Site':>4} {'Count':>10} {'Min':>10} {'Max':>12} {'Mean':>10} {'Std':>10} {'Skew':>7} {'Kurt':>7} {'Scale':<20}"
    print(header)
    print("-" * len(header))

    for s in sorted(all_stats, key=lambda x: (x['site'], x['file'])):
        print(f"{s['file'][:44]:<45} {s['site']:>4} {s['count']:>10,} {s['min']:>10.2f} {s['max']:>12.2f} "
              f"{s['mean']:>10.2f} {s['std']:>10.2f} {s['skewness']:>7.2f} {s['kurtosis']:>7.2f} {s['scale_guess']:<20}")

    # ==================== PER-FILE PERCENTILES ====================
    print("\n" + "=" * 140)
    print("PER-FILE PERCENTILE TABLE")
    print("=" * 140)

    header2 = f"{'File':<40} {'Site':>4} {'p0.1':>10} {'p1':>10} {'p5':>10} {'p25':>10} {'p50':>10} {'p75':>10} {'p95':>10} {'p99':>10} {'p99.9':>10} {'IQR':>10} {'%Extreme':>8}"
    print(header2)
    print("-" * len(header2))

    for s in sorted(all_stats, key=lambda x: (x['site'], x['file'])):
        print(f"{s['file'][:39]:<40} {s['site']:>4} "
              f"{s['p0.1']:>10.2f} {s['p1']:>10.2f} {s['p5']:>10.2f} {s['p25']:>10.2f} "
              f"{s['p50']:>10.2f} {s['p75']:>10.2f} {s['p95']:>10.2f} {s['p99']:>10.2f} "
              f"{s['p99.9']:>10.2f} {s['iqr']:>10.2f} {s['pct_extreme']:>7.3f}%")

    # ==================== SITE-LEVEL SUMMARY ====================
    print("\n" + "=" * 120)
    print("SITE-LEVEL SUMMARY (aggregated across files)")
    print("=" * 120)

    sites = sorted(site_values.keys())
    site_summaries = {}

    for site in sites:
        combined = np.concatenate(site_values[site])
        ss = analyze_reflectance(combined)
        ss['n_files'] = sum(1 for s in all_stats if s['site'] == site)
        site_summaries[site] = ss

        print(f"\n--- {site} ({ss['n_files']} files, {ss['count']:,} points sampled) ---")
        print(f"  Range:        [{ss['min']:.2f}, {ss['max']:.2f}]")
        print(f"  Mean +/- Std: {ss['mean']:.4f} +/- {ss['std']:.4f}")
        print(f"  Median:       {ss['median']:.4f}")
        print(f"  Skewness:     {ss['skewness']:.4f}   Kurtosis: {ss['kurtosis']:.4f}")
        print(f"  IQR:          {ss['iqr']:.4f}")
        print(f"  Percentiles:  p1={ss['p1']:.2f}  p5={ss['p5']:.2f}  p25={ss['p25']:.2f}  "
              f"p50={ss['p50']:.2f}  p75={ss['p75']:.2f}  p95={ss['p95']:.2f}  p99={ss['p99']:.2f}")
        print(f"  Extreme outliers (>Q3+3*IQR): {ss['n_extreme_outliers']:,} ({ss['pct_extreme']:.3f}%)")
        print(f"  Ratio p99/mean: {ss['ratio_beyond_p99']:.2f}")
        print(f"  Scale guess:  {ss['scale_guess']}")
        print(f"  Integer?      {ss['is_integer']}  |  Unique values (in sample): {ss['n_unique']:,}")
        print(f"  Histogram peaks: {ss['n_histogram_peaks']} at {[f'{p:.1f}' for p in ss['peak_locations']]}")

    # ==================== CROSS-SITE COMPARISON ====================
    print("\n" + "=" * 120)
    print("CROSS-SITE COMPARISON")
    print("=" * 120)

    if len(sites) > 1:
        comp_header = f"{'Site':>5} {'N_files':>7} {'Min':>10} {'Max':>12} {'Mean':>10} {'Std':>10} {'Median':>10} {'IQR':>10} {'Skew':>7} {'Kurt':>7} {'%Extreme':>9} {'Scale':<20}"
        print(comp_header)
        print("-" * len(comp_header))

        for site in sites:
            ss = site_summaries[site]
            print(f"{site:>5} {ss['n_files']:>7} {ss['min']:>10.2f} {ss['max']:>12.2f} "
                  f"{ss['mean']:>10.2f} {ss['std']:>10.2f} {ss['median']:>10.2f} {ss['iqr']:>10.2f} "
                  f"{ss['skewness']:>7.2f} {ss['kurtosis']:>7.2f} {ss['pct_extreme']:>8.3f}% {ss['scale_guess']:<20}")

        # Coefficient of variation across site means
        site_means = [site_summaries[s]['mean'] for s in sites]
        site_stds = [site_summaries[s]['std'] for s in sites]
        cv_means = np.std(site_means) / np.mean(site_means) * 100 if np.mean(site_means) != 0 else float('inf')
        print(f"\n  Cross-site CV of means: {cv_means:.1f}%")
        print(f"  Site mean range: [{min(site_means):.2f}, {max(site_means):.2f}]")
        print(f"  Site std range:  [{min(site_stds):.2f}, {max(site_stds):.2f}]")

        # Range comparison
        site_ranges = {s: site_summaries[s]['max'] - site_summaries[s]['min'] for s in sites}
        print(f"  Dynamic range:   {dict(sorted(site_ranges.items(), key=lambda x: x[1]))}")

    # ==================== FLAGS AND WARNINGS ====================
    print("\n" + "=" * 120)
    print("FLAGS AND WARNINGS")
    print("=" * 120)

    flags = []

    for s in all_stats:
        fname = s['file']
        site = s['site']

        # Long tail
        if s['skewness'] > 2.0:
            flags.append(f"  LONG RIGHT TAIL: {fname} ({site}) skewness={s['skewness']:.2f}")
        elif s['skewness'] < -2.0:
            flags.append(f"  LONG LEFT TAIL:  {fname} ({site}) skewness={s['skewness']:.2f}")

        # Heavy tails (leptokurtic)
        if s['kurtosis'] > 6.0:
            flags.append(f"  HEAVY TAILS:     {fname} ({site}) excess_kurtosis={s['kurtosis']:.2f}")

        # Extreme outlier prevalence
        if s['pct_extreme'] > 5.0:
            flags.append(f"  MANY EXTREMES:   {fname} ({site}) {s['pct_extreme']:.1f}% beyond Q3+3*IQR")

        # Bimodal
        if s['n_histogram_peaks'] >= 2:
            flags.append(f"  MULTIMODAL:      {fname} ({site}) {s['n_histogram_peaks']} peaks at {[f'{p:.1f}' for p in s['peak_locations'][:5]]}")

        # Unusual range
        if s['max'] > 100000:
            flags.append(f"  LARGE RANGE:     {fname} ({site}) max={s['max']:.0f}")

        # All zeros or near-zero std
        if s['std'] < 1e-6:
            flags.append(f"  ZERO VARIANCE:   {fname} ({site}) std={s['std']:.2e}")

        # Negative values
        if s['min'] < 0:
            flags.append(f"  NEGATIVE VALUES: {fname} ({site}) min={s['min']:.4f}")

    if flags:
        for f in sorted(flags):
            print(f)
    else:
        print("  No flags raised.")

    # ==================== QUANTILE NORMALIZATION IMPACT ====================
    print("\n" + "=" * 120)
    print("QUANTILE NORMALIZATION RISK ASSESSMENT")
    print("=" * 120)

    print("\nPer-site assessment of what would break naive quantile normalization:")
    for site in sites:
        ss = site_summaries[site]
        risks = []

        if ss['skewness'] > 3.0:
            risks.append(f"very high skewness ({ss['skewness']:.1f}) - quantile mapping will be unstable in tails")
        if ss['kurtosis'] > 10.0:
            risks.append(f"extreme kurtosis ({ss['kurtosis']:.1f}) - sharp peaks cause quantile collisions")
        if ss['pct_extreme'] > 5.0:
            risks.append(f"{ss['pct_extreme']:.1f}% extreme outliers - tail quantiles unreliable")
        if ss['n_histogram_peaks'] >= 2:
            risks.append(f"multimodal ({ss['n_histogram_peaks']} peaks) - quantile normalization will distort modes")
        if ss['is_integer'] and ss['n_unique'] < 50:
            risks.append(f"only {ss['n_unique']} unique values - heavy quantile ties")

        if risks:
            print(f"\n  {site}: WARNING")
            for r in risks:
                print(f"    - {r}")
        else:
            print(f"\n  {site}: OK (no major concerns)")

    # Check cross-site compatibility
    if len(sites) > 1:
        scales = set(site_summaries[s]['scale_guess'] for s in sites)
        if len(scales) > 1:
            print(f"\n  CROSS-SITE: MIXED SCALES DETECTED: {scales}")
            print("    -> Naive quantile normalization across sites will fail without per-site rescaling first")
        else:
            print(f"\n  CROSS-SITE: Consistent scale ({scales.pop()})")

        # Check if distributions are comparable
        ranges_differ = max(site_summaries[s]['max'] - site_summaries[s]['min'] for s in sites) / \
                        max(1e-10, min(site_summaries[s]['max'] - site_summaries[s]['min'] for s in sites))
        if ranges_differ > 10:
            print(f"  CROSS-SITE: Dynamic range differs by {ranges_differ:.1f}x across sites")
            print("    -> Per-site quantile normalization recommended over global")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
