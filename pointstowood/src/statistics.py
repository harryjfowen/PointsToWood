_prev_summary_metrics: dict = {}
_best_summary_metrics: dict = {}


def reset_metric_trackers():
    """Reset epoch-over-epoch metric trackers. Call between training runs."""
    global _prev_summary_metrics, _best_summary_metrics
    _prev_summary_metrics = {}
    _best_summary_metrics = {}


def _metric_tag(name, value, higher_is_better=True, eps=1e-6):
    """Compact indicator: ↑/↓/→ vs previous."""
    prev = _prev_summary_metrics.get(name, None)
    best = _best_summary_metrics.get(name, None)

    if prev is None:
        trend = "·"
    else:
        if abs(value - prev) <= eps:
            trend = "→"
        elif (value > prev and higher_is_better) or (value < prev and not higher_is_better):
            trend = "↑"
        else:
            trend = "↓"

    if best is None:
        is_best = True
    else:
        is_best = (value > best + eps) if higher_is_better else (value < best - eps)

    if is_best:
        _best_summary_metrics[name] = value
    _prev_summary_metrics[name] = value

    return trend


def calculate_harmonic_metrics(metrics_with_refl, metrics_no_refl):
    """Calculate harmonic means and performance gaps for key metrics."""

    # Extract metrics
    acc_with_refl = metrics_with_refl['accuracy']
    acc_no_refl = metrics_no_refl['accuracy']
    fbeta_with_refl = metrics_with_refl['fbeta']
    fbeta_no_refl = metrics_no_refl['fbeta']
    auprc_with_refl = metrics_with_refl['auprc']
    auprc_no_refl = metrics_no_refl['auprc']

    # Edge metrics
    auprc_edge_with_refl = metrics_with_refl.get('auprc_edge', 0)
    auprc_edge_no_refl = metrics_no_refl.get('auprc_edge', 0)
    edge_ratio_with_refl = metrics_with_refl.get('edge_ratio', 0)
    edge_ratio_no_refl = metrics_no_refl.get('edge_ratio', 0)

    # Stratified mIoU metrics
    stratified_miou_with_refl = metrics_with_refl.get('stratified_miou', 0)
    stratified_miou_no_refl = metrics_no_refl.get('stratified_miou', 0)
    miou_pure_with_refl = metrics_with_refl.get('miou_pure', 0)
    miou_pure_no_refl = metrics_no_refl.get('miou_pure', 0)
    miou_edge_with_refl = metrics_with_refl.get('miou_edge', 0)
    miou_edge_no_refl = metrics_no_refl.get('miou_edge', 0)

    # MCC, FPR, Recall, Fbeta Edge, ASD Edge, Edge Coherency
    mcc_with_refl = metrics_with_refl.get('mcc', 0)
    mcc_no_refl = metrics_no_refl.get('mcc', 0)
    fpr_with_refl = metrics_with_refl.get('fpr', 0)
    fpr_no_refl = metrics_no_refl.get('fpr', 0)
    recall_with_refl = metrics_with_refl.get('recall', 0)
    recall_no_refl = metrics_no_refl.get('recall', 0)
    fbeta_edge_with_refl = metrics_with_refl.get('fbeta_edge', 0)
    fbeta_edge_no_refl = metrics_no_refl.get('fbeta_edge', 0)
    edge_asd_with_refl = metrics_with_refl.get('edge_asd', 1.0)
    edge_asd_no_refl = metrics_no_refl.get('edge_asd', 1.0)
    edge_coherency_with_refl = metrics_with_refl.get('edge_coherency', 0)
    edge_coherency_no_refl = metrics_no_refl.get('edge_coherency', 0)

    # Stratified FPR (pure vs edge regions)
    fpr_pure_with_refl = metrics_with_refl.get('fpr_pure', 0)
    fpr_pure_no_refl = metrics_no_refl.get('fpr_pure', 0)
    fpr_edge_with_refl = metrics_with_refl.get('fpr_edge', 0)
    fpr_edge_no_refl = metrics_no_refl.get('fpr_edge', 0)

    # Brier Score (lower is better; mean is appropriate to represent worst-case calibration)
    brier_with_refl = metrics_with_refl.get('brier_score', float('nan'))
    brier_no_refl = metrics_no_refl.get('brier_score', float('nan'))
    mean_brier = (brier_with_refl + brier_no_refl) / 2

    # Calculate gaps
    acc_gap = abs(acc_with_refl - acc_no_refl)
    fbeta_gap = abs(fbeta_with_refl - fbeta_no_refl)
    auprc_gap = abs(auprc_with_refl - auprc_no_refl)

    # Calculate harmonic means
    harmonic_acc = 2 * (acc_with_refl * acc_no_refl) / (acc_with_refl + acc_no_refl) if (acc_with_refl + acc_no_refl) > 0 else 0
    harmonic_fbeta = 2 * (fbeta_with_refl * fbeta_no_refl) / (fbeta_with_refl + fbeta_no_refl) if (fbeta_with_refl + fbeta_no_refl) > 0 else 0
    harmonic_auprc = 2 * (auprc_with_refl * auprc_no_refl) / (auprc_with_refl + auprc_no_refl) if (auprc_with_refl + auprc_no_refl) > 0 else 0
    harmonic_fpr = 2 * (fpr_with_refl * fpr_no_refl) / (fpr_with_refl + fpr_no_refl) if (fpr_with_refl + fpr_no_refl) > 0 else max(fpr_with_refl, fpr_no_refl)

    # Edge Risk Score (lower is better): ASD * (1 + 2 * FPR)
    edge_risk_with_refl = edge_asd_with_refl * (1.0 + 2.0 * fpr_with_refl)
    edge_risk_no_refl = edge_asd_no_refl * (1.0 + 2.0 * fpr_no_refl)
    harmonic_edge_risk = (
        2 * (edge_risk_with_refl * edge_risk_no_refl) / (edge_risk_with_refl + edge_risk_no_refl)
        if (edge_risk_with_refl + edge_risk_no_refl) > 0
        else max(edge_risk_with_refl, edge_risk_no_refl)
    )

    # Harmonic edge coherency (higher is better): both modes must have good boundary quality
    harmonic_edge_coherency = (
        2 * (edge_coherency_with_refl * edge_coherency_no_refl) / (edge_coherency_with_refl + edge_coherency_no_refl)
        if (edge_coherency_with_refl + edge_coherency_no_refl) > 0 else 0
    )

    # Youden's J = Recall - FPR (balances finding wood vs avoiding false positives)
    youdens_j_with_refl = recall_with_refl - fpr_with_refl
    youdens_j_no_refl = recall_no_refl - fpr_no_refl
    # Harmonic mean of Youden's J (both modes must be good)
    harmonic_youdens_j = (
        2 * (youdens_j_with_refl * youdens_j_no_refl) / (youdens_j_with_refl + youdens_j_no_refl)
        if (youdens_j_with_refl + youdens_j_no_refl) > 0 else 0
    )

    # Harmonic MCC (both modes must be good)
    harmonic_mcc = (
        2 * (mcc_with_refl * mcc_no_refl) / (mcc_with_refl + mcc_no_refl)
        if (mcc_with_refl + mcc_no_refl) > 0 else 0
    )

    # H4MCC: 4-way harmonic MCC — the ultimate metric.
    # Forces the model to be good across all four conditions:
    #   with refl × pure,  with refl × edge,
    #   no refl  × pure,  no refl  × edge
    # Harmonic mean is dominated by the worst case (no-refl + edge).
    mcc_pure_with_refl = metrics_with_refl.get('mcc_pure', 0.0)
    mcc_edge_with_refl = metrics_with_refl.get('mcc_edge', 0.0)
    mcc_pure_no_refl   = metrics_no_refl.get('mcc_pure', 0.0)
    mcc_edge_no_refl   = metrics_no_refl.get('mcc_edge', 0.0)
    _h4_vals = [mcc_pure_with_refl, mcc_edge_with_refl, mcc_pure_no_refl, mcc_edge_no_refl]
    h4_mcc = (
        4.0 / sum(1.0 / max(v, 1e-8) for v in _h4_vals)
        if all(v > 0 for v in _h4_vals) else 0.0
    )

    return {
        'acc_with_refl': acc_with_refl,
        'acc_no_refl': acc_no_refl,
        'fbeta_with_refl': fbeta_with_refl,
        'fbeta_no_refl': fbeta_no_refl,
        'auprc_with_refl': auprc_with_refl,
        'auprc_no_refl': auprc_no_refl,
        'auprc_edge_with_refl': auprc_edge_with_refl,
        'auprc_edge_no_refl': auprc_edge_no_refl,
        'edge_ratio_with_refl': edge_ratio_with_refl,
        'edge_ratio_no_refl': edge_ratio_no_refl,
        'stratified_miou_with_refl': stratified_miou_with_refl,
        'stratified_miou_no_refl': stratified_miou_no_refl,
        'miou_pure_with_refl': miou_pure_with_refl,
        'miou_pure_no_refl': miou_pure_no_refl,
        'miou_edge_with_refl': miou_edge_with_refl,
        'miou_edge_no_refl': miou_edge_no_refl,
        'mcc_with_refl': mcc_with_refl,
        'mcc_no_refl': mcc_no_refl,
        'fpr_with_refl': fpr_with_refl,
        'fpr_no_refl': fpr_no_refl,
        'fbeta_edge_with_refl': fbeta_edge_with_refl,
        'fbeta_edge_no_refl': fbeta_edge_no_refl,
        'edge_asd_with_refl': edge_asd_with_refl,
        'edge_asd_no_refl': edge_asd_no_refl,
        'edge_coherency_with_refl': edge_coherency_with_refl,
        'edge_coherency_no_refl': edge_coherency_no_refl,
        'edge_risk_with_refl': edge_risk_with_refl,
        'edge_risk_no_refl': edge_risk_no_refl,
        'acc_gap': acc_gap,
        'fbeta_gap': fbeta_gap,
        'auprc_gap': auprc_gap,
        'harmonic_acc': harmonic_acc,
        'harmonic_fbeta': harmonic_fbeta,
        'harmonic_auprc': harmonic_auprc,
        'harmonic_fpr': harmonic_fpr,
        'harmonic_edge_risk': harmonic_edge_risk,
        'harmonic_edge_coherency': harmonic_edge_coherency,
        'recall_with_refl': recall_with_refl,
        'recall_no_refl': recall_no_refl,
        'youdens_j_with_refl': youdens_j_with_refl,
        'youdens_j_no_refl': youdens_j_no_refl,
        'harmonic_youdens_j': harmonic_youdens_j,
        'harmonic_mcc': harmonic_mcc,
        'mcc_pure_with_refl': mcc_pure_with_refl,
        'mcc_edge_with_refl': mcc_edge_with_refl,
        'mcc_pure_no_refl': mcc_pure_no_refl,
        'mcc_edge_no_refl': mcc_edge_no_refl,
        'h4_mcc': h4_mcc,
        'fpr_pure_with_refl': fpr_pure_with_refl,
        'fpr_pure_no_refl': fpr_pure_no_refl,
        'fpr_edge_with_refl': fpr_edge_with_refl,
        'fpr_edge_no_refl': fpr_edge_no_refl,
        'brier_with_refl': brier_with_refl,
        'brier_no_refl': brier_no_refl,
        'mean_brier': mean_brier,
    }


def print_validation_summary(epoch, harmonic_metrics):
    """Print a formatted validation summary table."""

    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY - Epoch {epoch}")
    print(f"{'='*70}")
    print(f"{'Metric':<22} {'With Refl':<12} {'No Refl':<12} {'H/Mean':<12}")
    print(f"{'-'*70}")
    print(f"{'MCC (pure)':<22} {harmonic_metrics['mcc_pure_with_refl']:<12.4f} {harmonic_metrics['mcc_pure_no_refl']:<12.4f}")
    print(f"{'MCC (edge)':<22} {harmonic_metrics['mcc_edge_with_refl']:<12.4f} {harmonic_metrics['mcc_edge_no_refl']:<12.4f}")
    print(f"{'MCC (global)':<22} {harmonic_metrics['mcc_with_refl']:<12.4f} {harmonic_metrics['mcc_no_refl']:<12.4f} {harmonic_metrics['harmonic_mcc']:<12.4f}")
    print(f"{'FPR (lower=better)':<22} {harmonic_metrics['fpr_with_refl']:<12.4f} {harmonic_metrics['fpr_no_refl']:<12.4f}")
    print(f"{'Brier (lower=better)':<22} {harmonic_metrics['brier_with_refl']:<12.4f} {harmonic_metrics['brier_no_refl']:<12.4f}")
    print(f"{'-'*70}")
    print(f"{'H4-MCC (save metric)':<22} {harmonic_metrics['h4_mcc']:.4f}")
    print(f"{'-'*70}")
    tag_h4 = _metric_tag("h4_mcc", harmonic_metrics['h4_mcc'], higher_is_better=True)
    tag_hfpr = _metric_tag("harmonic_fpr", harmonic_metrics['harmonic_fpr'], higher_is_better=False)
    tag_brier = _metric_tag("mean_brier", harmonic_metrics['mean_brier'], higher_is_better=False)
    print(f"Trend: H4MCC {tag_h4} | HFPR {tag_hfpr} | Brier {tag_brier}")
    print(f"{'='*70}")


def update_test_metrics_with_harmonic(test_metrics, harmonic_metrics):
    """Add harmonic metrics to test_metrics dict for logging."""

    test_metrics['harmonic_acc'] = harmonic_metrics['harmonic_acc']
    test_metrics['harmonic_fbeta'] = harmonic_metrics['harmonic_fbeta']
    test_metrics['harmonic_auprc'] = harmonic_metrics['harmonic_auprc']
    test_metrics['harmonic_fpr'] = harmonic_metrics['harmonic_fpr']
    test_metrics['harmonic_edge_risk'] = harmonic_metrics['harmonic_edge_risk']
    test_metrics['acc_gap'] = harmonic_metrics['acc_gap']
    test_metrics['fbeta_gap'] = harmonic_metrics['fbeta_gap']
    test_metrics['auprc_gap'] = harmonic_metrics['auprc_gap']
    test_metrics['acc_no_refl'] = harmonic_metrics['acc_no_refl']
    test_metrics['fbeta_no_refl'] = harmonic_metrics['fbeta_no_refl']
    test_metrics['fbeta_with_refl'] = harmonic_metrics['fbeta_with_refl']
    test_metrics['auprc_no_refl'] = harmonic_metrics['auprc_no_refl']
    test_metrics['auprc_with_refl'] = harmonic_metrics['auprc_with_refl']
    test_metrics['auprc_edge_with_refl'] = harmonic_metrics['auprc_edge_with_refl']
    test_metrics['auprc_edge_no_refl'] = harmonic_metrics['auprc_edge_no_refl']
    test_metrics['edge_ratio_with_refl'] = harmonic_metrics['edge_ratio_with_refl']
    test_metrics['edge_ratio_no_refl'] = harmonic_metrics['edge_ratio_no_refl']
    test_metrics['stratified_miou_with_refl'] = harmonic_metrics['stratified_miou_with_refl']
    test_metrics['stratified_miou_no_refl'] = harmonic_metrics['stratified_miou_no_refl']
    test_metrics['miou_pure_with_refl'] = harmonic_metrics['miou_pure_with_refl']
    test_metrics['miou_pure_no_refl'] = harmonic_metrics['miou_pure_no_refl']
    test_metrics['miou_edge_with_refl'] = harmonic_metrics['miou_edge_with_refl']
    test_metrics['miou_edge_no_refl'] = harmonic_metrics['miou_edge_no_refl']
    test_metrics['mcc_with_refl'] = harmonic_metrics['mcc_with_refl']
    test_metrics['mcc_no_refl'] = harmonic_metrics['mcc_no_refl']
    test_metrics['fpr_with_refl'] = harmonic_metrics['fpr_with_refl']
    test_metrics['fpr_no_refl'] = harmonic_metrics['fpr_no_refl']
    test_metrics['fbeta_edge_with_refl'] = harmonic_metrics['fbeta_edge_with_refl']
    test_metrics['fbeta_edge_no_refl'] = harmonic_metrics['fbeta_edge_no_refl']
    test_metrics['edge_asd_with_refl'] = harmonic_metrics['edge_asd_with_refl']
    test_metrics['edge_asd_no_refl'] = harmonic_metrics['edge_asd_no_refl']
    test_metrics['edge_coherency_with_refl'] = harmonic_metrics['edge_coherency_with_refl']
    test_metrics['edge_coherency_no_refl'] = harmonic_metrics['edge_coherency_no_refl']
    test_metrics['edge_risk_with_refl'] = harmonic_metrics['edge_risk_with_refl']
    test_metrics['edge_risk_no_refl'] = harmonic_metrics['edge_risk_no_refl']
    test_metrics['harmonic_edge_coherency'] = harmonic_metrics['harmonic_edge_coherency']
    test_metrics['recall_with_refl'] = harmonic_metrics['recall_with_refl']
    test_metrics['recall_no_refl'] = harmonic_metrics['recall_no_refl']
    test_metrics['youdens_j_with_refl'] = harmonic_metrics['youdens_j_with_refl']
    test_metrics['youdens_j_no_refl'] = harmonic_metrics['youdens_j_no_refl']
    test_metrics['harmonic_youdens_j'] = harmonic_metrics['harmonic_youdens_j']
    test_metrics['harmonic_mcc'] = harmonic_metrics['harmonic_mcc']
    test_metrics['h4_mcc'] = harmonic_metrics['h4_mcc']
    test_metrics['mcc_pure_with_refl'] = harmonic_metrics['mcc_pure_with_refl']
    test_metrics['mcc_edge_with_refl'] = harmonic_metrics['mcc_edge_with_refl']
    test_metrics['mcc_pure_no_refl'] = harmonic_metrics['mcc_pure_no_refl']
    test_metrics['mcc_edge_no_refl'] = harmonic_metrics['mcc_edge_no_refl']
    test_metrics['fpr_pure_with_refl'] = harmonic_metrics['fpr_pure_with_refl']
    test_metrics['fpr_pure_no_refl'] = harmonic_metrics['fpr_pure_no_refl']
    test_metrics['fpr_edge_with_refl'] = harmonic_metrics['fpr_edge_with_refl']
    test_metrics['fpr_edge_no_refl'] = harmonic_metrics['fpr_edge_no_refl']
    test_metrics['brier_with_refl'] = harmonic_metrics['brier_with_refl']
    test_metrics['brier_no_refl'] = harmonic_metrics['brier_no_refl']
    test_metrics['mean_brier'] = harmonic_metrics['mean_brier']

    return test_metrics
