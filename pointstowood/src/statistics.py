def calculate_harmonic_metrics(metrics_with_refl, metrics_no_refl):
    """Calculate harmonic means and performance gaps for key metrics."""
    
    # Extract metrics
    acc_with_refl = metrics_with_refl['accuracy']
    acc_no_refl = metrics_no_refl['accuracy']
    fbeta_with_refl = metrics_with_refl['fbeta']
    fbeta_no_refl = metrics_no_refl['fbeta']
    
    # Calculate gaps
    acc_gap = abs(acc_with_refl - acc_no_refl)
    fbeta_gap = abs(fbeta_with_refl - fbeta_no_refl)
    
    # Calculate harmonic means
    harmonic_acc = 2 * (acc_with_refl * acc_no_refl) / (acc_with_refl + acc_no_refl) if (acc_with_refl + acc_no_refl) > 0 else 0
    harmonic_fbeta = 2 * (fbeta_with_refl * fbeta_no_refl) / (fbeta_with_refl + fbeta_no_refl) if (fbeta_with_refl + fbeta_no_refl) > 0 else 0
    
    return {
        'acc_with_refl': acc_with_refl,
        'acc_no_refl': acc_no_refl,
        'fbeta_with_refl': fbeta_with_refl,
        'fbeta_no_refl': fbeta_no_refl,
        'acc_gap': acc_gap,
        'fbeta_gap': fbeta_gap,
        'harmonic_acc': harmonic_acc,
        'harmonic_fbeta': harmonic_fbeta
    }


def print_validation_summary(epoch, harmonic_metrics):
    """Print a formatted validation summary table."""
    
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY - Epoch {epoch}")
    print(f"{'='*70}")
    print(f"{'Metric':<12} {'With Refl':<10} {'No Refl':<10} {'Gap':<8} {'Harmonic':<10}")
    print(f"{'-'*70}")
    print(f"{'Accuracy':<12} {harmonic_metrics['acc_with_refl']:<10.4f} {harmonic_metrics['acc_no_refl']:<10.4f} {harmonic_metrics['acc_gap']:<8.4f} {harmonic_metrics['harmonic_acc']:<10.4f}")
    print(f"{'Fbeta':<12} {harmonic_metrics['fbeta_with_refl']:<10.4f} {harmonic_metrics['fbeta_no_refl']:<10.4f} {harmonic_metrics['fbeta_gap']:<8.4f} {harmonic_metrics['harmonic_fbeta']:<10.4f}")
    print(f"{'='*70}")


def update_test_metrics_with_harmonic(test_metrics, harmonic_metrics):
    """Add harmonic metrics to test_metrics dict for logging."""
    
    test_metrics['harmonic_acc'] = harmonic_metrics['harmonic_acc']
    test_metrics['harmonic_fbeta'] = harmonic_metrics['harmonic_fbeta']
    test_metrics['acc_gap'] = harmonic_metrics['acc_gap']
    test_metrics['fbeta_gap'] = harmonic_metrics['fbeta_gap']
    test_metrics['acc_no_refl'] = harmonic_metrics['acc_no_refl']
    test_metrics['fbeta_no_refl'] = harmonic_metrics['fbeta_no_refl']
    test_metrics['fbeta_with_refl'] = harmonic_metrics['fbeta_with_refl']
    
    return test_metrics