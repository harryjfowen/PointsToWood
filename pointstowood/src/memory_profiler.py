"""Memory profiling utilities for training."""

import torch
from collections import defaultdict
from typing import Dict, List, Tuple


class MemoryProfiler:
    """Track GPU memory usage at key training points."""

    def __init__(self, device='cuda'):
        self.device = device
        self.stats = defaultdict(list)
        self.peak_allocated = 0.0
        self.peak_reserved = 0.0

    def record(self, label: str) -> Tuple[float, float]:
        """Record memory at this point. Returns (allocated_GB, reserved_GB)."""
        if not torch.cuda.is_available():
            return 0.0, 0.0

        torch.cuda.synchronize(self.device)
        allocated_gb = torch.cuda.memory_allocated(self.device) / 1e9
        reserved_gb = torch.cuda.memory_reserved(self.device) / 1e9

        self.stats[label].append((allocated_gb, reserved_gb))
        self.peak_allocated = max(self.peak_allocated, allocated_gb)
        self.peak_reserved = max(self.peak_reserved, reserved_gb)

        return allocated_gb, reserved_gb

    def print_summary(self):
        """Print memory usage summary."""
        print("\n" + "="*70)
        print("MEMORY PROFILE SUMMARY")
        print("="*70)

        for label in sorted(self.stats.keys()):
            values = self.stats[label]
            if values:
                alloc_vals = [v[0] for v in values]
                avg_alloc = sum(alloc_vals) / len(alloc_vals)
                max_alloc = max(alloc_vals)
                print(f"  {label:30s} | Avg: {avg_alloc:6.2f}GB | Max: {max_alloc:6.2f}GB | Count: {len(values)}")

        print("-"*70)
        print(f"  {'Peak allocated':30s} | {self.peak_allocated:6.2f}GB")
        print(f"  {'Peak reserved':30s} | {self.peak_reserved:6.2f}GB")
        print("="*70 + "\n")

    def print_step(self, step: int, label: str = ""):
        """Print memory at current step (for inline monitoring)."""
        allocated_gb, reserved_gb = self.record(label if label else f"step_{step}")
        status = "⚠️ " if allocated_gb > 14.5 else "✓ "
        print(f"{status}[Step {step}] Allocated: {allocated_gb:5.2f}GB | Reserved: {reserved_gb:5.2f}GB | {label}")
        return allocated_gb, reserved_gb


def clear_memory():
    """Empty CUDA cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
