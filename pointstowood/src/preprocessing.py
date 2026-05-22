import torch
import glob
import os
import sys
import numpy as np
from tqdm import tqdm

from src.utils import (
    clear_gpu_memory,
    quantile_normalize_reflectance,
    downsample_points,
    downsample_points_max,
    create_point_grid,
    create_point_grid_with_overlap,
)
from src.memory_utils import should_use_cpu_preprocessing


SHARD_FILENAME = 'shard.pt'


def _write_shard(vxpath: str, entries: list):
    """Write all voxels into a single shard file for fast loading.

    Format: list of dicts, each with 'point_cloud' (Tensor), 'grid_size' (float|None),
    and 'name' (str). One torch.save call replaces thousands of individual file writes.
    """
    path = os.path.join(vxpath, SHARD_FILENAME)
    torch.save(entries, path)
    print(f"  Shard        {len(entries)} voxels → {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def _tqdm_label(text: str) -> str:
    if sys.stdout.isatty():
        return f"  \033[94m{text}\033[0m"
    return f"  {text}"


def _sor_voxel(voxel_xyz_np, k=10, std_mult=1.0):
    """Per-voxel SOR: filter points with anomalously high k-NN distances.

    Returns bool mask (keep inliers). Uses pykdtree for speed.
    """
    n = voxel_xyz_np.shape[0]
    if n < k + 2:
        return np.ones(n, dtype=bool)

    try:
        from pykdtree.kdtree import KDTree
    except ImportError:
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree', n_jobs=1).fit(voxel_xyz_np)
            dists, _ = nn.kneighbors(voxel_xyz_np)
        except ImportError:
            return np.ones(n, dtype=bool)
    else:
        # Use pykdtree: faster than sklearn for this use case
        tree = KDTree(voxel_xyz_np)
        dists, _ = tree.query(voxel_xyz_np, k=k + 1)

    # Mean k-NN distance (exclude self at k=0)
    mean_d = np.mean(dists[:, 1:], axis=1)
    thresh = np.mean(mean_d) + std_mult * (np.std(mean_d) + 1e-8)
    return mean_d <= thresh


class Voxelise:
    def __init__(self, pos, vxpath, minpoints=512, maxpoints=9999999, gridsize=None, pointspacing=None, overlap: float = 0.0, grid_method: str = 'max', sor=False, sor_k=10, sor_std=1.0, file_prefix: str = None, output_mode: str = 'disk'):
        """
        Initialize the voxelization process.

        Args:
            pos (Tensor): Point cloud positions and optional reflectance
            vxpath (str): Output path for voxel files
            minpoints (int): Minimum points required per voxel
            maxpoints (int): Maximum points per voxel
            gridsize (List[float]): Final model voxel sizes in metres. If None,
                fallback library defaults [2.0, 4.0] are used.
            pointspacing (float, optional): Representative spacing for the
                pre-downsampling stage. If None or <= 0, each grid size uses an
                adaptive spacing of grid_size / 100.
            sor (bool): If True, run per-voxel SOR before writing (fast, local to each voxel)
            sor_k (int): SOR neighbours (default 10)
            sor_std (float): SOR threshold = mean + sor_std*std (default 1.0)
            file_prefix (str, optional): Prefix for output files (e.g., source filename)
            output_mode (str): 'disk' to write `.pt` files, 'memory' to keep voxel payloads in RAM
        """
        self.pos = pos
        self.vxpath = vxpath
        self.minpoints = minpoints
        self.maxpoints = maxpoints
        self.gridsize = list(gridsize) if gridsize is not None else [2.0, 4.0]
        self.overlap = overlap
        self.pointspacing = pointspacing
        self.requested_pointspacing = pointspacing
        self.grid_method = grid_method
        self.sor = sor
        self.sor_k = sor_k
        self.sor_std = sor_std
        self.file_prefix = file_prefix
        if output_mode not in {'disk', 'memory', 'disk_individual'}:
            raise ValueError(f"Unsupported output_mode '{output_mode}'")
        self.output_mode = output_mode
        self.in_memory_voxels = [] if output_mode == 'memory' else None
    
    def downsample(self, spacing: float):
        """Downsample onto a small representative grid before large block extraction."""
        if self.grid_method == 'max':
            return downsample_points_max(self.pos, spacing)
        return downsample_points(self.pos, spacing)

    def _resolve_point_spacing(self, grid_size: float) -> float:
        """Resolve the representative spacing for this grid-size pass."""
        if self.requested_pointspacing is not None and float(self.requested_pointspacing) > 0:
            return float(self.requested_pointspacing)
        return float(grid_size) / 100.0
    
    def grid(self):
        """Create final model voxels from the already downsampled point cloud."""
        return create_point_grid(
            self.pos,
            self.gridsize,
            min_points=self.minpoints,
            max_points=None,
        )
    
    def write_voxels(self):
        """Process the cloud into final model voxels and persist the payloads.

        Two modes:
        - overlap > 0: overlapping XY grid origins for smoother edge coverage
        - overlap == 0: Multi-resolution grids (original behavior)
        """
        original_point_count = int(len(self.pos))
        self.original_point_count = original_point_count
        if not isinstance(self.pos, torch.Tensor):
            # Adaptive device selection based on memory requirements
            point_count = len(self.pos)
            has_reflectance = self.pos.shape[1] > 3

            use_cpu, estimated_gpu_mem, available_gpu_mem = should_use_cpu_preprocessing(
                point_count, has_reflectance, self.gridsize
            )
            if self.output_mode == 'memory':
                use_cpu = True

            if use_cpu:
                device = 'cpu'
                mode_suffix = " | in-memory cache" if self.output_mode == 'memory' else ""
                print(f"  Device       CPU ({point_count:,} points, ~{estimated_gpu_mem:.1f} GB estimated{mode_suffix})")
            else:
                device = 'cuda'
                print(f"  Device       GPU (~{estimated_gpu_mem:.1f} GB estimated)")

            self.pos = torch.tensor(self.pos.values, dtype=torch.float).to(device=device)

        if self.file_prefix:
            file_counter = len(glob.glob(os.path.join(self.vxpath, f'{self.file_prefix}_voxel_*.pt')))
        else:
            file_counter = len(glob.glob(os.path.join(self.vxpath, 'voxel_*.pt')))

        # Load existing shard so voxels accumulate across multiple preprocess() calls
        # (each .ply file calls preprocess() separately; without this, only the last file's voxels survive)
        _prior_entries = []
        if self.output_mode == 'disk':
            _shard_path = os.path.join(self.vxpath, SHARD_FILENAME)
            if os.path.isfile(_shard_path):
                _prior_entries = torch.load(_shard_path, map_location='cpu', weights_only=False)
                file_counter = max(file_counter, len(_prior_entries))

        if self.overlap > 0:
            file_counter, stats = self._write_voxels_with_overlap(file_counter, _prior_entries)
        else:
            file_counter, stats = self._write_voxels_multi_resolution(file_counter, _prior_entries)

        clear_gpu_memory()
        if self.output_mode == 'memory':
            return self.in_memory_voxels
        return file_counter

    def _estimate_written_points(self, voxels, pos_cpu) -> int:
        written_points = 0
        for voxel_indices in voxels:
            if voxel_indices.size(0) == 0:
                continue
            voxel = pos_cpu[voxel_indices]
            if voxel.numel() == 0:
                continue
            voxel = voxel[~torch.isnan(voxel).any(dim=1)]
            n_points = int(voxel.size(0))
            if n_points < self.minpoints:
                continue
            written_points += min(n_points, self.maxpoints)
        return int(written_points)

    def _write_voxels_with_overlap(self, file_counter: int, prior_entries=None):
        """Write voxels using offset grid origins for edge coverage.

        Supports multiple grid sizes: each gets its own adaptive spacing
        and overlap offsets, giving both scale diversity and edge coverage.
        """
        num_offsets = int(self.overlap)
        original_pos = self.pos.clone()

        # Normalize reflectance once upfront
        reflectance_not_zero = original_pos.shape[1] > 3 and not torch.all(original_pos[:, 3] == 0)
        if reflectance_not_zero:
            original_pos[:, 3] = quantile_normalize_reflectance(original_pos[:, 3])

        kept_points_first = None
        written_points_total = 0
        all_prepared = list(prior_entries) if prior_entries else []

        for grid_size in self.gridsize:
            self.pos = original_pos.clone()

            # Representative spacing is resolved per scale; never mutate the
            # original user setting or later passes inherit the wrong spacing.
            spacing = self._resolve_point_spacing(grid_size)
            self.pos = self.downsample(spacing)

            voxels = create_point_grid_with_overlap(
                self.pos,
                grid_size,
                min_points=self.minpoints,
                max_points=None,
                num_offsets=num_offsets
            )

            print(f"  Voxel layout {grid_size:.1f} m grid | {num_offsets} offsets | {len(voxels):,} voxels")

            if self.pos.device.type == 'cpu':
                pos_cpu = self.pos.detach()
            else:
                pos_cpu = self.pos.detach().clone().to('cpu')

            kept_points = int(self.pos.size(0))
            if kept_points_first is None:
                kept_points_first = kept_points
            written_points_est = self._estimate_written_points(voxels, pos_cpu)
            kept_pct = (100.0 * kept_points / max(1, self.original_point_count))
            repeat_factor = (written_points_est / kept_points) if kept_points > 0 else 0.0
            print(f"  Downsample   {kept_points:,} unique points ({kept_pct:.1f}% of input)")
            print(f"  Overlap      {written_points_est:,} voxel-point copies ({repeat_factor:.1f}x)")
            file_counter, written_points, prepared = self._write_voxel_list(voxels, pos_cpu, file_counter, f'{grid_size}m overlap', grid_size=grid_size)
            written_points_total += int(written_points)
            all_prepared.extend(prepared)

            del voxels, pos_cpu

        if all_prepared:
            _write_shard(self.vxpath, all_prepared)

        del original_pos, self.pos
        return file_counter, {'kept_points': int(kept_points_first or 0), 'written_points': written_points_total}

    def _write_voxels_multi_resolution(self, file_counter: int, prior_entries=None):
        """Write voxels using multiple grid resolutions (original behavior)."""
        original_pos = self.pos.clone()

        # Normalize reflectance once upfront on full cloud (before downsampling for any grid size)
        reflectance_not_zero = original_pos.shape[1] > 3 and not torch.all(original_pos[:, 3] == 0)
        if reflectance_not_zero:
            original_pos[:, 3] = quantile_normalize_reflectance(original_pos[:, 3])

        kept_points_first = None
        written_points_total = 0
        all_prepared = list(prior_entries) if prior_entries else []

        for grid_size in self.gridsize:
            # Reset to normalized original before per-grid processing
            self.pos = original_pos.clone()

            # Spacing: resolution>0 = fixed (same for all scales); otherwise
            # adapt per scale so a 2 m pass uses 2 cm, 4 m uses 4 cm, etc.
            spacing = self._resolve_point_spacing(grid_size)
            self.pos = self.downsample(spacing)

            # Build voxels for this grid size only
            voxels = create_point_grid(self.pos, [grid_size], min_points=self.minpoints, max_points=None)

            # Only move to CPU if not already there
            if self.pos.device.type == 'cpu':
                pos_cpu = self.pos.detach()
            else:
                pos_cpu = self.pos.detach().clone().to('cpu')

            if kept_points_first is None:
                kept_points_first = int(self.pos.size(0))
            written_points_est = self._estimate_written_points(voxels, pos_cpu)
            kept_pct = (100.0 * self.pos.size(0) / max(1, self.original_point_count))
            repeat_factor = (written_points_est / max(1, self.pos.size(0)))
            print(f"  Downsample   {int(self.pos.size(0)):,} unique points ({kept_pct:.1f}% of input)")
            print(f"  Overlap      {written_points_est:,} voxel-point copies ({repeat_factor:.1f}x)")
            file_counter, written_points, prepared = self._write_voxel_list(voxels, pos_cpu, file_counter, f'{grid_size}m', grid_size=grid_size)
            written_points_total += int(written_points)
            all_prepared.extend(prepared)

            del voxels, pos_cpu

        if all_prepared:
            _write_shard(self.vxpath, all_prepared)

        del original_pos, self.pos
        return file_counter, {'kept_points': int(kept_points_first or 0), 'written_points': written_points_total}

    def _write_voxel_list(self, voxels, pos_cpu, file_counter: int, desc: str, grid_size: float = None):
        """Prepare voxels and collect into a list for batch writing.

        Returns prepared voxels as dicts; actual I/O happens in write_voxels()
        after all grid sizes are processed, writing a single shard file.
        """
        written_points = 0
        prepared = []

        print()
        for voxel_indices in tqdm(voxels, desc=_tqdm_label(f'Preparing {desc} voxels')):
            if voxel_indices.size(0) == 0:
                continue

            voxel = pos_cpu[voxel_indices]
            if voxel.numel() == 0:
                continue

            # Drop NaN rows
            voxel = voxel[~torch.isnan(voxel).any(dim=1)]
            if voxel.size(0) < self.minpoints:
                continue

            # Per-voxel SOR before subsampling (denoise full voxel, then cap)
            if self.sor:
                mask = _sor_voxel(voxel[:, :3].numpy(), k=self.sor_k, std_mult=self.sor_std)
                voxel = voxel[torch.from_numpy(mask)]
                if voxel.size(0) < self.minpoints:
                    continue

            # Single source of truth for final voxel capping: once the final
            # voxel membership is known, optionally subsample it here.
            # Uniform (without-replacement) so the cap doesn't bake absolute
            # brightness into the training set — AnisotropicConv learns from
            # local contrast, not per-point magnitude.
            if voxel.size(0) > self.maxpoints:
                sample_idx = torch.randperm(voxel.size(0))[:self.maxpoints]
                voxel = voxel[sample_idx]

            if self.file_prefix:
                name = f'{self.file_prefix}_voxel_{file_counter}'
            else:
                name = f'voxel_{file_counter}'

            # Per-voxel difficulty: fraction of points inside 25 cm mixed-label
            # voxels. Used by PointBudgetSampler as a sample-level weight so
            # hard samples (dense wood/leaf transitions, twig-in-leaves) are
            # seen more often. Column layout: xyz (0:3), reflectance (3), label (4).
            from src.pointcutmix import recompute_edge_scores
            try:
                pos_for_edge = voxel[:, :3]
                label_for_edge = voxel[:, 4]
                edge_scores = recompute_edge_scores(pos_for_edge, label_for_edge, batch=None, voxel_size=0.25)
                edge_fraction = float(edge_scores.mean().item()) if edge_scores.numel() > 0 else 0.0
            except Exception:
                edge_fraction = 0.0

            entry = {
                'point_cloud': voxel.clone(),
                'grid_size': float(grid_size) if grid_size is not None else None,
                'name': name,
                'edge_fraction': edge_fraction,
            }
            if self.output_mode == 'memory':
                self.in_memory_voxels.append(entry)
            elif self.output_mode == 'disk_individual':
                torch.save(entry, os.path.join(self.vxpath, f'{name}.pt'))
            else:
                prepared.append(entry)
            written_points += int(voxel.size(0))
            file_counter += 1

        return file_counter, written_points, prepared

def preprocess(args):
    """Process point cloud data based on command-line arguments."""
    maxpoints = args.max_pts if args.max_pts > 0 else 9999999  # 0 = no subsampling
    if getattr(args, 'in_memory', False):
        output_mode = 'memory'
    elif getattr(args, 'low_memory', False):
        output_mode = 'disk_individual'
    else:
        output_mode = 'disk'
    return Voxelise(
        args.pc,
        vxpath=args.vxfile,
        minpoints=args.min_pts,
        maxpoints=maxpoints,
        pointspacing=args.resolution,
        gridsize=args.grid_size,
        overlap=getattr(args, 'overlap', 0.0),
        grid_method=getattr(args, 'grid_method', 'max'),
        sor=getattr(args, 'sor', False),
        sor_k=getattr(args, 'sor_k', 10),
        sor_std=getattr(args, 'sor_std', 1.0),
        file_prefix=getattr(args, 'source_file_prefix', getattr(args, 'eval_source_file', None)),
        output_mode=output_mode,
    ).write_voxels()
