import os
import sys
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import networkx as nx
from sklearn.neighbors import NearestNeighbors

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.csgraph import connected_components

try:
    import hdbscan  # fast_hdbscan is alias in recent wheels
except ImportError:
    raise ImportError("hdbscan package is required: pip install hdbscan")

import skfmm

# Silence scikit-learn FutureWarning about force_all_finite rename
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*"
)

# -----------------------------------------------------------------------------
# Optional: reuse helper from utils.shortest_path if available
# -----------------------------------------------------------------------------
try:
    from utils.shortest_path import add_nodes  # type: ignore
except ImportError:
    def add_nodes(G, base_node, indices, distance, threshold):
        """Fallback minimal add_nodes implementation."""
        for idx, dist in zip(indices, distance):
            if dist <= threshold:
                G.add_weighted_edges_from([(base_node, idx, dist)])

# -----------------------------------------------------------------------------
# Graph builder (simplified version of array_to_graph) – always builds full
# graph; root only affects starting frontier but we will run multi-source
# Dijkstra later, so connectivity is what matters.
# -----------------------------------------------------------------------------

def build_graph(coords: np.ndarray,
                root: int,
                kpairs: int = 8,
                knn: int = 32,
                nbrs_threshold: float = 0.30,
                nbrs_threshold_step: float = 0.05,
                graph_threshold: float = np.inf) -> nx.Graph:
    """Build undirected weighted graph over points.

    Parameters are tuned for LiDAR forest plots (~cm units).
    """
    G = nx.Graph()

    all_idx = np.arange(coords.shape[0])
    remaining = all_idx.copy()

    nbrs = NearestNeighbors(n_neighbors=knn, metric='euclidean', n_jobs=-1)
    nbrs.fit(coords)
    dists_knn, idx_knn = nbrs.kneighbors(coords)

    current = [root]
    processed = [root]

    while remaining.size:
        if current:
            nn_idx = idx_knn[current]
            nn_dist = dists_knn[current]
            # remove processed
            mask = ~np.isin(nn_idx, processed)
            candidates = []
            for row_idx, (row_nn, row_dist, base) in enumerate(zip(nn_idx, nn_dist, current)):
                new_idx = row_nn[mask[row_idx]][:kpairs + 1]
                new_dist = row_dist[mask[row_idx]][:kpairs + 1]
                add_nodes(G, base, new_idx, new_dist, graph_threshold)
                candidates.extend(new_idx)
            current = np.unique(candidates).tolist()
        else:
            # try to connect remaining points to graph within threshold
            idx_remaining = idx_knn[remaining]
            dist_remaining = dists_knn[remaining]
            mask_graph = np.isin(idx_remaining, processed)
            mask_thresh = dist_remaining < nbrs_threshold
            connect_mask = mask_graph & mask_thresh
            unique_rows = np.unique(np.where(connect_mask)[0])
            current = remaining[unique_rows].tolist()
            # if still empty, relax threshold and try again next loop
            if not current:
                nbrs_threshold += nbrs_threshold_step
        processed = np.unique(np.concatenate([processed, current])).astype(int)
        remaining = all_idx[~np.isin(all_idx, processed)]
    return G

# -----------------------------------------------------------------------------
# Voxel down-sample helpers
# -----------------------------------------------------------------------------

def voxel_downsample(coords: np.ndarray, voxel: float):
    """Return coordinates of voxel centroids and mapping from original idx to voxel idx."""
    voxel_keys = np.floor(coords / voxel).astype(np.int64)
    key_to_voxel_idx = {}
    voxel_coords = []
    mapping = np.empty(coords.shape[0], dtype=np.int64)
    for i, key in enumerate(map(tuple, voxel_keys)):
        if key not in key_to_voxel_idx:
            vidx = len(voxel_coords)
            key_to_voxel_idx[key] = vidx
            voxel_coords.append(coords[i])
        mapping[i] = key_to_voxel_idx[key]
    return np.vstack(voxel_coords), mapping

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute path-length feature for forest point cloud (wood stems).")
    parser.add_argument("input_ply", help="Path to input PLY/CSV file loadable by pandas (columns x,y,z,label)")
    parser.add_argument("-o", "--output", dest="output_ply", default=None,
                        help="Output file; defaults to <input>_pl.<ext> alongside input")
    parser.add_argument("--voxel", type=float, default=0.05,
                        help="Voxel size (m) for down-sampling [default 0.05 = 5 cm]")
    parser.add_argument("--single-tree", action="store_true",
                        help="Use only the largest stem cluster instead of multiple bases.")
    parser.add_argument("--slice-low", type=float, default=0.3,
                        help="Lower z-offset above local min z inside each XY tile (m, default 0.3)")
    parser.add_argument("--slice-high", type=float, default=1.3,
                        help="Upper z-offset above local min z inside each XY tile (m, default 1.3)")
    parser.add_argument("--tile-size", type=float, default=5.0,
                        help="XY tile size for local slicing (m, default 1.0)")
    parser.add_argument("--erosion-iter", type=int, default=2,
                        help="Number of 3×3 minimum-filter iterations to propagate ground heights across empty or canopy-only tiles (default 2, set 0 to disable).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Derive default output path if not provided
    # ------------------------------------------------------------------
    if args.output_ply is None:
        base, ext = os.path.splitext(args.input_ply)
        args.output_ply = f"{base}_pl{ext or '.ply'}"
        print(f"No --output supplied; writing results to {args.output_ply}")

    # ------------------------------------------------------------------
    # Load file – expects DataFrame with at least x y z label
    # ------------------------------------------------------------------
    if args.input_ply.endswith('.ply'):
        # lazy import to avoid heavy deps if not needed
        try:
            from src.io import load_file  # project helper returns DataFrame
            df = load_file(args.input_ply)
        except Exception as e:
            print("Failed to load with src.io; attempting pandas read_csv", file=sys.stderr)
            df = pd.read_csv(args.input_ply)
    else:
        df = pd.read_csv(args.input_ply)

    required_cols = {'x', 'y', 'z'}
    if not required_cols.issubset(df.columns.str.lower().str.strip().tolist()):
        print("ERROR: input must have x,y,z columns.")
        sys.exit(1)
    # Normalize column names
    def clean_colname(c):
        c = c.lower().strip()
        if c.startswith('scalar_'):
            c = c[len('scalar_'):]
        return c
    df.rename(columns=clean_colname, inplace=True)

    if 'label' not in df.columns:
        if 'truth' in df.columns:
            print("No 'label' column found, using 'truth' as 'label'.")
            df['label'] = df['truth']
        else:
            print("ERROR: neither 'label' nor 'truth' column found – cannot identify wood points.")
            sys.exit(1)

    coords = df[['x', 'y', 'z']].values.astype(np.float32)

    # ------------------------------------------------------------------
    # Detect stems: slice by z then HDBSCAN clustering on wood points
    # ------------------------------------------------------------------
    wood_mask = df['label'].values == 1
    if not wood_mask.any():
        print("ERROR: no wood points (label==1) found; aborting.")
        sys.exit(1)

    wood_coords = coords[wood_mask]

    # ------------------------------------------------------------------
    # Tile-wise slicing: compute z_min within each XY tile of size --tile-size
    # and keep points whose z lies between (z_min+slice_low, z_min+slice_high)
    # ------------------------------------------------------------------
    if args.slice_high <= args.slice_low:
        print("ERROR: --slice-high must be greater than --slice-low", file=sys.stderr)
        sys.exit(1)

    tile_size = args.tile_size
    tile_keys = np.floor(coords[:, :2] / tile_size).astype(np.int64)  # use ALL points for DEM

    # Map tile keys to continuous grid indices
    kx = tile_keys[:, 0]
    ky = tile_keys[:, 1]
    gx_min, gy_min = kx.min(), ky.min()
    gx_max, gy_max = kx.max(), ky.max()
    grid_shape = (gx_max - gx_min + 1, gy_max - gy_min + 1)
    dem = np.full(grid_shape, np.inf, dtype=np.float32)

    # Fill DEM with per-tile minimum z across *all* points
    for z, ix, iy in zip(coords[:, 2], kx - gx_min, ky - gy_min):
        if z < dem[ix, iy]:
            dem[ix, iy] = z

    # Propagate heights across gaps/canopy tiles via iterative 3×3 minimum filter
    if args.erosion_iter > 0:
        from scipy.ndimage import minimum_filter
        high_val = np.nanmax(dem[np.isfinite(dem)]) + 100.0
        dem[np.isinf(dem)] = high_val
        for _ in range(args.erosion_iter):
            dem = minimum_filter(dem, size=3, mode="nearest")

    # Build lookup from tile key to baseline z
    baseline_z = {}
    for ix in range(grid_shape[0]):
        for iy in range(grid_shape[1]):
            key = (ix + gx_min, iy + gy_min)
            baseline_z[key] = dem[ix, iy]

    # Determine slice mask for wood points using the DEM baseline
    wood_tile_keys = np.floor(wood_coords[:, :2] / tile_size).astype(np.int64)
    slice_mask = np.zeros(wood_coords.shape[0], dtype=bool)
    for i, (ix, iy) in enumerate(wood_tile_keys):
        z_base = baseline_z[(ix, iy)]
        z_low = z_base + args.slice_low
        z_high = z_base + args.slice_high
        z_val = wood_coords[i, 2]
        if z_low <= z_val <= z_high:
            slice_mask[i] = True

    slice_coords = wood_coords[slice_mask]
    if slice_coords.shape[0] < 128:
        print("ERROR: not enough wood points in 3 m slice to cluster; aborting.")
        sys.exit(1)

    # Run HDBSCAN – we cluster in XY to avoid vertical spread
    clusterer = hdbscan.HDBSCAN(min_cluster_size=512, min_samples=64, metric='euclidean')
    cluster_labels = clusterer.fit_predict(slice_coords[:, :2])
    unique_clusters = [c for c in np.unique(cluster_labels) if c >= 0]
    if len(unique_clusters) == 0:
        print("ERROR: HDBSCAN found no clusters; aborting.")
        sys.exit(1)

    if args.single_tree:
        # Only use the largest cluster
        cluster_sizes = [(cid, np.sum(cluster_labels == cid)) for cid in unique_clusters]
        largest_cid = max(cluster_sizes, key=lambda x: x[1])[0]
        cluster_indices = np.where(cluster_labels == largest_cid)[0]
        cluster_slice_pts = slice_coords[cluster_indices]
        slice_global_idx = np.where(wood_mask)[0][slice_mask][cluster_indices]
        local_min_idx = cluster_slice_pts[:, 2].argmin()
        base_idx = slice_global_idx[local_min_idx]
        bases = [int(base_idx)]
        print(f"--single-tree: Using only the largest cluster (size {len(cluster_indices)}) as the stem base.")
    else:
        bases = []
        for cid in unique_clusters:
            cluster_indices = np.where(cluster_labels == cid)[0]
            cluster_slice_pts = slice_coords[cluster_indices]
            slice_global_idx = np.where(wood_mask)[0][slice_mask][cluster_indices]

            # 1. Find minimum z in cluster
            min_z = cluster_slice_pts[:, 2].min()
            # 2. Select all points within epsilon of min_z
            epsilon = 0.05  # 5 cm
            base_mask = np.abs(cluster_slice_pts[:, 2] - min_z) < epsilon
            base_points = cluster_slice_pts[base_mask]
            # 3. Compute mean or median x, y
            base_x = np.median(base_points[:, 0])
            base_y = np.median(base_points[:, 1])
            # 4. Use min_z as z
            base_z = min_z
            # 5. Find the closest point in the original cloud to (base_x, base_y, base_z)
            dists = np.linalg.norm(coords - np.array([base_x, base_y, base_z]), axis=1)
            base_idx = np.argmin(dists)
            bases.append(int(base_idx))
        bases = np.unique(bases).tolist()
        print(f"Detected {len(bases)} stem bases (tree roots) that will act as Dijkstra sources.")

    # ------------------------------------------------------------------
    # Down-sample for grid build
    # ------------------------------------------------------------------
    voxel_coords, mapping = voxel_downsample(coords, args.voxel)

    # Map base indices to voxel space
    base_voxels = np.unique(mapping[bases]).tolist()

    # ------------------------------------------------------------------
    # Fast Marching Method (FMM) for geodesic pathlength
    # ------------------------------------------------------------------
    print(f"Down-sampled to {voxel_coords.shape[0]} voxels; running Fast Marching Method…")

    # 1. Build a 3D grid covering your point cloud
    min_corner = voxel_coords.min(axis=0)
    max_corner = voxel_coords.max(axis=0)
    voxel_size = args.voxel

    grid_shape = np.ceil((max_corner - min_corner) / voxel_size).astype(int) + 1
    grid = np.zeros(grid_shape, dtype=bool)

    # 2. Map voxel_coords to grid indices
    grid_indices = np.round((voxel_coords - min_corner) / voxel_size).astype(int)
    for idx in grid_indices:
        grid[tuple(idx)] = True

    # 3. Mark source voxels (tree bases)
    source_mask = np.zeros_like(grid, dtype=bool)
    for base in base_voxels:
        idx = tuple(grid_indices[base])
        source_mask[idx] = True

    # 4. Prepare phi for FMM: negative inside sources, positive elsewhere
    phi = np.ones_like(grid, dtype=float)
    phi[source_mask] = -1

    # 5. Run FMM
    distance = skfmm.distance(phi, dx=voxel_size)
    distance[~grid] = np.nan  # Mask out empty voxels

    # 6. Assign distances back to each point
    voxel_distances = []
    for idx in grid_indices:
        d = distance[tuple(idx)]
        voxel_distances.append(d if not np.isnan(d) else 0.0)
    voxel_distances = np.array(voxel_distances, dtype=np.float32)
    point_distance = voxel_distances[mapping]
    df['pathlength'] = point_distance

    # ------------------------------------------------------------------
    # Save output – try src.io.save_file else CSV
    # ------------------------------------------------------------------
    saved = False
    try:
        from src.io import save_file  # type: ignore
        save_file(df, args.output_ply)
        saved = True
    except Exception:
        pass
    if not saved:
        try:
            from plyfile import PlyData, PlyElement  # type: ignore
            dtype_list = []
            for col in df.columns:
                if np.issubdtype(df[col].dtype, np.floating):
                    dtype_list.append((col, 'f4'))
                else:
                    dtype_list.append((col, 'i4'))
            vertex = np.empty(df.shape[0], dtype=dtype_list)
            for col in df.columns:
                vertex[col] = df[col].values
            el = PlyElement.describe(vertex, 'vertex')
            PlyData([el], text=True).write(args.output_ply)
            saved = True
        except Exception as e:
            print("plyfile save failed (" + str(e) + ") – falling back to CSV.")
    if not saved:
        df.to_csv(args.output_ply, index=False)
    print(f"Wrote pathlength-enriched file to {args.output_ply}")


if __name__ == "__main__":
    main() 