import numpy as np
from scipy.spatial import cKDTree
from collections import deque
from src.io import load_file, save_file
import argparse
import os 
from multiprocessing import Pool, RawArray
import ctypes

class EuclideanCluster:
    def __init__(self, cluster_tolerance, min_cluster_size, max_cluster_size=np.inf):
        self.cluster_tolerance = cluster_tolerance
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

    def cluster(self, points):
        n_points = len(points)
        tree = cKDTree(points)
        
        # Create shared memory array
        processed_array = RawArray(ctypes.c_bool, n_points)
        processed = np.frombuffer(processed_array, dtype=bool)

        with Pool(initializer=self._init_worker, initargs=(points, tree, processed_array)) as pool:
            clusters = pool.map(self._process_point, range(n_points))

        labels = np.full(n_points, -1, dtype=np.int32)
        for cluster_id, cluster in enumerate(filter(None, clusters)):
            labels[cluster] = cluster_id

        return labels

    @staticmethod
    def _init_worker(points, tree, processed_array):
        global shared_points, shared_tree, shared_processed
        shared_points = points
        shared_tree = tree
        shared_processed = np.frombuffer(processed_array, dtype=bool)

    def _process_point(self, i):
        if shared_processed[i]:
            return None
        cluster = self._grow_cluster(i)
        if self.min_cluster_size <= len(cluster) <= self.max_cluster_size:
            return cluster
        return None

    def _grow_cluster(self, seed_point_idx):
        cluster = []
        queue = deque([seed_point_idx])

        while queue:
            point_idx = queue.popleft()
            if shared_processed[point_idx]:
                continue

            shared_processed[point_idx] = True
            cluster.append(point_idx)

            neighbors = shared_tree.query_ball_point(shared_points[point_idx], self.cluster_tolerance)
            queue.extend([idx for idx in neighbors if not shared_processed[idx]])

        return cluster

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Euclidean Clustering on a point cloud file.")
    parser.add_argument("input_file", help="Path to the input point cloud file")
    parser.add_argument("--cluster_tolerance", type=float, default=0.1, help="Cluster tolerance (epsilon)")
    parser.add_argument("--min_cluster_size", type=int, default=10, help="Minimum cluster size")
    parser.add_argument("--max_cluster_size", type=int, default=10000000000, help="Maximum cluster size")
    args = parser.parse_args()

    pc_data, headers = load_file(filename=args.input_file, additional_headers=True, verbose=True)
    points = pc_data[['x', 'y', 'z']].values

    clusterer = EuclideanCluster(
        cluster_tolerance=args.cluster_tolerance,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size
    )
    labels = clusterer.cluster(points)

    print(f"Number of clusters: {len(np.unique(labels[labels != -1]))}")
    print(f"Number of noise points: {np.sum(labels == -1)}")

    pc_data['cluster_id'] = labels
    headers = list(dict.fromkeys(headers + ['cluster_id']))

    input_dir = os.path.dirname(args.input_file)
    base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_filename = os.path.join(input_dir, f"{base_filename}_clustered.ply")

    save_file(output_filename, pc_data.copy(), additional_fields=headers, verbose=True)
    print(f"Clustered point cloud saved to: {output_filename}")