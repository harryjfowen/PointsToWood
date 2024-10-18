import numpy as np
from scipy.spatial import cKDTree
from collections import deque
from src.io import load_file, save_file
import argparse
import os 

class EuclideanCluster:
    def __init__(self, cluster_tolerance, min_cluster_size, max_cluster_size=np.inf):
        self.cluster_tolerance = cluster_tolerance  # This is our epsilon
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

    def cluster(self, points):
        tree = cKDTree(points)
        clusters = []
        processed = set()

        for i in range(len(points)):
            if i in processed:
                continue

            cluster = self._grow_cluster(i, tree, points, processed)
            if self.min_cluster_size <= len(cluster) <= self.max_cluster_size:
                clusters.append(cluster)

        labels = np.full(len(points), -1, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            labels[cluster] = cluster_id

        return labels

    def _grow_cluster(self, seed_point_idx, tree, points, processed):
        cluster = []
        queue = deque([seed_point_idx])

        while queue:
            point_idx = queue.popleft()
            if point_idx in processed:
                continue

            processed.add(point_idx)
            cluster.append(point_idx)

            neighbors = tree.query_ball_point(points[point_idx], self.cluster_tolerance)
            queue.extend([idx for idx in neighbors if idx not in processed])

        return cluster
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Euclidean Clustering on a point cloud file.")
    parser.add_argument("input_file", help="Path to the input point cloud file")
    parser.add_argument("--cluster_tolerance", type=float, default=0.1, help="Cluster tolerance (epsilon)")
    parser.add_argument("--min_cluster_size", type=int, default=10, help="Minimum cluster size")
    parser.add_argument("--max_cluster_size", type=int, default=10000, help="Maximum cluster size")
    args = parser.parse_args()

    # Load the point cloud file
    pc_data, headers = load_file(filename=args.input_file, additional_headers=True, verbose=True)

    # Extract XYZ coordinates from the loaded data
    # Adjust this based on your data structure
    points = pc_data[['x', 'y', 'z']].values

    # Perform clustering
    clusterer = EuclideanCluster(
        cluster_tolerance=args.cluster_tolerance,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size
    )
    labels = clusterer.cluster(points)

    print(f"Number of clusters: {len(np.unique(labels[labels != -1]))}")
    print(f"Number of noise points: {np.sum(labels == -1)}")

    # Add cluster_id as a new column to the point cloud data
    pc_data['cluster_id'] = labels

    # Prepare headers for saving
    headers = list(dict.fromkeys(headers + ['cluster_id']))

    # Generate output filename
    input_dir = os.path.dirname(args.input_file)
    base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_filename = os.path.join(input_dir, f"{base_filename}_clustered.ply")

    # Save the clustered point cloud
    save_file(output_filename, pc_data.copy(), additional_fields=headers, verbose=True)

    print(f"Clustered point cloud saved to: {output_filename}")