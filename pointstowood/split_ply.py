import os
import numpy as np
from plyfile import PlyData, PlyElement

CHUNK_TO_SUBFOLDER = {1: "train", 2: "test", 3: "eval"}

def split_into_three_and_save(filepath, out_root: str | None = None):
    plydata = PlyData.read(filepath)
    
    x = plydata['vertex']['x']
    x = np.asarray(x)

    p80 = np.percentile(x, 80)
    p90 = np.percentile(x, 90)

    mask1 = x <= p80
    mask2 = (x > p80) & (x <= p90)
    mask3 = x > p90

    def make_chunk(mask):
        return PlyData(
            [PlyElement.describe(plydata['vertex'][mask], 'vertex')],
            text=False,
            byte_order='<'
        )

    if out_root is None:
        out_root = os.path.dirname(filepath)

    for sub in CHUNK_TO_SUBFOLDER.values():
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    chunk_paths = [
        os.path.join(out_root, CHUNK_TO_SUBFOLDER[idx], f"{base_name}_{idx}.ply")
        for idx in (1, 2, 3)
    ]
    for idx, mask in enumerate([mask1, mask2, mask3], start=1):
        chunk_data = make_chunk(mask)
        chunk_data.write(chunk_paths[idx - 1])
        print(f"Saved: {chunk_paths[idx - 1]}")

def process_folder(folder_path: str, out_root: str | None = None):
    ply_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]
    
    for ply_file in ply_files:
        full_path = os.path.join(folder_path, ply_file)
        split_into_three_and_save(full_path, out_root)

if __name__ == "__main__":
    import sys
    if len(sys.argv) not in (2, 3):
        print("Usage: python split_ply.py <folder_path> [output_root]")
        sys.exit(1)

    folder = sys.argv[1]
    out_root = sys.argv[2] if len(sys.argv) == 3 else None

    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    process_folder(folder, out_root)