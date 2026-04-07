import os
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

SPLITS = ["train", "test", "eval"]


def split_ply(filepath: str, split_dirs: dict, stem: str):
    """Split a single .ply file 80/10/10 along x-axis into train/test/eval."""
    plydata = PlyData.read(filepath)
    x = np.asarray(plydata['vertex']['x'])

    if len(x) == 0:
        raise ValueError(f"No vertices in {filepath}")

    p80 = np.percentile(x, 80)
    p90 = np.percentile(x, 90)

    masks = {
        "train": x <= p80,
        "test":  (x > p80) & (x <= p90),
        "eval":  x > p90,
    }

    for split_name, mask in masks.items():
        if not mask.any():
            print(f"  [{split_name}] WARNING: no points in split, skipping")
            continue
        chunk = PlyData(
            [PlyElement.describe(plydata['vertex'][mask], 'vertex')],
            text=False,
            byte_order='<',
        )
        out_dir = split_dirs[split_name]
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{stem}.ply")
        chunk.write(out_path)
        print(f"  [{split_name}] {int(mask.sum()):,} pts → {out_path}")


def collect_files(input_path: str) -> list:
    """Return sorted list of .ply files from a file or folder."""
    if os.path.isfile(input_path):
        if not input_path.lower().endswith('.ply'):
            raise ValueError(f"File must be a .ply: {input_path}")
        return [input_path]
    elif os.path.isdir(input_path):
        files = sorted(f for f in os.listdir(input_path) if f.lower().endswith('.ply'))
        if not files:
            raise FileNotFoundError(f"No .ply files found in: {input_path}")
        return [os.path.join(input_path, f) for f in files]
    else:
        raise FileNotFoundError(f"Path not found: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split .ply file(s) 80/10/10 along x-axis into train/test/eval subfolders.\n\n"
            "Outputs to {output}/train/, {output}/test/, {output}/eval/ — the global pool\n"
            "structure expected by train.py and distill.py.\n\n"
            "Example:\n"
            "  python split_ply.py /data/plots/ --prefix fin --output /path/to/pointstowood/data/\n"
            "  → data/train/fin01.ply, data/test/fin01.ply, data/eval/fin01.ply ..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'input', type=str,
        help='Path to a single .ply file or a folder of .ply files.',
    )
    parser.add_argument(
        '--prefix', type=str, default=None,
        help='Biome prefix for output filenames (e.g. "fin", "spa", "cam", "uk"). '
             'Files are numbered sequentially: fin01.ply, fin02.ply ... '
             'Must match the prefix used in --region when training/distilling. '
             'If omitted, the original filename stem is kept.',
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output root directory containing train/, test/, eval/ subfolders. '
             'Defaults to the input folder (or parent dir for a single file).',
    )
    args = parser.parse_args()

    files = collect_files(args.input)

    if args.output is not None:
        out_root = args.output
    elif os.path.isdir(args.input):
        out_root = args.input
    else:
        out_root = os.path.dirname(os.path.abspath(args.input))

    split_dirs = {s: os.path.join(out_root, s) for s in SPLITS}

    print(f"Input:   {args.input} ({len(files)} file{'s' if len(files) != 1 else ''})")
    print(f"Output:  {out_root}")
    if args.prefix:
        print(f"Prefix:  {args.prefix} → {args.prefix}01.ply, {args.prefix}02.ply ...")
    for s, d in split_dirs.items():
        print(f"  [{s}] → {d}/")
    print()

    for i, filepath in enumerate(files, start=1):
        stem = f"{args.prefix}{i:02d}" if args.prefix else os.path.splitext(os.path.basename(filepath))[0]
        print(f"[{i}/{len(files)}] {os.path.basename(filepath)} → {stem}.ply")
        split_ply(filepath, split_dirs, stem)

    print(f"\nDone. {len(files)} file(s) processed.")
    if args.prefix:
        print(f"\nNext step (preprocess + train/distill):")
        print(f"  python train.py   --region {args.prefix} --preprocess ...")
        print(f"  python distill.py --region {args.prefix} --preprocess ...")


if __name__ == "__main__":
    main()
