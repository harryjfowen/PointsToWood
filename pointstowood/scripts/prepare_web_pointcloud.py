#!/usr/bin/env python3
"""Build compact web point-cloud data from a binary little-endian PLY file.

Output:
- docs/data/pointcloud.bin        (float32 LE interleaved: x, y, z, prediction)
- docs/data/pointcloud.meta.json  (loader metadata)

Sampling policy:
- Keep wood points first (prediction >= wood_threshold), up to target_points.
- Use remaining budget for leaf points (prediction < wood_threshold).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from array import array
from pathlib import Path
from typing import Dict, List, Tuple


TYPE_INFO: Dict[str, Tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


def parse_header(blob: bytes):
    probe = blob[: min(len(blob), 1024 * 1024)]
    marker = b"end_header\n"
    idx = probe.find(marker)
    if idx < 0:
        marker = b"end_header\r\n"
        idx = probe.find(marker)
    if idx < 0:
        raise ValueError("Could not locate PLY end_header marker")

    header_bytes = idx + len(marker)
    header_text = blob[:header_bytes].decode("ascii", errors="strict")

    fmt = None
    vertex_count = None
    in_vertex = False
    properties: List[Dict[str, object]] = []

    for raw in header_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("format "):
            _, fmt, _ = line.split()
            continue
        if line.startswith("element "):
            _, name, count = line.split()
            in_vertex = name == "vertex"
            if in_vertex:
                vertex_count = int(count)
            continue
        if in_vertex and line.startswith("property "):
            toks = line.split()
            if toks[1] == "list":
                raise ValueError("Unsupported PLY: list vertex property encountered")
            p_type = toks[1]
            p_name = toks[2]
            if p_type not in TYPE_INFO:
                raise ValueError(f"Unsupported PLY property type: {p_type}")
            _, byte_size = TYPE_INFO[p_type]
            properties.append({"name": p_name, "type": p_type, "byte_size": byte_size})

    if fmt != "binary_little_endian":
        raise ValueError(f"Expected binary_little_endian PLY, got {fmt!r}")
    if not vertex_count or not properties:
        raise ValueError("Failed to parse vertex metadata from PLY header")

    offset = 0
    for prop in properties:
        prop["offset"] = offset
        offset += int(prop["byte_size"])

    return {
        "header_bytes": header_bytes,
        "vertex_count": vertex_count,
        "properties": properties,
        "stride": offset,
    }


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def hash01(index: int, seed: int) -> float:
    # Small deterministic hash in [0, 1).
    x = (index ^ seed) & 0xFFFFFFFF
    x ^= (x >> 16)
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= (x >> 15)
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= (x >> 16)
    return x / 4294967296.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="uk01_lw_pl_3_p2w.ply")
    parser.add_argument("--output-dir", default="docs/data")
    parser.add_argument("--target-points", type=int, default=260000)
    parser.add_argument("--wood-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_bin = output_dir / "pointcloud.bin"
    output_meta = output_dir / "pointcloud.meta.json"

    if args.target_points <= 0:
        raise ValueError("target-points must be > 0")
    if not input_path.exists():
        raise FileNotFoundError(f"Input PLY not found: {input_path}")

    blob = input_path.read_bytes()
    header = parse_header(blob)
    props = header["properties"]
    stride = int(header["stride"])
    vertex_count = int(header["vertex_count"])
    payload = memoryview(blob)[int(header["header_bytes"]) :]

    required_bytes = stride * vertex_count
    if len(payload) < required_bytes:
        raise ValueError("PLY payload appears truncated")

    def find_property(name: str) -> Dict[str, object]:
        for prop in props:
            if prop["name"] == name:
                return prop
        raise ValueError(f"PLY property not found: {name}")

    px = find_property("x")
    py = find_property("y")
    pz = find_property("z")
    pp = find_property("prediction")

    unpack_x = struct.Struct("<" + TYPE_INFO[str(px["type"])][0]).unpack_from
    unpack_y = struct.Struct("<" + TYPE_INFO[str(py["type"])][0]).unpack_from
    unpack_z = struct.Struct("<" + TYPE_INFO[str(pz["type"])][0]).unpack_from
    unpack_p = struct.Struct("<" + TYPE_INFO[str(pp["type"])][0]).unpack_from

    ox = int(px["offset"])
    oy = int(py["offset"])
    oz = int(pz["offset"])
    op = int(pp["offset"])

    wood_count = 0
    leaf_count = 0
    for i in range(vertex_count):
        row = i * stride
        pred = clamp01(float(unpack_p(payload, row + op)[0]))
        if pred >= args.wood_threshold:
            wood_count += 1
        else:
            leaf_count += 1

    target = min(args.target_points, vertex_count)
    target_wood = min(wood_count, target)
    target_leaf = max(0, target - target_wood)
    keep_wood = (target_wood / wood_count) if wood_count else 0.0
    keep_leaf = (target_leaf / leaf_count) if leaf_count else 0.0

    positions = array("f")
    wood_values = array("f")
    bins = [0] * 101

    min_x = float("inf")
    min_y = float("inf")
    min_z = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    max_z = float("-inf")
    kept_wood = 0
    kept_leaf = 0
    remaining_wood = wood_count
    remaining_leaf = leaf_count
    budget_wood = target_wood
    budget_leaf = target_leaf

    for i in range(vertex_count):
        row = i * stride
        pred = clamp01(float(unpack_p(payload, row + op)[0]))
        is_wood = pred >= args.wood_threshold
        if is_wood:
            if budget_wood <= 0:
                remaining_wood -= 1
                continue
            keep_prob = budget_wood / max(1, remaining_wood)
            pick = (keep_prob >= 1.0) or (hash01(i, args.seed) <= keep_prob)
            remaining_wood -= 1
            if not pick:
                continue
            budget_wood -= 1
        else:
            if budget_leaf <= 0:
                remaining_leaf -= 1
                continue
            keep_prob = budget_leaf / max(1, remaining_leaf)
            pick = (keep_prob >= 1.0) or (hash01(i, args.seed) <= keep_prob)
            remaining_leaf -= 1
            if not pick:
                continue
            budget_leaf -= 1

        x = float(unpack_x(payload, row + ox)[0])
        y = float(unpack_y(payload, row + oy)[0])
        z = float(unpack_z(payload, row + oz)[0])

        positions.extend((x, y, z))
        wood_values.append(pred)
        if is_wood:
            kept_wood += 1
        else:
            kept_leaf += 1

        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
        if z < min_z:
            min_z = z
        if z > max_z:
            max_z = z

        bin_idx = int(round(pred * 100.0))
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx > 100:
            bin_idx = 100
        bins[bin_idx] += 1

    kept = len(wood_values)
    if kept == 0:
        raise RuntimeError("No points selected. Increase target-points or lower wood-threshold.")

    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    center_z = (min_z + max_z) * 0.5
    scale = max(max_x - min_x, max_y - min_y, max_z - min_z) or 1.0

    for i in range(kept):
        base = i * 3
        positions[base] = (positions[base] - center_x) / scale
        positions[base + 1] = (positions[base + 1] - center_y) / scale
        positions[base + 2] = (positions[base + 2] - center_z) / scale

    packed = array("f")
    packed.extend([0.0] * (kept * 4))
    for i in range(kept):
        src = i * 3
        dst = i * 4
        packed[dst] = positions[src]
        packed[dst + 1] = positions[src + 1]
        packed[dst + 2] = positions[src + 2]
        packed[dst + 3] = wood_values[i]

    if sys.byteorder != "little":
        packed.byteswap()

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_bin.open("wb") as f:
        packed.tofile(f)

    meta = {
        "format": "xyzp_f32_le_v1",
        "binary": os.path.basename(output_bin),
        "pointCount": kept,
        "sourceVertexCount": vertex_count,
        "sourceFile": str(input_path),
        "woodThreshold": args.wood_threshold,
        "targetPoints": target,
        "targetWoodPoints": target_wood,
        "targetLeafPoints": target_leaf,
        "selection": {
            "sourceWood": wood_count,
            "sourceLeaf": leaf_count,
            "keptWood": kept_wood,
            "keptLeaf": kept_leaf,
            "keepWoodProbability": keep_wood,
            "keepLeafProbability": keep_leaf,
            "seed": args.seed,
        },
        "normalization": {
            "center": [center_x, center_y, center_z],
            "scale": scale,
        },
        "predictionHistogram101": bins,
    }
    output_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    size_mb = output_bin.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_bin} ({size_mb:.2f} MB)")
    print(f"Wrote {output_meta}")
    print(
        "Selected "
        f"{kept:,}/{vertex_count:,} points "
        f"(wood {kept_wood:,}/{wood_count:,}, leaf {kept_leaf:,}/{leaf_count:,})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
