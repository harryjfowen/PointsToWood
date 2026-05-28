# PointsToWood

Semantic segmentation of forest 3D point clouds — classifies each point as wood or leaf material.

## Prerequisites

- Python 3.10+
- CUDA 12.8 (for GPU inference)
- PyTorch 2.7.0 + PyTorch Geometric

## Installation

```bash
bash install_torch.sh
source ~/venvs/coan/bin/activate
pip install psutil laspy lazrs-python
```

## Models

Pre-trained weights live in `model/`. Download with Git LFS.

### Teacher model (EU-wide, ~20M parameters)

Trained on pan-European forest data. Used as the default and as the distillation source for biome-specific students.

| File | Description |
|------|-------------|
| `h4mcc-eu.pth` | EU-wide teacher (default) |

### Biome-specific student models (~1M parameters)

Distilled from the EU teacher on region-specific data. Faster inference, lower memory, and better calibrated to local forest structure. Selected automatically with `--region`.

| File | Region | `--region` flag |
|------|--------|-----------------|
| `h4mcc-finland.pth` | Finnish forest | `finland` |
| `h4mcc-poland.pth` | Polish forest | `poland` |
| `h4mcc-spain.pth` | Spanish forest | `spain` |

## Usage

### Inference

```bash
python predict.py --point-cloud forest.ply
```

Output: `forest_p2w.ply` in the same directory, with per-point wood/leaf labels.

#### Auto-select a biome-specific model with `--region`

```bash
python predict.py --point-cloud forest.ply --region finland
python predict.py --point-cloud forest.ply --region poland
python predict.py --point-cloud forest.ply --region spain
```

Passing `--region` selects the corresponding distilled student model automatically. Omit `--region` (or pass `eu`) to use the full EU teacher.

#### Key options

```
--region finland|poland|spain  auto-select biome-specific student model
--model h4mcc-eu.pth           override model checkpoint directly
--grid-size 2.0                voxel grid size in metres (auto-detected if omitted)
--fast                         quick preview mode (~2× faster, near-standard accuracy)
--verbose                      print performance summary
```

### Training

```bash
python train.py --region eu --preprocess --num-epochs 180
```

Training data expected at `data/{region}_train/` and `data/{region}_test/`.

### Distillation

Distil a biome-specific student (~1M params) from the EU teacher (~20M params):

```bash
python distill.py --region spain --teacher-model h4mcc-eu.pth
```

The student uses a lighter architecture (NetLight: C=16, K=8, 1-2-1 block pattern) vs the teacher (NetFull: C=128, K=16, 2-4-2 block pattern), making it ~20× smaller with minimal accuracy loss on in-distribution data.

## Evaluation metric

Models are selected and compared using **H4MCC** — the harmonic mean of Matthews Correlation Coefficient (MCC) across four conditions:

| Condition | Description |
|-----------|-------------|
| Pure + reflectance | Unambiguous voxels, full sensor data |
| Edge + reflectance | Wood/leaf boundary voxels, full sensor data |
| Pure + geometry | Unambiguous voxels, XYZ only |
| Edge + geometry | Wood/leaf boundary voxels, XYZ only |

MCC is preferred over F1 because it accounts for both wood and leaf errors — F1 ignores true negatives and can appear high even when leaf classification is poor. The harmonic mean across all four conditions means a model must perform well in every setting: strong reflectance performance cannot mask failure on geometry-only inputs, and easy voxels cannot hide poor boundary classification.

## License

[LICENSE]
