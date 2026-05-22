
# PointsToWood

Wood-leaf semantic segmentation of TLS forest point clouds.

![Probability of wood predicted by our model from blue to red](images/our_tropical.png)
<sub>Probability of wood (blue = low, red = high). Data: Wang et al., 2021.</sub>

---

### Paper

> **PointsToWood: Reflectance-Modulated Anisotropic Convolutions for Leaf-Wood Segmentation Across Diverse Forest TLS Data**
> Owen, H. J. F., Allen, M. J. A., Grieve, S. W. D., Wilkes, P., Lines, E. R. *(under review)*

For the implementation described in the preprint, see the `version1.0-paper` branch.

---

## Quick Start

```bash
cd pointstowood
python predict.py --point-cloud your_plot.ply
```

Output: `your_plot_p2w.ply` with per-point columns `prediction` (0 = leaf, 1 = wood) and `pwood` (0.0–1.0).

---

## Installation

**Requirements:** Linux, CUDA 12.9, NVIDIA driver 570+, conda/mamba

```bash
bash pointstowood/install.sh
conda activate ptw
```

The script creates a `ptw` conda environment with Python 3.11, PyTorch 2.8.0, and all PyG extensions.

---

## Model Weights (Git LFS)

```bash
# Install Git LFS if needed
sudo apt-get install git-lfs
git lfs install

# Pull weights after cloning
git lfs pull
```

| File | Description |
|---|---|
| `model/h4mcc-eu.pth` | EU teacher model (default) |
| `model/h4mcc-finland.pth` | Distilled — Finnish forest |
| `model/h4mcc-poland.pth` | Distilled — Polish forest |
| `model/h4mcc-spain.pth` | Distilled — Spanish forest |

---

## Inference

```bash
# Default (EU teacher, adaptive settings)
python predict.py --point-cloud your_plot.ply

# Biome-specific model
python predict.py --point-cloud your_plot.ply --model h4mcc-finland.pth

# Geometry only (no reflectance)
python predict.py --point-cloud your_plot.ply --no-refl
```

Key options:

```
--model h4mcc-eu.pth     model checkpoint in pointstowood/model/ (default: h4mcc-eu.pth)
--inference-level 2      1=fast/single-scale, 2=default, 3=thorough, 4=maximum
--tta 2                  test-time augmentation: N z-axis rotations (default 2)
--any-wood 0.5           wood if any point in voxel exceeds threshold (higher recall)
```

---

## Training

### 1. Prepare data

Split raw `.ply` files into train/test/eval along the x-axis:

```bash
python split_ply.py /path/to/plots/ --prefix fin --output data/
```

Files are named by region prefix (`fin`, `spa`, `pol`, `gbr`, etc.) and written to `data/train/`, `data/test/`, `data/eval/`.

### 2. Train

```bash
# EU model (all European data)
python train.py --region eu --preprocess --device cuda

# Single biome
python train.py --region fin --preprocess --device cuda
```

---

## Knowledge Distillation

Distil a lightweight biome-specific student from the EU teacher:

```bash
python distill.py --region fin --preprocess --teacher-model eu.pth
```

---

## Architecture

A 3-stage encoder-decoder with a custom anisotropic convolution operator. LiDAR reflectance acts as a spatial modulator — weighting geometric neighbour contributions based on material properties rather than being treated as a flat input feature. The model falls back to geometry alone when reflectance is absent.

Supervision uses a multi-scale contrastive boundary loss at every encoder stage and a cyclical focal loss schedule (0 → peak → 0 over training) to stabilise early gradients while focusing on hard boundary points mid-training. Decoder unpooling uses encoder cluster indices rather than k-NN interpolation.

Student models use the same architecture compressed by channel width and block depth, trained via knowledge distillation from the teacher.

**Save criterion — H4-MCC:** harmonic mean of MCC across four conditions (pure/edge × with/without reflectance). Penalises any weak condition disproportionately.

---

## Region Prefixes

| `--region` | Data used |
|---|---|
| `eu` | All European prefixes |
| `global` | All files in pool |
| `fin` / `spa` / `pol` / ... | That prefix only |

---

## References

<sub>Mspace Lab (2024) ForestSemantic: A Dataset for Semantic Learning of Forest from Close-Range Sensing. Zenodo. https://doi.org/10.5281/zenodo.13285640.</sub>

<sub>Wang, Di; Takoudjou, Stéphane Momo; Casella, Eric (2021). LeWoS: A universal leaf-wood classification method to facilitate the 3D modelling of large tropical trees using terrestrial LiDAR. Dryad. https://doi.org/10.5061/dryad.np5hqbzp6.</sub>

<sub>Wan, Peng; Zhang, Wuming; Jin, Shuangna (2021). Plot-level wood-leaf separation for terrestrial laser scanning point clouds. Dryad. https://doi.org/10.5061/dryad.rfj6q5799.</sub>

<sub>Weiser, Hannah et al. (2024). Manually labeled terrestrial laser scanning point clouds of individual trees for leaf-wood separation. https://doi.org/10.11588/data/UUMEDI.</sub>

<sub>Owen, H. J. F., Lines, E., & Grieve, S. (2024). Plot-level semantically labelled terrestrial laser scanning point clouds (1.0). Zenodo. https://doi.org/10.5281/zenodo.13268500.</sub>

<sub>Van den Broeck, W.A.J., Terryn, L., Chen, S., Cherlet, W., Cooper, Z. T., & Calders, K. (2025). Pointwise deep learning for leaf-wood segmentation of tropical tree point clouds from terrestrial laser scanning. ISPRS Journal of Photogrammetry and Remote Sensing, 227, 366-382. https://doi.org/10.1016/j.isprsjprs.2025.06.023</sub>
