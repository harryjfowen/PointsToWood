
# Semantic classification of wood and leaf in TLS forest point clouds

![Probability of wood predicted by our model from blue to red (Data from Wang et al., 2021](images/our_tropical.png)
<sub>Figure is displaying probability of wood predicted by our model from blue (low probability) to red (high probability). (Data sourced from Wang et al., 2021)</sub>

### This model is described in the paper:
PointsToWood: A deep learning framework for complete canopy leaf-wood segmentation of TLS data across diverse European forests. Owen, H. J. F.,  Allen, M. J. A., Grieve S.W.D., Wilkes P., Lines, E. R. (under review)

## Version Information

### Current Version (version2.0) - Advanced Features
This branch contains the latest version with significant architectural improvements that will be described in the forthcoming paper:

**New Features:**
- **Directional Anisotropic Convolution** with reflectance-based attention mechanisms
- **Inverted Residual Blocks** for improved feature extraction and regularization
- **Squeeze-Excitation (SE) Channel Attention** for adaptive feature recalibration
- **Adaptive Receptive Field Scaling** with learnable ρ parameters
- **Cyclical Edge Weighted Focal Loss** for better training stability and convergence
- **Enhanced Data Processing** with denoising and efficient batching
- **Robust Multimodal Learning** handling both geometric and reflectance data

### arXiv Version (version1.0)
For the exact implementation described in the arXiv preprint, please switch to the `version1.0` branch:
```bash
git checkout version1.0
```

#

### Development Environment

- **Operating System:** Ubuntu LTS 22.04
- **GPU:** NVIDIA Quadro RTX 6000 24GB
- **NVIDIA Driver:** 535.183.06
- **CUDA Version:** 12.2

### Setup Instructions

1. Install the Ubuntu NVIDIA driver (535.183.06 recommended).
   '''bash
   sudo ubuntu-drivers install nvidia:535

2. Install NVIDIA toolkit (https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

3. Set up a Conda environment:
   ```bash
   conda create --name myenv python=3.10 mamba -c conda-forge
   conda activate myenv

4. install packages within your Conda environment
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
   pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
   pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
   pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
   pip install torch-geometric
   pip install pandas pykdtree numba 

📎 [Pytorch](https://pytorch.org/get-started/locally/) instructions for each OS can be found here.

📎 [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) instructions for each OS can be found here.

### 5. Download Model Weights Using Git LFS

**Install Git LFS:**
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs
# macOS
brew install git-lfs
# Windows: https://git-lfs.github.com/
git lfs install
```

**After cloning the repository, download model weights (choose one):**

#### Download Options
- **All models (~500MB):**
  ```bash
  git lfs pull
  ```
- **Only EU models (~200MB):**
  ```bash
  git lfs pull --include="pointstowood/model/fbeta-eu.pth"
  git lfs pull --include="pointstowood/model/ba-eu.pth"
  ```
- **Only biome models (~100MB):**
  ```bash
  git lfs pull --include="pointstowood/model/ba-spain.pth"
  git lfs pull --include="pointstowood/model/ba-poland.pth"
  git lfs pull --include="pointstowood/model/ba-finland.pth"
  ```
- **A specific model:**
  ```bash
  git lfs pull --include="pointstowood/model/fbeta-eu.pth"  # Replace as needed
  ```

*Tip: You can always run `git lfs pull` later to fetch any missing models.*


#

### Running PointsToWood
   
1. Activate your conda environment.
   
```
conda activate myenv. 
```

2. Navigate to the PointsToWood directory.
   
```
cd ~/PointsToWood/pointstowood/
```

3. Run PointsToWood.

**EU Models (Recommended for General Use):**
```bash
# Precision-focused model
python3 predict.py --point-cloud ~/PointsToWood/pointstowood/data/eu_eval/uk01_lw_pl_3.ply --model fbeta-eu.pth --batch-size 4 --any-wood 0.50 --grid-size 2.0 3.0 --resolution 0.02 --min-pts 512 --max-pts 16384

# Balanced accuracy model
python3 predict.py --point-cloud ~/PointsToWood/pointstowood/data/eu_eval/uk01_lw_pl_3.ply --model ba-eu.pth --batch-size 4 --any-wood 0.50 --grid-size 2.0 3.0 --resolution 0.02 --min-pts 512 --max-pts 16384
```

**Biome-Specific Models (Faster Inference):**
```bash
# Spanish forests
python3 predict.py --point-cloud your_data.ply --model ba-spain.pth --batch-size 8 --any-wood 0.50 --grid-size 2.0 --resolution 0.02 --min-pts 512 --max-pts 16384

# Polish forests
python3 predict.py --point-cloud your_data.ply --model ba-poland.pth --batch-size 8 --any-wood 0.50 --grid-size 2.0 --resolution 0.02 --min-pts 512 --max-pts 16384

# Finnish forests
python3 predict.py --point-cloud your_data.ply --model ba-finland.pth --batch-size 8 --any-wood 0.50 --grid-size 2.0 --resolution 0.02 --min-pts 512 --max-pts 16384
```

**Detection Strategies:**
- **`--any-wood`**: Aggressive wood detection - classifies as wood if ANY neighbor exceeds threshold
- **`--is-wood`**: Conservative wood detection - classifies as wood if ALL neighbors exceed threshold
- **`--max-probabilities`**: Uses most confident prediction in each neighborhood

## Data Requirements

### Input Format
- **File Format**: Point cloud must be in `.ply` format
- **Point Cloud Type**: TLS (Terrestrial Laser Scanner) data
- **Required Columns**: `x y z` (coordinates)
- **Optional Columns**: `reflectance` or `intensity` (recommended for best performance)
- **Point Spacing**: Sub 2 cm optimal, but can function beyond that (not ideal for larger spacing)
- **Processing**: Handles downsampling from raw TLS output automatically

### Output Format
The model will append two new columns to your point cloud:
- **`prediction`**: Binary classification (0 = leaf, 1 = wood)
- **`pwood`**: Probability of wood classification (0.0 to 1.0)

## Model Information

### Available Models

#### **EU Models (67M parameters)**
- **`fbeta-eu.pth`**: F1-optimized model with slight preference for precision (β = 0.9)
  - Best for applications where precision is slightly more important than recall
  - Trained on European forest data
  - Recommended for most use cases
- **`ba-eu.pth`**: Balanced accuracy optimized model
  - Optimized for balanced accuracy across all classes
  - Good general-purpose model for European forests
  - Equal emphasis on precision and recall

#### **Biome-Specific Models (3.5M parameters)**
Lightweight models optimized for specific biomes:
- **`ba-spain.pth`**: Balanced accuracy model for Spanish forests
- **`ba-poland.pth`**: Balanced accuracy model for Polish forests  
- **`ba-finland.pth`**: Balanced accuracy model for Finnish forests

**Model Selection Guide:**
- **Use EU models** for general European forest applications
- **Use biome-specific models** for targeted regions (faster inference, smaller memory footprint)
- **Use `fbeta-eu.pth`** when precision is slightly more important
- **Use `ba-*` models** when balanced performance is desired 


### References 

<sub>Mspace Lab (2024) ‘ForestSemantic: A Dataset for Semantic Learning of Forest from Close-Range Sensing’, Geo-spatial Information Science. Zenodo. https://doi.org/10.5281/zenodo.13285640. Distributed under a Creative Commons Attribution Non Commercial No Derivatives 4.0 International licence. <</sub>

<sub>Wang, Di; Takoudjou, Stéphane Momo; Casella, Eric (2021). LeWoS: A universal leaf‐wood classification method to facilitate the 3D modelling of large tropical trees using terrestrial LiDAR [Dataset]. Dryad. https://doi.org/10.5061/dryad.np5hqbzp6. Distributed under a Creative Commons 0 1.0 Universal licence. <</sub>

<sub>Wan, Peng; Zhang, Wuming; Jin, Shuangna (2021). Plot-level wood-leaf separation for terrestrial laser scanning point clouds [Dataset]. Dryad. https://doi.org/10.5061/dryad.rfj6q5799. Distributed under a Creative Commons CC0 1.0 Universal licence. <</sub>

<sub>Weiser, Hannah; Ulrich, Veit; Winiwarter, Lukas; Esmorís, Alberto M.; Höfle, Bernhard, 2024, "Manually labeled terrestrial laser scanning point clouds of individual trees for leaf-wood separation", https://doi.org/10.11588/data/UUMEDI, heiDATA, V1, UNF:6:9U7BGTgjjsWd1GduT1qXjA== [fileUNF]. Distributed under a Creative Commons Attribution 4.0 International Deed.<</sub>

<sub>Harry, J. F. O., Emily, L., & Grieve, S. (2024). Plot-level semantically labelled terrestrial laser scanning point clouds (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13268500<</sub>

