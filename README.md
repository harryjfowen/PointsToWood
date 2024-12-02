
# Semantic classification of wood and leaf in TLS forest point clouds

![Probability of wood predicted by our model from blue to red (Data from Wang et al., 2021](images/our_tropical.png)
<sub>Figure is displaying probability of wood predicted by our model from blue (low probability) to red (high probability). (Data sourced from Wang et al., 2021)</sub>

### This model is fully described in the paper:
PointsToWood: A deep learning framework for complete canopy leaf-wood segmentation of TLS data across diverse European forests. Owen, H. J. F.,  Allen, M. J. A., Grieve S.W.D., Wilkes P., Lines, E. R. (in review)

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

4. install packages within your Conda environment using mamba
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
   pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
   pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
   pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
   pip install torch-geometric

📎 [Pytorch](https://pytorch.org/get-started/locally/) instructions for each OS can be found here.

📎 [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) instructions for each OS can be found here.

#

### Running PointsToWood
   
1. Activate your conda environment.
   
```
conda activate myenv. 
```

2. Navigate to the PointsToWood directory.
   
```
cd ~/points2wood/points2wood/
```

3. Run PointsToWood.
   
```
python3 predict.py --point-cloud /x/x/cloud.ply --model f1-eu.pth --batch_size 8 --is-wood 0.50 --grid_size 2.0 4.0 --min_pts 2048 --max_pts 16384;
```

*NOTE Make sure the point cloud contains columns x y z as a minimum and x y z reflectance if available to you.

### Model Selection

Within the model folder, we have biome specific as well as more general ecosystem agnostic models. 


### References 

<sub>Mspace Lab (2024) ‘ForestSemantic: A Dataset for Semantic Learning of Forest from Close-Range Sensing’, Geo-spatial Information Science. Zenodo. https://doi.org/10.5281/zenodo.13285640. Distributed under a Creative Commons Attribution Non Commercial No Derivatives 4.0 International licence. <</sub>

<sub>Wang, Di; Takoudjou, Stéphane Momo; Casella, Eric (2021). LeWoS: A universal leaf‐wood classification method to facilitate the 3D modelling of large tropical trees using terrestrial LiDAR [Dataset]. Dryad. https://doi.org/10.5061/dryad.np5hqbzp6. Distributed under a Creative Commons 0 1.0 Universal licence. <</sub>

<sub>Wan, Peng; Zhang, Wuming; Jin, Shuangna (2021). Plot-level wood-leaf separation for terrestrial laser scanning point clouds [Dataset]. Dryad. https://doi.org/10.5061/dryad.rfj6q5799. Distributed under a Creative Commons CC0 1.0 Universal licence. <</sub>

<sub>Weiser, Hannah; Ulrich, Veit; Winiwarter, Lukas; Esmorís, Alberto M.; Höfle, Bernhard, 2024, "Manually labeled terrestrial laser scanning point clouds of individual trees for leaf-wood separation", https://doi.org/10.11588/data/UUMEDI, heiDATA, V1, UNF:6:9U7BGTgjjsWd1GduT1qXjA== [fileUNF]. Distributed under a Creative Commons Attribution 4.0 International Deed.<</sub>

<sub>Harry, J. F. O., Emily, L., & Grieve, S. (2024). Plot-level semantically labelled terrestrial laser scanning point clouds (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13268500<</sub>


