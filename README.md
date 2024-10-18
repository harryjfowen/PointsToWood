
# Semantic classification of wood and leaf in TLS forest point clouds

### Development Environment

- **Operating System:** Ubuntu LTS 22.04
- **GPU:** NVIDIA Quadro RTX 6000 24GB
- **NVIDIA Driver:** 535.183.06
- **CUDA Version:** 12.2

### Setup Instructions

1. Install the Ubuntu NVIDIA driver (535.183.06 recommended).

2. Set up a Conda environment:
   ```bash
   conda create --name myenv python=3.10
   conda activate myenv
   conda install mamba -c conda-forge

3. install packages within your Conda environment using mamba

   mamba install pytorch==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
   mamba install pyg==2.5.3 -c pyg

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



