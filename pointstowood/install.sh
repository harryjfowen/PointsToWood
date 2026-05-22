#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="ptw"
PYTHON_VERSION="3.11"

# Load conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda.sh in ~/miniconda3/etc/profile.d/"
    exit 1
fi

# Create env if it doesn't already exist
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists. Skipping creation."
else
    mamba create -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip -c conda-forge -y
fi

conda activate "$ENV_NAME"

# PyTorch + CUDA 12.9
pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu129

# PyG compiled extensions matched to torch 2.8.0 + cu129
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu129.html

# PyG
pip install torch_geometric

# Other dependencies
pip install pandas pykdtree numba plyfile wandb

# Sanity check
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)

import torch_geometric
print("pyg:", torch_geometric.__version__)

import pyg_lib, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv
print("PyG extensions imported OK")
PY
