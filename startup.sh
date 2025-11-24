#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure Python 3.11 is installed
if ! command -v python3.11 &> /dev/null; then
  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# Create virtual environment if it doesn't exist
if [ ! -f ".venv/bin/activate" ]; then
  rm -rf .venv
  python3.11 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip and install dependencies
python3.11 -m pip install -U pip setuptools wheel
[ -r requirements.txt ] && python3.11 -m pip install -r requirements.txt

# Explicitly install additional dependencies
python3.11 -m pip install ipykernel
python3.11 -m pip install lightgbm
python3.11 -m pip install graphviz
python3.11 -m pip install matplotlib
python3.11 -m pip install seaborn
python3.11 -m pip install fastparquet
python3.11 -m pip install scikit-learn

# Install cuDF (ensure compatibility with CUDA version)
python3.11 -m pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.08.* dask-cudf-cu12==24.08.*

# Install Kaggle API
python3.11 -m pip install kaggle
chmod 600 ~/.kaggle/kaggle.json

# Install Jupyter kernel for the virtual environment
python3.11 -m ipykernel install --user --name hm-venv --display-name "Python (HM venv)"