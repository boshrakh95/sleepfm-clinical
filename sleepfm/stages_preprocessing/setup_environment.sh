#!/bin/bash
# Setup STAGES Preprocessing Environment
# =======================================
#
# This script installs all required dependencies for STAGES data preparation.
#
# Usage:
#   bash setup_environment.sh

set -e

echo "=============================================="
echo "STAGES Data Preparation - Environment Setup"
echo "=============================================="
echo ""

# Check Python version
echo "Checking Python version..."
python --version

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]] && [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo ""
    echo "⚠️  WARNING: Not in a virtual environment!"
    echo "   It's recommended to use a conda or virtual environment."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Installing required packages..."
echo "----------------------------------------------"

pip install -r requirements_stages_prep.txt

echo ""
echo "----------------------------------------------"
echo "Installation complete!"
echo ""

# Verify installation
echo "Verifying installation..."
python -c "
import numpy as np
import scipy
import pandas as pd
import h5py
import yaml
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

print('✓ All packages installed successfully!')
print()
print('Package versions:')
print(f'  numpy: {np.__version__}')
print(f'  scipy: {scipy.__version__}')
print(f'  pandas: {pd.__version__}')
print(f'  h5py: {h5py.__version__}')
"

echo ""
echo "=============================================="
echo "Setup complete! You can now run:"
echo "  python test_single_subject.py --auto"
echo "  bash run_pipeline.sh"
echo "=============================================="
