#!/bin/bash
# =============================================================================
# Experiment 6 - Cloud GPU Setup Script
# Run this ONCE after connecting to your rented cloud instance.
# Usage: bash setup.sh
# =============================================================================

set -e
echo "============================================"
echo "Experiment 6 - Cloud GPU Environment Setup"
echo "============================================"

# 1. System updates
echo "[1/6] Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-dev git 2>/dev/null || true

# 2. Upgrade pip
echo "[2/6] Upgrading pip..."
python3 -m pip install --upgrade pip -q

# 3. Install PyTorch with CUDA 12.1 (for A100/H100 GPUs)
echo "[3/6] Installing PyTorch with CUDA support..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# 4. Install project dependencies
echo "[4/6] Installing project requirements..."
python3 -m pip install -r experiment_6/requirements_full.txt -q

# 5. Verify GPU availability
echo "[5/6] Verifying GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.device_count()} GPU(s)')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: CUDA not available. Training will use CPU.')
"

# 6. Verify key imports
echo "[6/6] Verifying imports..."
python3 -c "
import numpy, pandas, matplotlib, torch
import gymnasium, stable_baselines3, optuna, yfinance
print('All imports OK')
print(f'  numpy={numpy.__version__}')
print(f'  torch={torch.__version__}')
print(f'  stable-baselines3={stable_baselines3.__version__}')
print(f'  gymnasium={gymnasium.__version__}')
print(f'  optuna={optuna.__version__}')
"

echo ""
echo "============================================"
echo "SETUP COMPLETE - Ready to train!"
echo "============================================"
echo ""
echo "Next commands:"
echo "  # Quick test (5 min, no Optuna)"
echo "  python3 experiment_6/train.py --pretrain 5000 --finetune 50000"
echo ""
echo "  # Full training with Optuna (2-3 hours on 40 GPUs)"
echo "  python3 experiment_6/train.py --optuna --trials 50 --pretrain 10000 --finetune 500000"
echo ""
echo "  # Run the Colab notebook interactively"
echo "  jupyter notebook experiment_6/full_colab_notebook.ipynb"
