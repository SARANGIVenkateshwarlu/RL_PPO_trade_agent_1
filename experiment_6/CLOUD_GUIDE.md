# Experiment 6 - Cloud GPU Training Guide

## Files to Upload to Cloud Instance

Upload this **entire folder structure** to your cloud instance:

```
Trading_agent_1/
├── experiment_6/
│   ├── setup.sh                    ← Run first (installs everything)
│   ├── run_all.py                  ← Single entry point
│   ├── requirements_full.txt       ← Python dependencies
│   ├── train.py                    ← Main training pipeline
│   ├── full_colab_notebook.ipynb   ← Jupyter/Colab notebook
│   ├── src/
│   │   ├── indicators.py           ← Feature engineering
│   │   ├── enterprise_env.py       ← Trading environment
│   │   ├── pretrain.py             ← Expert policy + cloning
│   │   └── optuna_optimizer.py     ← Hyperparameter search
│   ├── data/                       ← (empty - will auto-download)
│   ├── models/                     ← (empty - trained models go here)
│   ├── results/                    ← (empty - outputs go here)
│   └── checkpoints/                ← (empty - intermediate saves)
├── experiment_3/
│   ├── data/SP500_daily.csv        ← OR will be auto-downloaded
│   └── src/market_regime.py        ← SPX regime detection
└── data/
    └── Tickers_list_USA.csv        ← Ticker symbols
```

**Minimal upload** (if you want to download data on the cloud):
```
Trading_agent_1/
├── experiment_6/       ← Everything in experiment_6/
├── experiment_3/src/market_regime.py
└── data/Tickers_list_USA.csv
```

---

## Step-by-Step Cloud Setup

### Step 1: Connect & Upload

```bash
# From your local machine, upload the project
scp -r Trading_agent_1/ user@cloud-ip:/home/user/

# OR use rsync for large files
rsync -avz --progress Trading_agent_1/ user@cloud-ip:/home/user/Trading_agent_1/
```

### Step 2: Run Setup Script

```bash
ssh user@cloud-ip
cd /home/user/Trading_agent_1
bash experiment_6/setup.sh
```

This installs: PyTorch+CUDA, stable-baselines3, gymnasium, optuna, yfinance, etc.

### Step 3: Quick Test (Verification)

```bash
# Test the pipeline works (2-3 minutes)
python3 experiment_6/run_all.py --mode test
```

Expected output:
```
[Pretrain] Collecting expert demonstrations (2000 steps)...
  Expert actions: HOLD=1950 BUY=30 SELL=20
[Pretrain] Behavioral cloning from expert...
  Epoch 1/15 — Loss: 0.30, Acc: 0.92
  Epoch 15/15 — Loss: 0.002, Acc: 1.00
[Finetune] PPO training for 30,000 steps...
FINAL RESULTS
  Test Return: +X.XX% | Sharpe: X.XX | Max DD: X.XX%
```

### Step 4: Full Training (40 GPUs)

```bash
# Full enterprise training with Optuna optimization
python3 experiment_6/run_all.py --mode full --optuna --trials 50 --finetune 500000
```

**Time estimates:**
| GPUs | --finetune | --trials | Total Time |
|------|-----------|----------|------------|
| 1× A100 | 500,000 | 50 | ~8 hours |
| 4× A100 | 500,000 | 50 | ~2 hours |
| 8× A100 | 500,000 | 50 | ~1 hour |
| 40× A100 | 500,000 | 50 | **~15 minutes** |

### Step 5: Download Results

```bash
# From your local machine
scp -r user@cloud-ip:/home/user/Trading_agent_1/experiment_6/results/ ./
scp -r user@cloud-ip:/home/user/Trading_agent_1/experiment_6/models/ ./
```

---

## Running the Jupyter Notebook

```bash
# Start Jupyter on cloud (port 8888)
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# On your local machine, tunnel to the cloud
ssh -N -L 8888:localhost:8888 user@cloud-ip

# Open browser: http://localhost:8888
# Navigate to: experiment_6/full_colab_notebook.ipynb
```

---

## Command Reference

```bash
# ─── Quick Commands ───

# Test pipeline (5 min)
python3 experiment_6/run_all.py --mode test

# Train with defaults (15 min CPU, 2 min GPU)
python3 experiment_6/run_all.py --mode full

# Full with Optuna (2-8 hours)
python3 experiment_6/run_all.py --mode full --optuna --trials 50

# Full with custom steps
python3 experiment_6/run_all.py --mode full --pretrain 10000 --finetune 500000

# Evaluate existing model
python3 experiment_6/run_all.py --mode eval --model experiment_6/models/exp6_final.zip

# Force CPU (even on GPU machine)
python3 experiment_6/run_all.py --mode test --device cpu

# ─── Advanced ───

# Direct train.py usage
python3 experiment_6/train.py --optuna --trials 100 --pretrain 20000 --finetune 1000000

# Run only Optuna without full training
python3 -c "
from experiment_6.src.optuna_optimizer import optimize_hyperparameters
# ... customize as needed
"

# View Optuna dashboard
optuna-dashboard sqlite:///experiment_6/optuna_studies/exp6.db
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'experiment_6'` | Run from `Trading_agent_1/` directory, or `export PYTHONPATH=$PWD` |
| CUDA out of memory | Reduce `batch_size` to 32, `n_steps` to 1024 |
| yfinance download fails | Use VPN or pre-download data locally |
| Optuna DB locked | Delete `experiment_6/optuna_studies/exp6.db` and restart |
| 0 trades in results | Increase `--pretrain`, lower `squeeze_min_level` in env |

---

## Expected Output Files

After training completes, you'll find:

```
experiment_6/
├── models/
│   ├── pretrained_policy.zip    ← After behavioral cloning
│   └── exp6_final.zip           ← After PPO fine-tuning
├── results/
│   ├── final_results.json       ← All metrics in JSON
│   ├── comprehensive_report.png ← 9-panel plot
│   └── training_curves.png      ← Loss/DD curves
├── checkpoints/
│   └── exp6_*.zip               ← Periodic saves
└── optuna_studies/
    ├── exp6.db                  ← Optuna study database
    └── best_params.json         ← Best hyperparameters
```
