# Experiment 7 - Cloud GPU Training Guide

Inside Bar Trend-Following RL Agent | Long-Only | Multi-Position | 10% Risk

---

## Files to Upload to Cloud Instance

Upload this **entire folder structure** to your cloud instance:

```
Trading_agent_1/
├── experiment_7/
│   ├── setup.sh                    ← Run first (installs everything)
│   ├── run_all.py                  ← Single entry point
│   ├── requirements_full.txt       ← Python dependencies
│   ├── train.py                    ← Main training pipeline
│   ├── train_2000.py               ← 2000-stock training pipeline
│   ├── download_2000.py            ← Bulk data downloader
│   ├── src/
│   │   ├── indicators.py           ← Feature engineering (RTI + 25 features)
│   │   ├── RTI.py                  ← Range Tightening Index formula
│   │   ├── enterprise_env.py       ← Trading environment (long-only)
│   │   ├── pretrain.py             ← Expert policy + behavioral cloning
│   │   └── optuna_optimizer.py     ← Hyperparameter search
│   ├── data/                       ← (empty - will auto-download)
│   ├── models/                     ← (empty - trained models go here)
│   ├── results/                    ← (empty - outputs go here)
│   └── checkpoints/                ← (empty - intermediate saves)
├── experiment_3/
│   ├── data/SP500_daily.csv        ← OR will be auto-downloaded
│   └── src/market_regime.py        ← SPX regime detection (required dependency)
└── data/
    └── Tickers_list_USA.csv        ← Ticker symbols
```

**Minimal upload** (if you want to download data on the cloud):
```
Trading_agent_1/
├── experiment_7/        ← Everything in experiment_7/
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
bash experiment_7/setup.sh
```

This installs: PyTorch+CUDA, stable-baselines3, gymnasium, optuna, yfinance, etc.

### Step 3: Download Data (Required Before Training)

```bash
# Download S&P 500 + stock data (2000 symbols, ~15-30 minutes)
python3 experiment_7/download_2000.py --max_stocks 2000 --workers 12

# Or download a smaller test set first
python3 experiment_7/download_2000.py --max_stocks 50 --workers 8

# For small-scale training, use experiment_3 data (pre-existing)
# Ensure experiment_3/data/SP500_daily.csv AND stocks_daily.parquet exist
```

### Step 4: Quick Test (Verification)

```bash
# Test the pipeline works (2-5 minutes)
python3 experiment_7/run_all.py --mode test
```

Expected output:
```
Experiment 7: Inside Bar Trend-Following RL Agent
============================================================
[Data] Computing indicators...
  X rows | Y symbols | 26 features
  Entry gates active: ZZZ (X.XX% of bars)

[Pretrain] Collecting expert demonstrations (2000 steps)...
  Expert actions: HOLD=1950 BUY=50

[Pretrain] Behavioral cloning from expert...
  Epoch 1/15 — Loss: 0.XX, Acc: 0.XX
  Epoch 15/15 — Loss: 0.00X, Acc: 1.XX

[Finetune] PPO training for 30,000 steps...
[Eval] Final evaluation on test set...

============================================================
FINAL RESULTS
============================================================
  Train time: XXs | Device: cpu/cuda
  Val Return:  +X.XX% | Val Trades: X
  Test Return: +X.XX% | Test Sharpe: X.XX | Max DD: X.XX%
  Test Trades: X
```

### Step 5: Full Training

```bash
# Full training without Optuna (30-60 min on GPU)
python3 experiment_7/run_all.py --mode full --finetune 200000

# Full training with Optuna (2-8 hours on GPU)
python3 experiment_7/run_all.py --mode full --optuna --trials 50 --finetune 500000

# 2000-stock variant (memory-optimized, chunked processing)
python3 experiment_7/train_2000.py --symbols 2000 --finetune 200000

# 2000-stock with Optuna
python3 experiment_7/train_2000.py --symbols 2000 --optuna --trials 50 --finetune 500000
```

**Time estimates:**
| Configuration | --finetune | --trials | Total Time |
|---------------|-----------|----------|------------|
| 1× CPU (small test) | 30,000 | 0 | ~10 minutes |
| 1× A100 (10 stocks) | 200,000 | 0 | ~15 minutes |
| 1× A100 (2000 stocks) | 200,000 | 0 | ~1 hour |
| 4× A100 (2000 stocks) | 500,000 | 50 | ~3 hours |

### Step 6: Download Results

```bash
# From your local machine
scp -r user@cloud-ip:/home/user/Trading_agent_1/experiment_7/results/ ./
scp -r user@cloud-ip:/home/user/Trading_agent_1/experiment_7/models/ ./
```

---

## Command Reference

```bash
# ─── Quick Commands ───

# Test pipeline (5 min)
python3 experiment_7/run_all.py --mode test

# Train with defaults (15 min CPU, 2 min GPU)
python3 experiment_7/run_all.py --mode full

# Full with Optuna (2-8 hours)
python3 experiment_7/run_all.py --mode full --optuna --trials 50

# Full with custom steps
python3 experiment_7/run_all.py --mode full --pretrain 10000 --finetune 500000

# Evaluate existing model
python3 experiment_7/run_all.py --mode eval --model experiment_7/models/exp7_final.zip

# Force CPU (even on GPU machine)
python3 experiment_7/run_all.py --mode test --device cpu

# ─── 2000-Stock Commands ───

# Download data first
python3 experiment_7/download_2000.py --max_stocks 2000 --workers 12

# Train 2000 stocks (chunked, memory-optimized)
python3 experiment_7/train_2000.py --symbols 2000 --finetune 200000

# Train 2000 stocks with Optuna
python3 experiment_7/train_2000.py --symbols 2000 --optuna --trials 50

# ─── Advanced ───

# Direct train.py usage
python3 experiment_7/train.py --optuna --trials 100 --pretrain 20000 --finetune 1000000

# View Optuna dashboard
optuna-dashboard sqlite:///experiment_7/optuna_studies/exp7.db
```

---

## Strategy Quick Reference

| # | Constraint | Threshold | Code Column |
|---|-----------|-----------|-------------|
| 1 | SPX > 50/100/200 EMA | All 3 positive | `spx_bull` |
| 2 | Within 25% of 52-week high | < 25% | `near_52w_high` |
| 3 | RTI range contraction | ≤ 15 (curr or prev) | `low_vol` |
| 4 | Strong prior uptrend | >25% (1m) or >30% (3-12m) | `strong_uptrend` |
| 5 | Weekly inside bar | High ≤ prev week high, Low ≥ prev week low | `weekly_inside_bar` |
| 6 | Daily inside bar | High ≤ prev high, Low ≥ prev low | `daily_inside_bar` |
| 7 | Low above prev low | Current Low > Previous Low | `low_holding` |
| 8 | Cross above prev high | Current High > Previous High | `cross_above_prev_high` |
| 9 | Entry near prev close | Close within 2% of prev close | `entry_near_prev_close` |

**Exit rules (automated, not agent decisions):**
- SL = current day low
- At 1:2 R:R → SL moves to breakeven
- Close below 10 EMA (first time) → book 50%
- Close below 21 EMA → book remaining 50%

---

## Troubleshooting

### Common Errors & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: No module named 'experiment_7'` | Running from wrong directory | Must run from repo root: `cd Trading_agent_1` |
| `ModuleNotFoundError: No module named 'experiment_3.src.market_regime'` | Missing experiment_3 dependency | Upload experiment_3/src/market_regime.py or copy to experiment_7 |
| CUDA out of memory | Too many symbols loaded at once | Use `train_2000.py` with `--symbols 100` for testing; reduce `chunk_size` to 100 |
| **0 trades / entry gate never active** | Strategy too strict for the data | Run this diagnostic first: `python3 -c "from experiment_7.train import load_data; df,fc=load_data(); print('Gates:',df['entry_gate'].sum(),'/',len(df))"` |
| RTI division by zero / NaN features | `max_range == min_range` during flat periods | Fixed: code uses `np.where` with fallback; add `min_periods=5` in rolling() |
| `entry_gate` = 0% for all bars | All 9 conditions must be simultaneously true; this is very restrictive | 1) Check individual condition stats with diagnostic; 2) Relax `require_entry_gate=False` in env to let agent learn timing; 3) Remove weakest conditions from entry_gate in indicators.py |
| yfinance download fails / rate limited | yfinance API throttling | Use VPN, reduce `--workers` to 4, or pre-download locally |
| No SP500 data column in stock DF | merge_regime_to_stocks failed | Ensure `SP500_daily.csv` has valid `Date` and `Close` columns |
| Optuna DB locked | Previous Optuna run crashed | Delete `experiment_7/optuna_studies/exp7.db` and restart |
| Gymnasium import fails | Old Python or package version | `pip install gymnasium>=0.29 --upgrade` |
| `_row` access error: KeyError | Feature columns missing from DF | Ensure `compute_indicators()` runs BEFORE creating env; check `get_feature_columns()` output |
| Partial exit shares rounding | Odd share counts cause remainder | Code handles via integer division (`shares // 2`); tiny remainder stays in position until EMA21 exit |

### Diagnostic Script

Run this to check indicator quality before training:

```python
import sys; sys.path.insert(0, '.')
from experiment_7.src.indicators import compute_indicators, get_feature_columns
from experiment_3.src.market_regime import compute_sp500_regime, merge_regime_to_stocks
import pandas as pd

# Load data
sp_df = pd.read_csv("experiment_3/data/SP500_daily.csv", parse_dates=["Date"])
sp_df = compute_sp500_regime(sp_df)
st_df = pd.read_parquet("experiment_3/data/stocks_daily.parquet")
st_df = merge_regime_to_stocks(st_df, sp_df)

# Compute indicators for first 10 symbols
for sym in sorted(st_df["Symbol"].unique())[:10]:
    sdf = st_df[st_df["Symbol"] == sym].copy()
    if len(sdf) < 200: continue
    sdf = compute_indicators(sdf)
    gates = sdf["entry_gate"].sum()
    total = len(sdf)
    
    # Per-condition breakdown
    print(f"\n{sym} ({total} bars): entry_gate = {gates} ({100*gates/total:.2f}%)")
    print(f"  spx_bull:          {sdf['spx_bull'].sum()}")
    print(f"  near_52w_high:     {sdf['near_52w_high'].sum()}")
    print(f"  low_vol (RTI):     {sdf['low_vol'].sum()}")
    print(f"  strong_uptrend:    {sdf['strong_uptrend'].sum()}")
    print(f"  weekly_inside_bar: {sdf['weekly_inside_bar'].sum()}")
    print(f"  daily_inside_bar:  {sdf['daily_inside_bar'].sum()}")
    print(f"  low_holding:       {sdf['low_holding'].sum()}")
    print(f"  cross_above_prev_high: {sdf['cross_above_prev_high'].sum()}")
    print(f"  entry_near_prev_close: {sdf['entry_near_prev_close'].sum()}")
    if gates == 0:
        print(f"  >>> WARNING: No entry gates for {sym}. Strategy too strict or data incompatible.")
```

---

## Expected Output Files

After training completes, you'll find:

```
experiment_7/
├── models/
│   ├── pretrained_policy.zip     ← After behavioral cloning
│   ├── pretrained_2000.zip        ← After 2000-stock cloning
│   ├── exp7_final.zip            ← After PPO fine-tuning
│   └── exp7_2000_final.zip       ← After 2000-stock finetuning
├── results/
│   ├── final_results.json        ← All metrics (small-scale)
│   ├── final_results_2000.json   ← All metrics (2000-stock)
│   ├── comprehensive_report.png  ← 9-panel plot
│   ├── comprehensive_2000.png    ← 2000-stock 9-panel plot
│   └── training_curves.png       ← Drawdown curves
├── checkpoints/
│   └── exp7_*.zip                ← Periodic saves (every 50k steps)
└── optuna_studies/
    ├── exp7.db                   ← Optuna study database
    └── best_params.json          ← Best hyperparameters
```

---

## Relaxing Strategy Constraints (If Too Few Trades)

If the diagnostic shows 0% entry gates, relax the most restrictive conditions in `src/indicators.py`:

```python
# Example: relax inside bar requirement (remove weekly inside bar)
df["entry_gate"] = (
    (df["spx_bull"] == 1) &
    (df["near_52w_high"] == 1) &
    (df["low_vol"] == 1) &
    (df["strong_uptrend"] == 1) &
    # (df["weekly_inside_bar"] == 1) &    # REMOVED - too restrictive
    (df["daily_inside_bar"] == 1) &
    (df["low_holding"] == 1) &
    (df["cross_above_prev_high"] == 1) &
    (df["entry_near_prev_close"] == 1)
).astype(int)
```

Or let the agent learn timing by setting `require_entry_gate=False` in the environment (keeps gates for pretraining, removes them for PPO):

```python
# In train.py make_env():
InsideBarTradingEnv(
    ...
    require_entry_gate=False,  # Agent learns own entry timing
    use_action_masking=True,
)
```
