# Reinforcement Learning Forex Trading Agent - Project Development Report

## Overview

This project implements a Reinforcement Learning (RL) trading agent for the EURUSD forex pair, inspired by [ZiadFrancis/ReinforcementTrading_Part_1](https://github.com/ZiadFrancis/ReinforcementTrading_Part_1) and the YouTube tutorial *"Reinforcement Learning Trading Bot in Python | Train an AI Agent on Forex (EURUSD)."*

The agent uses an **EMA crossover-based feature set** combined with PPO (Proximal Policy Optimization) to learn optimal entry/exit decisions with dynamic stop-loss and take-profit levels.

---

## 1. Source Material Summary

### YouTube Tutorial (oW4hgB1vIoY)
The video demonstrates building a complete RL trading pipeline:
- Custom Gym/Gymnasium environment for forex trading
- Position persistence (agent holds positions across timesteps)
- Multi-discrete action space: HOLD, CLOSE, OPEN(direction, SL, TP)
- PPO training with Stable-Baselines3
- Checkpoint-based model selection by OOS performance

### Original GitHub Repository
The original repo provides:
- `trading_env.py` - Custom Gym environment with position persistence, SL/TP logic, reward shaping, and random episode starts
- `indicators.py` - Technical indicator computation (RSI, ATR, MA crosses)
- `train_agent.py` - Training pipeline with checkpoint evaluation
- `test_agent.py` - Evaluation and trade history export

---

## 2. Enhancements & Refinements Made

### 2.1 Feature Engineering (indicators.py)
| Original | Enhanced |
|----------|----------|
| RSI(14), ATR(14) | Same + ATR as % of price |
| MA20, MA50, basic slopes | **EMA 9/21/50** with normalized slopes |
| Simple MA spread | EMA spread signals (9-21, 21-50, 9-50) + spread slopes |
| - | **MACD** (12,26,9) line, signal, histogram |
| - | **Bollinger Bands** (%B, bandwidth) |
| - | **Stochastic** (%K, %D) |
| - | **ADX** trend strength |
| - | Volatility ratio, volume ratio, OBV slope |
| pandas-ta dependency | **Pure numpy/pandas** (no external TA library) |

**Why**: The EMA crossover strategy is captured via EMA spreads and their slopes, supplemented by momentum (MACD, Stoch), volatility (BB, ATR), and trend (ADX) indicators.

### 2.2 Environment Enhancements (trading_env.py)
- **Feature normalization**: Z-score normalization from training data
- **Drawdown tracking**: Monitors peak equity and current drawdown
- **Drawdown-based early termination**: Episode ends if max drawdown threshold (30%) is hit
- **Drawdown penalty**: Large negative reward for exceeding max drawdown
- **4 state features**: position, time_in_trade, unrealized_pnl, drawdown_ratio
- **Position sizing**: Configurable lot multiplier
- **Trade history tracking**: Stores all closed trade details for analysis

### 2.3 Training Enhancements (train_agent.py)
- **Feature normalization statistics**: Computed from training data, applied to both train and test
- **Larger action space**: 8 SL options x 8 TP options x 2 directions = 128 OPEN actions + HOLD + CLOSE = **130 total**
- **Deeper network**: [256, 256, 128] for both policy and value networks
- **Higher entropy coefficient** (0.01): Encourages exploration
- **Comprehensive metrics**: Total return, Sharpe, Sortino, max drawdown
- **Visualization**: Side-by-side equity curves and drawdown plots

### 2.4 Evaluation Enhancements (test_agent.py)
- Trade statistics: win rate, avg win/loss, profit factor, trade reasons
- Returns distribution histogram
- Downside deviation (Sortino ratio)
- Trade PnL bar chart

---

## 3. Training Results

### Configuration
| Parameter | Value |
|-----------|-------|
| Data | EURUSD Hourly (2024-04 to 2026-04, 12,260 bars) |
| Train/Test Split | 80% / 20% (9,808 / 2,452 bars) |
| Features | 25 scale-invariant indicators |
| Window Size | 30 bars |
| Actions | 130 (8 SL x 8 TP x 2 dirs + HOLD + CLOSE) |
| Algorithm | PPO (MLP Policy) |
| Network | pi=[256,256,128], vf=[256,256,128] |
| Total Timesteps | 200,000 |
| Learning Rate | 3e-4 |
| Batch Size | 64 |

### Results Summary

| Metric | In-Sample (Train) | Out-of-Sample (Test) |
|--------|-------------------|---------------------|
| **Final Equity** | $10,751.86 | **$7,850.00** |
| **Total Return** | +7.52% | **-21.50%** |
| **Sharpe Ratio** | - | -1.10 |
| **Sortino Ratio** | - | -0.26 |
| **Max Drawdown** | - | **31.14%** |

### Key Observations
1. **Significant overfitting**: The model achieved positive returns on training data (+7.52%) but lost 21.5% on test data
2. **High drawdown**: Max drawdown of 31.14% on test data triggered the drawdown-based early termination
3. **Negative Sharpe**: -1.10 indicates poor risk-adjusted returns
4. The model likely learned patterns specific to the training period that did not generalize

---

## 4. Analysis: Why Did It Underperform?

### Root Causes
1. **Limited training data**: Only ~9,800 hourly bars (approx 14 months) for training. RL requires substantially more data.
2. **Stationarity issues**: Forex market regime changes across 2024-2026 (Fed policy shifts, geopolitical events) cause distribution shift between train/test.
3. **Large discrete action space**: 130 actions with only 200k timesteps means ~1,500 steps per action on average — insufficient for robust learning.
4. **Reward signal sparsity**: Trades only close on manual close, SL, or TP. Rewards are infrequent, making credit assignment challenging.
5. **No transaction cost on hold**: The model has little incentive to stay flat during uncertain periods.

### Improvement Recommendations
1. **More training data**: Use 5-10 years of data with multiple currency pairs
2. **Data augmentation**: Add synthetic noise, time reversal, or GAN-generated data
3. **Reduce action space**: Use 4-5 SL/TP combos instead of 8
4. **Curriculum learning**: Start with larger timeframes, gradually reduce
5. **Ensemble methods**: Train multiple agents with different seeds, ensemble predictions
6. **Better reward shaping**: Add cost for holding during high volatility, reward for low-drawdown periods
7. **Longer training**: 500k-1M timesteps with learning rate decay
8. **Walk-forward validation**: Instead of single split, use rolling window backtesting
9. **Simplify features**: Reduce to core EMA crossover signals first, then layer in complexity

---

## 5. Project Structure

```
Trading_agent_1/
├── src/
│   ├── __init__.py
│   ├── indicators.py      # Technical indicator computation (EMA crossovers, MACD, BB, Stoch, ADX)
│   ├── trading_env.py     # Custom Gymnasium environment (position persistence, SL/TP, drawdown control)
│   ├── train_agent.py     # Training pipeline with checkpointing and evaluation
│   └── test_agent.py      # Comprehensive evaluation with trade statistics
├── data/
│   └── EURUSD_Hourly.csv  # Downloaded EURUSD hourly data (12,260 bars)
├── models/
│   └── forex_trader_best.zip  # Trained PPO model
├── checkpoints/           # Training checkpoints
├── results/
│   ├── metrics.json       # Training/evaluation metrics
│   ├── training_results.png  # Equity curves + drawdown plot
│   ├── evaluation_results.json
│   └── evaluation_plots.png
├── run_training.py        # Standalone training script
├── download_data.py       # EURUSD data downloader
├── requirements.txt       # Python dependencies
├── full_Notebook.ipynb    # Complete interactive notebook
└── PROJECT_REPORT.md      # This file
```

---

## 6. How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Download Data
```bash
python download_data.py
```

### Train
```bash
python run_training.py
# Or for the full pipeline:
python -m src.train_agent
```

### Evaluate
```bash
python -m src.test_agent
```

### Jupyter Notebook
```bash
jupyter notebook full_Notebook.ipynb
```

---

## 7. Conclusion

This project successfully implemented and enhanced the RL trading agent from the original tutorial. While the model demonstrated learning capability (positive in-sample returns), it **overfit significantly** to training data, resulting in a -21.5% OOS loss.

The key takeaways:
- RL for trading is highly prone to overfitting due to low signal-to-noise ratio in financial data
- Large discrete action spaces require proportionally more training
- Reward shaping and regularization are critical for generalization
- Next steps should focus on data quantity, feature simplification, and robust validation

The infrastructure built here (environment, indicators, training pipeline) provides a solid foundation for further experimentation with different architectures (e.g., continuous actions, recurrent policies) and data sources.

---

*Report generated: 2026-04-29*
