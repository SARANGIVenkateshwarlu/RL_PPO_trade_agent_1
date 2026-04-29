"""
Experiment 4: Breakout-Constrained RL Trading Agent - Training Script

Core Trading Rules (MANDATORY):
  BUY:  Only when High_t > High_{t-1} AND model signals BUY
  SELL: Only when Low_t < Low_{t-1} AND model signals SELL
  NO TRADE: If breakout fails → Force HOLD

Position Sizing: 20% cash, 1% ATR risk, 3% reward (1:3 RR)
Reward: R_t = Sharpe_Return_t × Hold_Duration_Penalty - Slippage_Cost
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from experiment_4.src.indicators import compute_all_indicators, get_feature_columns
from experiment_4.src.breakout_env import BreakoutTradingEnv
from experiment_4.src.backtest import run_backtest_comparison
from experiment_3.src.market_regime import compute_sp500_regime, merge_regime_to_stocks


def load_and_prepare():
    print("=" * 60)
    print("Experiment 4: Breakout-Constrained RL Trading Agent")
    print("=" * 60)

    # Load SP500 regime
    sp_path = "experiment_3/data/SP500_daily.csv"
    sp_df = pd.read_csv(sp_path, parse_dates=["Date"])
    sp_df = compute_sp500_regime(sp_df)

    # Load stocks
    st_path = "experiment_3/data/stocks_daily.parquet"
    st_df = pd.read_parquet(st_path)
    st_df = merge_regime_to_stocks(st_df, sp_df)
    print(f"[Data] {len(st_df)} rows, {st_df['Symbol'].nunique()} symbols")

    # Compute indicators per symbol
    print("[Indicators] Computing breakout flags + technicals...")
    all_results = []
    for sym in sorted(st_df["Symbol"].unique()):
        sdf = st_df[st_df["Symbol"] == sym].copy()
        if len(sdf) < 200:
            continue
        sdf = compute_all_indicators(sdf)
        sdf["Symbol"] = sym
        all_results.append(sdf)

    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    feature_cols = get_feature_columns(combined)
    print(f"  Rows: {len(combined)}, Features: {len(feature_cols)}")
    print(f"  Breakout UP days: {(combined['breakout_up']==1).sum()} | DOWN: {(combined['breakout_down']==1).sum()}")
    return combined, feature_cols


def split_data(df, train_ratio=0.7):
    train, test = [], []
    for sym in sorted(df["Symbol"].unique()):
        sdf = df[df["Symbol"] == sym].sort_values("Date")
        n = int(len(sdf) * train_ratio)
        train.append(sdf.iloc[:n])
        test.append(sdf.iloc[n:])
    return pd.concat(train).reset_index(drop=True), pd.concat(test).reset_index(drop=True)


def main():
    df, feature_cols = load_and_prepare()

    # Split
    train_df, test_df = split_data(df, 0.7)
    print(f"\n[Split] Train: {len(train_df)} | Test: {len(test_df)}")

    # Normalization
    mean = train_df[feature_cols].values.astype(np.float32).mean(axis=0)
    std = train_df[feature_cols].values.astype(np.float32).std(axis=0)
    std[std == 0] = 1.0

    # --- Training Env ---
    print("\n[Env] Building Breakout-Constrained environment...")
    env_fn = lambda: BreakoutTradingEnv(
        df=train_df, window_size=30,
        feature_columns=feature_cols,
        feature_mean=mean, feature_std=std,
        cash_fraction=0.20, risk_per_trade_pct=0.01, reward_target_pct=0.03,
        commission_pct=0.001, slippage_pct=0.001,
        hold_duration_penalty=0.001,
        random_start=True, min_episode_steps=200, episode_max_steps=1500,
    )
    train_env = DummyVecEnv([env_fn])

    # --- Model ---
    print("\n[Model] PPO with MLP [256,256,128]")
    model = PPO(
        "MlpPolicy", train_env, verbose=1,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
    )

    os.makedirs("experiment_4/checkpoints", exist_ok=True)
    ckpt = CheckpointCallback(save_freq=50_000, save_path="experiment_4/checkpoints/", name_prefix="exp4")

    TOTAL = 200_000
    print(f"\n[Train] {TOTAL:,} timesteps...")
    model.learn(total_timesteps=TOTAL, callback=ckpt)
    print("Training complete!")

    os.makedirs("experiment_4/models", exist_ok=True)
    model.save("experiment_4/models/exp4_breakout")

    # --- Evaluation Env ---
    print("\n[Eval] Building test environment...")
    test_env = DummyVecEnv([lambda: BreakoutTradingEnv(
        df=test_df, window_size=30,
        feature_columns=feature_cols,
        feature_mean=mean, feature_std=std,
        cash_fraction=0.20, risk_per_trade_pct=0.01, reward_target_pct=0.03,
        commission_pct=0.001, slippage_pct=0.001,
        hold_duration_penalty=0.0,
        random_start=False, episode_max_steps=None,
    )])

    # --- Backtest ---
    comparison = run_backtest_comparison(model, test_env, test_df, init_equity=100000.0)

    # --- Save ---
    os.makedirs("experiment_4/results", exist_ok=True)
    results = {
        "timestamp": datetime.now().isoformat(),
        "rl": comparison["rl"],
        "buy_hold": comparison["buy_hold"],
        "random_breakout": comparison["random_breakout"],
        "config": {
            "cash_fraction": 0.20, "risk_per_trade": 0.01, "reward_target": 0.03,
            "rr_ratio": "1:3", "timesteps": TOTAL, "features": feature_cols,
        },
    }
    with open("experiment_4/results/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to experiment_4/results/results.json")

    # Quick summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  RL Agent:      {comparison['rl']['total_return_pct']:+.2f}% | Sharpe: {comparison['rl']['sharpe_ratio']:.2f}")
    print(f"  Buy & Hold:    {comparison['buy_hold']['return_pct']:+.2f}% | Max DD: {comparison['buy_hold']['max_dd']:.1f}%")
    print(f"  Random BO:     {comparison['random_breakout']['return_pct']:+.2f}% | Trades: {comparison['random_breakout']['trades']}")

    return model, results


if __name__ == "__main__":
    main()
