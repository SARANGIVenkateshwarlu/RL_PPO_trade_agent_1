"""
Experiment 5: BB Squeeze + Breakout + 9 EMA Exit — Training Script

Entry: breakout_up + squeeze + model BUY signal
Exit:  entry candle low (SL) / 9 EMA cross (trailing exit)
Size:  20% cash, 1% risk (candle SL based), 3% target (1:3 RR)
"""
import os, sys, json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from experiment_5.src.indicators import compute_all_indicators, get_feature_columns
from experiment_5.src.squeeze_env import SqueezeBreakoutEnv
from experiment_5.src.backtest import run_backtest
from experiment_3.src.market_regime import compute_sp500_regime, merge_regime_to_stocks


def load_and_prepare():
    print("=" * 60)
    print("Experiment 5: BB Squeeze + Breakout + 9 EMA Exit")
    print("=" * 60)

    sp_path = "experiment_3/data/SP500_daily.csv"
    sp_df = pd.read_csv(sp_path, parse_dates=["Date"])
    sp_df = compute_sp500_regime(sp_df)

    st_path = "experiment_3/data/stocks_daily.parquet"
    st_df = pd.read_parquet(st_path)
    st_df = merge_regime_to_stocks(st_df, sp_df)

    print("[Indicators] Computing BB squeeze + breakout features...")
    results = []
    for sym in sorted(st_df["Symbol"].unique()):
        sdf = st_df[st_df["Symbol"] == sym].copy()
        if len(sdf) < 200: continue
        sdf = compute_all_indicators(sdf)
        sdf["Symbol"] = sym
        results.append(sdf)

    combined = pd.concat(results, ignore_index=True)
    combined = combined.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    fc = get_feature_columns(combined)

    sqz_counts = combined["squeeze_signal"].value_counts().to_dict()
    print(f"  Rows: {len(combined)} | Features: {len(fc)}")
    print(f"  Squeeze: Full={sqz_counts.get(2,0)} Moderate={sqz_counts.get(1,0)} None={sqz_counts.get(0,0)}")
    return combined, fc


def split_data(df, train_ratio=0.7):
    train, test = [], []
    for sym in sorted(df["Symbol"].unique()):
        sdf = df[df["Symbol"] == sym].sort_values("Date")
        n = int(len(sdf) * train_ratio)
        train.append(sdf.iloc[:n]); test.append(sdf.iloc[n:])
    return pd.concat(train).reset_index(drop=True), pd.concat(test).reset_index(drop=True)


def main():
    df, fc = load_and_prepare()
    train_df, test_df = split_data(df, 0.7)
    print(f"\n[Split] Train: {len(train_df)} | Test: {len(test_df)}")

    mean = train_df[fc].values.astype(np.float32).mean(0)
    std = train_df[fc].values.astype(np.float32).std(0)
    std[std == 0] = 1.0

    print("\n[Env] Squeeze-Breakout environment (squeeze gate Lv1+)")
    env_fn = lambda: SqueezeBreakoutEnv(
        df=train_df, window_size=30, feature_columns=fc,
        feature_mean=mean, feature_std=std,
        cash_fraction=0.20, risk_per_trade_pct=0.01, reward_target_pct=0.03,
        require_squeeze=True, squeeze_min_level=1,
        random_start=True, min_episode_steps=200, episode_max_steps=1500,
    )
    train_env = DummyVecEnv([env_fn])

    print("\n[Model] PPO MLP [256,256,128] ent_coef=0.02")
    model = PPO(
        "MlpPolicy", train_env, verbose=1,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.02,
        policy_kwargs=dict(net_arch=dict(pi=[256,256,128], vf=[256,256,128])),
    )

    os.makedirs("experiment_5/checkpoints", exist_ok=True)
    ckpt = CheckpointCallback(save_freq=50_000, save_path="experiment_5/checkpoints/", name_prefix="exp5")

    TOTAL = 200_000
    print(f"\n[Train] {TOTAL:,} timesteps...")
    model.learn(total_timesteps=TOTAL, callback=ckpt)
    print("Training complete!")

    os.makedirs("experiment_5/models", exist_ok=True)
    model.save("experiment_5/models/exp5_squeeze")

    test_env = DummyVecEnv([lambda: SqueezeBreakoutEnv(
        df=test_df, window_size=30, feature_columns=fc,
        feature_mean=mean, feature_std=std,
        cash_fraction=0.20, risk_per_trade_pct=0.01, reward_target_pct=0.03,
        require_squeeze=True, squeeze_min_level=1,
        random_start=False, episode_max_steps=None,
    )])

    comparison = run_backtest(model, test_env, test_df)

    os.makedirs("experiment_5/results", exist_ok=True)
    results = {
        "timestamp": datetime.now().isoformat(),
        "rl": comparison["rl"], "buy_hold": comparison["buy_hold"], "random": comparison["random"],
        "config": {"squeeze_min_level": 1, "sl": "entry_candle_low/high", "exit": "9_ema_cross",
                   "rr": "1:3", "timesteps": TOTAL},
    }
    with open("experiment_5/results/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved.")

    print(f"\n{'='*60}")
    print(f"SUMMARY: RL={comparison['rl']['total_return_pct']:+.2f}% | "
          f"B&H={comparison['buy_hold']['return_pct']:+.1f}% | "
          f"Rand={comparison['random']['return_pct']:+.1f}%")
    return model, results


if __name__ == "__main__":
    main()
