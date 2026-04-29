"""
RL Stock Trading Agent Training (EMA Crossover + BB Squeeze, Buy Only)

Downloads stock data, trains PPO agent, evaluates performance.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from src.stock_indicators import load_and_process_stocks
from src.stock_env import StockTradingEnv


def split_data(df, feature_cols, train_ratio=0.7):
    """Time-based split respecting symbol boundaries."""
    symbols = sorted(df["Symbol"].unique())
    train_parts = []
    test_parts = []

    for sym in symbols:
        sym_df = df[df["Symbol"] == sym].sort_values("Date").copy()
        n = len(sym_df)
        split_n = int(n * train_ratio)
        train_parts.append(sym_df.iloc[:split_n])
        test_parts.append(sym_df.iloc[split_n:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df


def evaluate(model, eval_df, feature_cols, mean, std, SL, TP):
    env = DummyVecEnv([lambda: StockTradingEnv(
        df=eval_df, window_size=30,
        sl_options_pct=SL, tp_options_pct=TP,
        feature_columns=feature_cols,
        feature_mean=mean, feature_std=std,
        random_start=False, episode_max_steps=None,
        hold_reward_weight=0.0, buy_penalty_pct=0.0, time_penalty_pct=0.0,
    )])
    obs = env.reset()
    eq_curve = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(action)
        if len(step_out) == 4:
            obs, reward, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, reward, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq_curve.append(info.get("equity", env.get_attr("equity")[0]))
        if done:
            break
    return eq_curve, eq_curve[-1]


def main():
    print("=" * 60)
    print("RL Stock Trading Agent - EMA Crossover + BB Squeeze")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/6] Loading and processing stock data...")
    df, feature_cols = load_and_process_stocks("data/stocks/combined_stocks.parquet")
    print(f"  Total rows: {len(df)}, Symbols: {df['Symbol'].nunique()}")

    # Split
    train_df, test_df = split_data(df, feature_cols, train_ratio=0.7)
    print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    # Normalization
    print("\n[2/6] Computing normalization...")
    mean = train_df[feature_cols].values.astype(np.float32).mean(axis=0)
    std = train_df[feature_cols].values.astype(np.float32).std(axis=0)
    std[std == 0] = 1.0

    # Configuration
    SL_PCT = [2, 3, 5, 7, 10]     # Stop loss %
    TP_PCT = [3, 5, 8, 12, 20]    # Take profit %
    TOTAL_STEPS = 200_000

    print("\n[3/6] Building environment...")
    env_fn = lambda: StockTradingEnv(
        df=train_df, window_size=30,
        sl_options_pct=SL_PCT, tp_options_pct=TP_PCT,
        feature_columns=feature_cols,
        feature_mean=mean, feature_std=std,
        random_start=True, min_episode_steps=200, episode_max_steps=1500,
        hold_reward_weight=0.05, buy_penalty_pct=0.1, time_penalty_pct=0.005,
        max_drawdown_pct=0.25, drawdown_penalty_weight=2.0,
    )
    train_env = DummyVecEnv([env_fn])

    print("\n[4/6] Creating PPO model...")
    model = PPO(
        "MlpPolicy", train_env, verbose=1,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
    )

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path="./checkpoints/", name_prefix="stock_ppo")

    print(f"\n[5/6] Training {TOTAL_STEPS:,} steps...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt_cb)
    print("Training complete!")

    # Save
    os.makedirs("models", exist_ok=True)
    model.save("models/stock_trader_best")
    print("Model saved: models/stock_trader_best.zip")

    # Evaluate
    print("\n[6/6] Evaluating...")
    train_eq, train_final = evaluate(model, train_df, feature_cols, mean, std, SL_PCT, TP_PCT)
    test_eq, test_final = evaluate(model, test_df, feature_cols, mean, std, SL_PCT, TP_PCT)

    init = 100000.0
    train_ret = (train_final / init - 1) * 100
    test_ret = (test_final / init - 1) * 100

    test_arr = np.array(test_eq)
    peak = np.maximum.accumulate(test_arr)
    dd = (peak - test_arr) / peak
    max_dd = np.max(dd) * 100

    returns = np.diff(test_arr) / test_arr[:-1]
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0.0

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Train (IS):  ${train_final:,.2f}  ({train_ret:+.2f}%)")
    print(f"  Test (OOS):  ${test_final:,.2f}  ({test_ret:+.2f}%)")
    print(f"  Max DD:      {max_dd:.2f}%")
    print(f"  Sharpe:      {sharpe:.2f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        "timestamp": datetime.now().isoformat(),
        "train_final": float(train_final), "test_final": float(test_final),
        "train_return_pct": float(train_ret), "test_return_pct": float(test_ret),
        "max_drawdown_pct": float(max_dd), "sharpe": float(sharpe),
        "config": {"sl_pct": SL_PCT, "tp_pct": TP_PCT, "timesteps": TOTAL_STEPS, "features": feature_cols},
    }
    with open("results/stock_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(train_eq, label=f"Train (IS) ${train_final:,.0f}", alpha=0.8)
    ax1.plot(test_eq, label=f"Test (OOS) ${test_final:,.0f}", alpha=0.8)
    ax1.axhline(y=init, color="gray", linestyle="--", alpha=0.5, label=f"Initial ${init:,.0f}")
    ax1.set_title("Equity Curves")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.fill_between(range(len(dd)), 0, -dd * 100, color="red", alpha=0.3)
    ax2.plot(-dd * 100, color="red", linewidth=1)
    ax2.set_title(f"Test Drawdown (Max: {max_dd:.2f}%)")
    ax2.grid(alpha=0.3)
    plt.suptitle("RL Stock Agent — EMA Crossover + BB Squeeze", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/stock_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plots saved to results/stock_results.png")
    print("\nDONE!")

    return model, results


if __name__ == "__main__":
    main()
