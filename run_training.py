"""
Full training pipeline script for the RL Forex Trading Agent.
Runs training on EURUSD hourly data with the enhanced EMA crossover features.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from src.indicators import load_and_preprocess_data
from src.trading_env import ForexTradingEnv


def evaluate(model, df_eval, fc, mean, std, SL, TP):
    """Run evaluation and return equity curve + final equity."""
    env_eval = DummyVecEnv([
        lambda: ForexTradingEnv(
            df=df_eval,
            window_size=30,
            sl_options=SL,
            tp_options=TP,
            feature_columns=fc,
            feature_mean=mean,
            feature_std=std,
            random_start=False,
            episode_max_steps=None,
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0,
        )
    ])
    obs = env_eval.reset()
    eq_curve = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env_eval.step(action)
        if len(step_out) == 4:
            obs, reward, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, reward, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
        if isinstance(infos, (list, tuple)):
            info = infos[0]
        else:
            info = infos
        eq_curve.append(info.get("equity_usd", env_eval.get_attr("equity_usd")[0]))
        if done:
            break
    return eq_curve, eq_curve[-1] if eq_curve else 10000.0


def main():
    print("=" * 60)
    print("RL Forex Trading Agent - Full Training Pipeline")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/5] Loading and preprocessing data...")
    df, fc = load_and_preprocess_data("data/EURUSD_Hourly.csv")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"  Train bars: {len(train_df)}, Test bars: {len(test_df)}")
    print(f"  Features: {len(fc)}")

    # --- Normalization ---
    print("\n[2/5] Computing normalization statistics...")
    mean = train_df[fc].values.astype(np.float32).mean(axis=0)
    std = train_df[fc].values.astype(np.float32).std(axis=0)
    std[std == 0] = 1.0

    # --- Configuration ---
    SL = [10, 15, 20, 30, 40, 60, 90, 120]
    TP = [10, 15, 20, 30, 40, 60, 90, 120]
    TOTAL_STEPS = 200_000

    # --- Build environment ---
    print("\n[3/5] Building training environment...")
    env_fn = lambda: ForexTradingEnv(
        df=train_df,
        window_size=30,
        sl_options=SL,
        tp_options=TP,
        feature_columns=fc,
        feature_mean=mean,
        feature_std=std,
        episode_max_steps=2000,
        min_episode_steps=500,
        random_start=True,
        hold_reward_weight=0.01,
        open_penalty_pips=0.3,
        time_penalty_pips=0.005,
        unrealized_delta_weight=0.01,
        max_drawdown_pct=0.30,
        drawdown_penalty_weight=2.0,
    )
    train_env = DummyVecEnv([env_fn])

    # --- Create model ---
    print("\n[4/5] Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
        ),
    )

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix="ppo_forex",
    )

    # --- Train ---
    print(f"\n[5/5] Training for {TOTAL_STEPS:,} timesteps...")
    print("-" * 60)
    model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt_cb)
    print("Training complete!")

    # --- Save model ---
    os.makedirs("models", exist_ok=True)
    model.save("models/forex_trader_best")
    print("Model saved to models/forex_trader_best.zip")

    # --- Evaluate ---
    print("\nEvaluating on train and test sets...")
    test_eq, test_final = evaluate(model, test_df, fc, mean, std, SL, TP)
    train_eq, train_final = evaluate(model, train_df, fc, mean, std, SL, TP)

    initial = 10000.0
    train_ret = (train_final / initial - 1) * 100
    test_ret = (test_final / initial - 1) * 100

    # Metrics
    eq_arr = np.array(test_eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = (peak - eq_arr) / peak
    max_dd = np.max(dd) * 100 if len(dd) > 0 else 0.0

    returns = np.diff(eq_arr) / eq_arr[:-1]
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
    sharpe = (
        np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
        if len(returns) > 0 and np.std(returns) > 0
        else 0.0
    )

    # Downside deviation
    downside = returns[returns < 0]
    sortino = (
        np.mean(returns) / np.std(downside) * np.sqrt(252 * 24)
        if len(downside) > 0 and np.std(downside) > 0
        else 0.0
    )

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Train (IS)  - Final Equity: ${train_final:,.2f} | Return: {train_ret:.2f}%")
    print(f"  Test  (OOS) - Final Equity: ${test_final:,.2f} | Return: {test_ret:.2f}%")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"  Sharpe (approx): {sharpe:.2f}")
    print(f"  Sortino (approx): {sortino:.2f}")

    # --- Save results ---
    os.makedirs("results", exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "train_final_equity": float(train_final),
        "test_final_equity": float(test_final),
        "train_return_pct": float(train_ret),
        "test_return_pct": float(test_ret),
        "max_drawdown_pct": float(max_dd),
        "sharpe_approx": float(sharpe),
        "sortino_approx": float(sortino),
        "config": {
            "sl_options": SL,
            "tp_options": TP,
            "total_timesteps": TOTAL_STEPS,
            "features": fc,
        },
    }
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Metrics saved to results/metrics.json")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(train_eq, label=f"Train (IS) - ${train_final:,.0f}", alpha=0.8)
    ax1.plot(test_eq, label=f"Test (OOS) - ${test_final:,.0f}", alpha=0.8)
    ax1.axhline(y=initial, color="gray", linestyle="--", alpha=0.5, label="Initial ($10,000)")
    ax1.set_title("Equity Curves: In-Sample vs Out-of-Sample")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Equity ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    dd_pcts = dd * 100
    ax2.fill_between(range(len(dd_pcts)), 0, -dd_pcts, color="red", alpha=0.3)
    ax2.plot(-dd_pcts, color="red", linewidth=1)
    ax2.set_title(f"Test Drawdown (Max: {max_dd:.2f}%)")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("RL Forex Trading Agent - Training Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/training_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plot saved to results/training_results.png")

    print("\nDONE!")
    return model, results


if __name__ == "__main__":
    main()
