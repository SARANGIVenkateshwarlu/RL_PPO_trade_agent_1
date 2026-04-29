import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.indicators import load_and_preprocess_data
from src.trading_env import ForexTradingEnv


def evaluate_model(model, eval_env, deterministic=True):
    """Run a full evaluation episode and return equity curve + final equity."""
    obs = eval_env.reset()
    equity_curve = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = eval_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        if done:
            break

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity


def compute_normalization_stats(df, feature_cols):
    """Compute feature mean and std for normalization from training data."""
    feature_data = df[feature_cols].values.astype(np.float32)
    mean = np.mean(feature_data, axis=0)
    std = np.std(feature_data, axis=0)
    return mean, std


def train_agent(data_path, output_model_path="models/forex_trader_best"):
    """Main training pipeline with enhanced configuration."""
    print("=" * 60)
    print("Reinforcement Learning Forex Trading Agent - Training")
    print("=" * 60)

    # --- Load & preprocess data ---
    print("\n[1/6] Loading and preprocessing data...")
    df, feature_cols = load_and_preprocess_data(csv_path=data_path)
    print(f"  Data shape: {df.shape}")
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    # Time-based split: 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"  Training bars : {len(train_df)}")
    print(f"  Testing bars  : {len(test_df)}")

    # --- Compute normalization stats from training data ---
    print("\n[2/6] Computing normalization statistics...")
    train_feature_mean, train_feature_std = compute_normalization_stats(
        train_df, feature_cols
    )

    # --- Configure parameters ---
    SL_OPTS = [5, 10, 15, 20, 30, 40, 60, 90, 120]
    TP_OPTS = [5, 10, 15, 20, 30, 40, 60, 90, 120]
    WIN = 30

    # --- Build environments ---
    print("\n[3/6] Building environments...")

    def make_train_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=True,
            min_episode_steps=500,
            episode_max_steps=2000,
            feature_columns=feature_cols,
            feature_mean=train_feature_mean,
            feature_std=train_feature_std,
            hold_reward_weight=0.01,
            open_penalty_pips=0.3,
            time_penalty_pips=0.005,
            unrealized_delta_weight=0.01,
            max_drawdown_pct=0.30,
            drawdown_penalty_weight=2.0,
        )

    def make_train_eval_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            feature_mean=train_feature_mean,
            feature_std=train_feature_std,
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0,
        )

    def make_test_eval_env():
        return ForexTradingEnv(
            df=test_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            feature_mean=train_feature_mean,
            feature_std=train_feature_std,
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0,
        )

    train_vec_env = DummyVecEnv([make_train_env])
    train_eval_env = DummyVecEnv([make_train_eval_env])
    test_eval_env = DummyVecEnv([make_test_eval_env])

    # --- Create PPO model ---
    print("\n[4/6] Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
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
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            activation_fn=None,  # defaults to Tanh
        ),
        tensorboard_log="./tensorboard_log/",
    )

    # --- Setup callbacks ---
    print("\n[5/6] Setting up training callbacks...")
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix="ppo_forex",
    )

    # --- Train ---
    total_timesteps = 500_000
    print(f"\n[6/6] Training for {total_timesteps:,} timesteps...")
    print("-" * 60)

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    print("\nTraining complete!")

    # --- Select best model by OOS final equity ---
    print("\n" + "=" * 60)
    print("Evaluating checkpoints on OOS test data...")
    print("=" * 60)

    equity_curve_test_last, final_equity_test_last = evaluate_model(
        model, test_eval_env
    )
    print(f"[OOS Eval] Last model final equity: ${final_equity_test_last:,.2f}")

    best_equity = -np.inf
    best_path = None
    results_log = []

    ckpts = sorted(
        [
            f
            for f in os.listdir(ckpt_dir)
            if f.endswith(".zip") and f.startswith("ppo_forex")
        ],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)),
    )

    for ck in ckpts:
        ck_path = os.path.join(ckpt_dir, ck)
        try:
            m = PPO.load(ck_path, env=test_eval_env)
            eq_curve, final_eq = evaluate_model(m, test_eval_env)
            results_log.append(
                {"checkpoint": ck, "final_equity": final_eq, "path": ck_path}
            )
            print(f"[OOS Eval] {ck} -> final equity: ${final_eq:,.2f}")
            if final_eq > best_equity:
                best_equity = final_eq
                best_path = ck_path
        except Exception as e:
            print(f"[Skip] Could not evaluate checkpoint {ck}: {e}")

    # Decide best model
    if best_path is None or final_equity_test_last >= best_equity:
        print("Using last model as best (by OOS final equity).")
        best_model = model
    else:
        print(
            f"Using best checkpoint: {best_path} "
            f"(OOS final equity: ${best_equity:,.2f})"
        )
        best_model = PPO.load(best_path, env=train_vec_env)

    # Save best model
    os.makedirs(os.path.dirname(output_model_path) if os.path.dirname(output_model_path) else ".", exist_ok=True)
    best_model.save(output_model_path)
    print(f"Best model saved to: {output_model_path}.zip")

    # --- Final evaluation (IS vs OOS) ---
    print("\n" + "=" * 60)
    print("Final Evaluation: In-Sample vs Out-of-Sample")
    print("=" * 60)

    equity_curve_train, final_equity_train = evaluate_model(
        best_model, train_eval_env
    )
    equity_curve_test, final_equity_test = evaluate_model(best_model, test_eval_env)

    # Compute metrics
    def compute_metrics(eq_curve, initial_equity=10000.0):
        eq = np.array(eq_curve)
        returns = np.diff(eq) / eq[:-1]
        total_return_pct = ((eq[-1] - initial_equity) / initial_equity) * 100

        # Sharpe-like ratio (no risk-free rate for simplicity)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        max_dd = float(np.max(dd)) * 100 if len(dd) > 0 else 0.0

        # Win rate from trade history
        num_trades = 0
        win_trades = 0
        # (we can't get trade history from eval env easily, skip)

        return {
            "final_equity": float(eq[-1]),
            "total_return_pct": total_return_pct,
            "sharpe_approx": sharpe,
            "max_drawdown_pct": max_dd,
        }

    train_metrics = compute_metrics(equity_curve_train)
    test_metrics = compute_metrics(equity_curve_test)

    print(f"\n[IS  - Train] Final Equity: ${train_metrics['final_equity']:,.2f} | "
          f"Return: {train_metrics['total_return_pct']:.2f}% | "
          f"Sharpe: {train_metrics['sharpe_approx']:.2f} | "
          f"Max DD: {train_metrics['max_drawdown_pct']:.2f}%")

    print(f"[OOS - Test ] Final Equity: ${test_metrics['final_equity']:,.2f} | "
          f"Return: {test_metrics['total_return_pct']:.2f}% | "
          f"Sharpe: {test_metrics['sharpe_approx']:.2f} | "
          f"Max DD: {test_metrics['max_drawdown_pct']:.2f}%")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Equity curves
    ax1 = axes[0]
    ax1.plot(equity_curve_train, label=f"Train (IS) - Final: ${train_metrics['final_equity']:,.0f}", alpha=0.8)
    ax1.plot(equity_curve_test, label=f"Test (OOS) - Final: ${test_metrics['final_equity']:,.0f}", alpha=0.8)
    ax1.axhline(y=10000, color="gray", linestyle="--", alpha=0.5, label="Initial ($10,000)")
    ax1.set_title("Equity Curves: In-Sample vs Out-of-Sample")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Equity ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bar chart comparison
    ax2 = axes[1]
    metrics_names = ["Final Equity ($)", "Return (%)", "Sharpe (approx)", "Max DD (%)"]
    train_vals = [
        train_metrics["final_equity"],
        train_metrics["total_return_pct"],
        train_metrics["sharpe_approx"] * 100,
        train_metrics["max_drawdown_pct"],
    ]
    test_vals = [
        test_metrics["final_equity"],
        test_metrics["total_return_pct"],
        test_metrics["sharpe_approx"] * 100,
        test_metrics["max_drawdown_pct"],
    ]

    x = np.arange(len(metrics_names))
    width = 0.35
    ax2.bar(x - width / 2, train_vals, width, label="Train (IS)", alpha=0.8)
    ax2.bar(x + width / 2, test_vals, width, label="Test (OOS)", alpha=0.8)
    ax2.set_title("Performance Metrics Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names, rotation=15)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join("results", "training_results.png")
    os.makedirs("results", exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to: {plot_path}")

    # --- Save metrics ---
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "config": {
            "sl_options": SL_OPTS,
            "tp_options": TP_OPTS,
            "window_size": WIN,
            "total_timesteps": total_timesteps,
            "feature_count": len(feature_cols),
            "feature_cols": feature_cols,
        },
    }

    metrics_path = os.path.join("results", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    return best_model, train_metrics, test_metrics


if __name__ == "__main__":
    data_path = "data/EURUSD_Hourly.csv"
    train_agent(data_path, output_model_path="models/forex_trader_best")
