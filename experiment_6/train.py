"""
Experiment 6: Enterprise-Grade RL Trading Agent — Main Training Pipeline

Pipeline:
  1. Load & preprocess data
  2. Expert pretraining (behavioral cloning, 10k steps)
  3. Optuna hyperparameter optimization (optional)
  4. PPO fine-tuning with best parameters
  5. Validation & comprehensive plotting
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from experiment_6.src.indicators import compute_indicators, get_feature_columns
from experiment_6.src.enterprise_env import EnterpriseTradingEnv
from experiment_6.src.pretrain import ExpertPolicy, collect_expert_demonstrations, pretrain_policy
from experiment_6.src.optuna_optimizer import create_ppo_model, optimize_hyperparameters
from experiment_3.src.market_regime import compute_sp500_regime, merge_regime_to_stocks


# ─── Custom Callbacks ─────────────────────────────────────────────

class MetricsCallback(BaseCallback):
    """Track training/validation metrics during PPO training."""
    def __init__(self, eval_env, eval_freq=5000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.metrics_history = []
        self.loss_history = []

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                pass
            self.loss_history.append({
                "step": self.num_timesteps,
                "loss": getattr(self.model, 'ep_info_buffer', None),
            })

        if self.n_calls % self.eval_freq == 0:
            eq, trades = quick_eval(self.model, self.eval_env)
            init = 100000.0
            ret = (eq[-1] / init - 1) * 100 if len(eq) > 0 else 0

            dr = np.diff(eq) / eq[:-1]; dr = dr[~np.isnan(dr)&~np.isinf(dr)]
            sharpe = np.mean(dr)/np.std(dr)*np.sqrt(252) if len(dr)>0 and np.std(dr)>0 else 0

            peak = np.maximum.accumulate(eq); dd = np.max((peak-eq)/peak)*100

            self.metrics_history.append({
                "step": self.num_timesteps,
                "return_pct": ret, "sharpe": sharpe, "max_dd": dd,
                "n_trades": len(trades),
            })
            if self.verbose:
                print(f"[{self.num_timesteps:>7d}] Ret={ret:+.2f}% Sharpe={sharpe:.2f} DD={dd:.1f}% Trades={len(trades)}")

        return True


def quick_eval(model, eval_env, max_steps=2000):
    eq, trades = [], []
    obs = eval_env.reset()
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        step_out = eval_env.step(action)
        if len(step_out) == 4:
            obs, _, dones, infos = step_out; done = dones[0] if hasattr(dones,'__getitem__') else dones
        else:
            obs, _, terminated, truncated, infos = step_out
            done = (terminated[0] | truncated[0]) if hasattr(terminated,'__getitem__') else (terminated | truncated)
        info = infos[0] if isinstance(infos,(list,tuple)) else infos
        eq.append(info.get("equity", eval_env.get_attr("equity")[0]))
        if done: break
    trades = eval_env.get_attr("trade_history")[0] if hasattr(eval_env,"get_attr") else []
    return np.array(eq), trades


# ─── Data Loading ──────────────────────────────────────────────────

def load_data():
    print("=" * 60)
    print("Experiment 6: Enterprise RL Trading Agent")
    print("=" * 60)

    sp_path = "experiment_3/data/SP500_daily.csv"
    sp_df = pd.read_csv(sp_path, parse_dates=["Date"])
    sp_df = compute_sp500_regime(sp_df)

    st_path = "experiment_3/data/stocks_daily.parquet"
    st_df = pd.read_parquet(st_path)
    st_df = merge_regime_to_stocks(st_df, sp_df)

    print("[Data] Computing indicators...")
    results = []
    for sym in sorted(st_df["Symbol"].unique()):
        sdf = st_df[st_df["Symbol"] == sym].copy()
        if len(sdf) < 200: continue
        sdf = compute_indicators(sdf); sdf["Symbol"] = sym
        results.append(sdf)

    df = pd.concat(results, ignore_index=True).sort_values(["Symbol","Date"]).reset_index(drop=True)
    fc = get_feature_columns(df)
    print(f"  {len(df)} rows | {df['Symbol'].nunique()} symbols | {len(fc)} features")
    if "squeeze_signal" in df.columns:
        sqz = df["squeeze_signal"].value_counts().to_dict()
        print(f"  Squeeze: Mod={sqz.get(1,0)} Full={sqz.get(2,0)}")

    return df, fc


def split_data(df, train_r=0.7, val_r=0.15):
    train, val, test = [], [], []
    for sym in sorted(df["Symbol"].unique()):
        sd = df[df["Symbol"]==sym].sort_values("Date")
        n = len(sd); n1 = int(n*train_r); n2 = int(n*(train_r+val_r))
        train.append(sd.iloc[:n1]); val.append(sd.iloc[n1:n2]); test.append(sd.iloc[n2:])
    return (pd.concat(train).reset_index(drop=True),
            pd.concat(val).reset_index(drop=True),
            pd.concat(test).reset_index(drop=True))


# ─── Training ──────────────────────────────────────────────────────

def train_pipeline(run_optuna: bool = False, n_optuna_trials: int = 20,
                   pretrain_steps: int = 10000, finetune_steps: int = 200000,
                   device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Main training pipeline."""

    # 1. Load data
    df, fc = load_data()
    train_df, val_df, test_df = split_data(df)
    print(f"\n[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Normalization
    mean = train_df[fc].values.astype(np.float32).mean(0)
    std = train_df[fc].values.astype(np.float32).std(0)
    std[std == 0] = 1.0

    # Env factories
    def make_env(data, random_start=True, ep_max=1500):
        return EnterpriseTradingEnv(
            df=data, window_size=30, feature_columns=fc,
            feature_mean=mean, feature_std=std,
            cash_fraction=0.20, risk_per_trade_pct=0.01, reward_target_pct=0.03,
            require_squeeze=True, squeeze_min_level=1, use_action_masking=True,
            random_start=random_start, min_episode_steps=200, episode_max_steps=ep_max,
        )

    train_env = DummyVecEnv([lambda: make_env(train_df, True, 1500)])
    val_env = DummyVecEnv([lambda: make_env(val_df, False, None)])
    test_env = DummyVecEnv([lambda: make_env(test_df, False, None)])

    # 2. Expert Pretraining
    print(f"\n[Pretrain] Collecting expert demonstrations ({pretrain_steps} steps)...")
    expert_env = DummyVecEnv([lambda: make_env(train_df, True, 500)])
    expert = ExpertPolicy(ema_spread_threshold=0.0, squeeze_min=1)
    obs_data, act_data = collect_expert_demonstrations(expert_env, expert, pretrain_steps, fc)

    action_counts = np.bincount(act_data, minlength=3)
    print(f"  Expert actions: HOLD={action_counts[0]} BUY={action_counts[1]} SELL={action_counts[2]}")

    # 3. Optuna optimization (optional)
    best_params = {}
    if run_optuna:
        print(f"\n[Optuna] Running hyperparameter optimization ({n_optuna_trials} trials)...")
        study = optimize_hyperparameters(
            lambda: DummyVecEnv([lambda: make_env(train_df, True, 800)]),
            lambda: DummyVecEnv([lambda: make_env(val_df, False, None)]),
            n_trials=n_optuna_trials,
            n_finetune_steps=30000,
        )
        best_params = study.best_params
    else:
        # Default best params
        best_params = {
            "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 128,
            "gamma": 0.995, "gae_lambda": 0.98, "ent_coef": 0.01,
            "clip_range": 0.2, "n_epochs": 10,
            "pi_layers": [256, 256, 128], "vf_layers": [256, 256, 128],
        }

    # 4. Create model with best params
    print(f"\n[Model] Creating PPO with best params: {best_params}")
    model = PPO(
        "MlpPolicy", train_env, verbose=1,
        learning_rate=best_params.get("learning_rate", 3e-4),
        n_steps=best_params.get("n_steps", 2048),
        batch_size=best_params.get("batch_size", 128),
        n_epochs=best_params.get("n_epochs", 10),
        gamma=best_params.get("gamma", 0.995),
        gae_lambda=best_params.get("gae_lambda", 0.98),
        clip_range=best_params.get("clip_range", 0.2),
        ent_coef=best_params.get("ent_coef", 0.01),
        policy_kwargs=dict(net_arch=dict(
            pi=best_params.get("pi_layers", [256, 256, 128]),
            vf=best_params.get("vf_layers", [256, 256, 128]),
        )),
        tensorboard_log=None,
    )

    # 5. Pretrain policy
    print(f"\n[Pretrain] Behavioral cloning from expert ({len(obs_data)} samples)...")
    pretrain_history = pretrain_policy(model, obs_data, act_data, epochs=15, batch_size=64, lr=1e-3, device=device)

    # Save pretrained model
    os.makedirs("experiment_6/models", exist_ok=True)
    model.save("experiment_6/models/pretrained_policy")
    print("[Pretrain] Pretrained model saved.")

    # 6. PPO Fine-tuning
    print(f"\n[Finetune] PPO training for {finetune_steps:,} steps...")
    os.makedirs("experiment_6/checkpoints", exist_ok=True)

    metrics_cb = MetricsCallback(val_env, eval_freq=10000, verbose=1)
    ckpt_cb = CheckpointCallback(save_freq=50000, save_path="experiment_6/checkpoints/", name_prefix="exp6")

    start_time = time.time()
    model.learn(total_timesteps=finetune_steps, callback=[metrics_cb, ckpt_cb])
    train_time = time.time() - start_time
    print(f"[Finetune] Complete in {train_time:.1f}s ({train_time/60:.1f} min)")

    # 7. Final evaluation
    print("\n[Eval] Final evaluation on test set...")
    test_eq, test_trades = quick_eval(model, test_env, max_steps=10000)

    init = 100000.0
    test_ret = (test_eq[-1] / init - 1) * 100
    dr = np.diff(test_eq)/test_eq[:-1]; dr = dr[~np.isnan(dr)&~np.isinf(dr)]
    test_sharpe = np.mean(dr)/np.std(dr)*np.sqrt(252) if len(dr)>0 and np.std(dr)>0 else 0
    peak = np.maximum.accumulate(test_eq); test_dd = np.max((peak-test_eq)/peak)*100

    val_eq, val_trades = quick_eval(model, val_env, max_steps=5000)
    val_ret = (val_eq[-1]/init - 1)*100

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Train time: {train_time:.0f}s | Device: {device}")
    print(f"  Val Return:  {val_ret:+.2f}% | Val Trades: {len(val_trades)}")
    print(f"  Test Return: {test_ret:+.2f}% | Test Sharpe: {test_sharpe:.2f} | Max DD: {test_dd:.2f}%")
    print(f"  Test Trades: {len(test_trades)}")

    # 8. Save all results
    os.makedirs("experiment_6/results", exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "train_time_seconds": train_time, "device": device,
        "val_return_pct": float(val_ret), "test_return_pct": float(test_ret),
        "test_sharpe": float(test_sharpe), "test_max_dd": float(test_dd),
        "n_val_trades": len(val_trades), "n_test_trades": len(test_trades),
        "pretrain_steps": pretrain_steps, "finetune_steps": finetune_steps,
        "best_params": best_params,
        "pretrain_history": pretrain_history,
        "training_metrics": metrics_cb.metrics_history,
    }
    with open("experiment_6/results/final_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    model.save("experiment_6/models/exp6_final")

    # 9. Plotting
    plot_results(pretrain_history, metrics_cb.metrics_history,
                 test_eq, val_eq, test_trades, val_trades, init)

    return model, results


def plot_results(pretrain_hist, train_metrics, test_eq, val_eq,
                 test_trades, val_trades, init=100000.0):
    """Generate comprehensive plots."""
    fig, axes = plt.subplots(3, 3, figsize=(22, 16))

    # 1. Pretrain Loss & Accuracy
    ax = axes[0, 0]
    if pretrain_hist:
        epochs = [h["epoch"] for h in pretrain_hist]
        losses = [h["loss"] for h in pretrain_hist]
        accs = [h["accuracy"] for h in pretrain_hist]
        ax2 = ax.twinx()
        ax.plot(epochs, losses, "b-o", markersize=4, label="Loss")
        ax2.plot(epochs, [a*100 for a in accs], "g-s", markersize=4, label="Accuracy %")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax2.set_ylabel("Accuracy (%)")
        ax.set_title("Pretraining: Loss & Accuracy vs Epoch")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No pretrain data", ha="center"); ax.set_title("Pretraining")

    # 2. Training Metrics (Return vs Step)
    ax = axes[0, 1]
    if train_metrics:
        steps = [m["step"] for m in train_metrics]
        rets = [m["return_pct"] for m in train_metrics]
        ax.plot(steps, rets, "b-", linewidth=2, label="Val Return %")
        ax.axhline(y=0, color="r", ls="--", alpha=0.5)
        ax.fill_between(steps, 0, rets, where=(np.array(rets)>=0), alpha=0.2, color="green")
        ax.fill_between(steps, 0, rets, where=(np.array(rets)<0), alpha=0.2, color="red")
        ax.set_xlabel("Timesteps"); ax.set_ylabel("Return (%)")
        ax.set_title("Validation Return vs Training Steps")
        ax.legend(); ax.grid(alpha=0.3)

    # 3. Training Sharpe vs Step
    ax = axes[0, 2]
    if train_metrics:
        sharpes = [m["sharpe"] for m in train_metrics]
        ax.plot(steps, sharpes, "g-", linewidth=2)
        ax.axhline(y=0, color="r", ls="--", alpha=0.5)
        ax.set_xlabel("Timesteps"); ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Validation Sharpe vs Training Steps")
        ax.grid(alpha=0.3)

    # 4. Training/Validation Equity Curves
    ax = axes[1, 0]
    ax.plot(val_eq, label=f"Validation ({len(val_trades)} trades)", alpha=0.8, lw=1.5, color="orange")
    ax.plot(test_eq, label=f"Test ({len(test_trades)} trades)", alpha=0.8, lw=1.5, color="blue")
    ax.axhline(y=init, color="gray", ls="--", alpha=0.5, label=f"Initial ${init:,.0f}")
    ax.set_xlabel("Steps"); ax.set_ylabel("Equity ($)")
    ax.set_title("Equity Curves: Validation vs Test")
    ax.legend(); ax.grid(alpha=0.3)

    # 5. Drawdown (Test)
    ax = axes[1, 1]
    peak = np.maximum.accumulate(test_eq)
    dd = (peak - test_eq) / peak * 100
    ax.fill_between(range(len(dd)), 0, -dd, color="red", alpha=0.3)
    ax.plot(-dd, "r-", lw=1)
    ax.set_title(f"Test Drawdown (Max: {np.max(dd):.2f}%)")
    ax.set_xlabel("Steps"); ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)

    # 6. Trade PnL (Test)
    ax = axes[1, 2]
    if test_trades:
        pnls = [t.get("pnl_pct", 0) for t in test_trades]
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, edgecolor="black", lw=0.3)
        ax.axhline(y=0, color="black", lw=0.5)
        wins = sum(1 for p in pnls if p > 0)
        ax.set_title(f"Test Trades ({len(pnls)} trades, WR: {wins/len(pnls)*100:.1f}%)")
        ax.set_xlabel("Trade #"); ax.set_ylabel("PnL %")
        ax.grid(alpha=0.3)

    # 7. Rolling Sharpe (Test)
    ax = axes[2, 0]
    rets = np.diff(test_eq) / test_eq[:-1] * 100
    rets = rets[~np.isnan(rets)&~np.isinf(rets)]
    if len(rets) > 20:
        roll_mean = pd.Series(rets).rolling(20).mean() * 20 * 252 / 100
        roll_std = pd.Series(rets).rolling(20).std() * np.sqrt(20*252) / 100
        roll_sharpe = roll_mean / roll_std
        ax.plot(roll_sharpe.values, "b-", lw=1)
        ax.axhline(y=0, color="r", ls="--", alpha=0.5)
        ax.set_title("Rolling Sharpe (20-day)")
        ax.set_xlabel("Steps"); ax.set_ylabel("Sharpe")
        ax.grid(alpha=0.3)

    # 8. Returns Distribution (Test)
    ax = axes[2, 1]
    if len(rets) > 0:
        ax.hist(rets, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        ax.axvline(x=0, color="red", ls="--")
        ax.set_title(f"Test Returns Distribution (Mean: {np.mean(rets):.4f}%)")
        ax.set_xlabel("Return (%)"); ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)

    # 9. Trade Exit Reasons (Test)
    ax = axes[2, 2]
    if test_trades:
        reasons = {}
        for t in test_trades:
            r = t.get("reason", "?")
            reasons[r] = reasons.get(r, 0) + 1
        labels = list(reasons.keys()); sizes = list(reasons.values())
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("Trade Exit Reasons")

    plt.suptitle("Experiment 6: Enterprise RL Trading Agent — Comprehensive Report",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("experiment_6/results/comprehensive_report.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plot] Saved comprehensive_report.png")

    # Additional: Training loss curve
    if train_metrics:
        fig2, ax = plt.subplots(figsize=(12, 4))
        dd_vals = [m["max_dd"] for m in train_metrics]
        ax.plot(steps, dd_vals, "r-", lw=2, label="Max Drawdown %")
        ax.fill_between(steps, 0, dd_vals, alpha=0.2, color="red")
        ax.set_xlabel("Timesteps"); ax.set_ylabel("Drawdown (%)")
        ax.set_title("Validation Drawdown vs Training Steps")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("experiment_6/results/training_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[Plot] Saved training_curves.png")


# ─── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna", action="store_true", help="Run Optuna optimization")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials")
    parser.add_argument("--pretrain", type=int, default=10000, help="Pretraining steps")
    parser.add_argument("--finetune", type=int, default=200000, help="Fine-tuning steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_pipeline(
        run_optuna=args.optuna,
        n_optuna_trials=args.trials,
        pretrain_steps=args.pretrain,
        finetune_steps=args.finetune,
        device=args.device,
    )
