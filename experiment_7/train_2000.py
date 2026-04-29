"""
Experiment 7: 2000-Stock Training Pipeline
Inside Bar Trend-Following Strategy — scaled to large universe.
"""
import os
import sys
import time
import json
import gc
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

from experiment_7.src.indicators import compute_indicators, get_feature_columns
from experiment_7.src.enterprise_env import InsideBarTradingEnv
from experiment_7.src.pretrain import ExpertPolicy, collect_expert_demonstrations, pretrain_policy
from experiment_7.src.optuna_optimizer import create_ppo_model, optimize_hyperparameters
from experiment_3.src.market_regime import compute_sp500_regime, merge_regime_to_stocks


def load_and_prepare_2000(sp500_path: str = "experiment_7/data/SP500_daily.csv",
                          stocks_path: str = "experiment_7/data/stocks_2000_daily.parquet",
                          max_symbols: int = 2000,
                          chunk_size: int = 200):
    print("=" * 70)
    print("Experiment 7: 2000-Stock Inside Bar Trend-Following RL Agent")
    print("=" * 70)

    print("\n[SP500] Loading regime data...")
    sp_df = pd.read_csv(sp500_path, parse_dates=["Date"])
    sp_df = compute_sp500_regime(sp_df)

    print(f"[Stocks] Loading from {stocks_path}...")
    st_df = pd.read_parquet(stocks_path)
    print(f"  Raw: {len(st_df):,} rows, {st_df['Symbol'].nunique()} symbols")

    print("[Data] Normalizing dates for merge...")
    if "Date" in st_df.columns:
        st_df["Date"] = pd.to_datetime(st_df["Date"]).dt.normalize()
    if "Date" in sp_df.columns:
        sp_df["Date"] = pd.to_datetime(sp_df["Date"]).dt.normalize()
    st_df = merge_regime_to_stocks(st_df, sp_df)

    symbols = sorted(st_df["Symbol"].unique())[:max_symbols]
    n_chunks = (len(symbols) + chunk_size - 1) // chunk_size

    print(f"[Indicators] Processing {len(symbols)} symbols in {n_chunks} chunks...")

    all_chunks = []
    gate_stats = 0
    total_bars = 0

    for i in range(n_chunks):
        chunk_syms = symbols[i * chunk_size:(i + 1) * chunk_size]
        chunk_data = st_df[st_df["Symbol"].isin(chunk_syms)].copy()

        results = []
        for sym in chunk_syms:
            sdf = chunk_data[chunk_data["Symbol"] == sym].copy()
            if len(sdf) < 200:
                continue
            sdf = compute_indicators(sdf)
            sdf["Symbol"] = sym
            results.append(sdf)

        if results:
            chunk_df = pd.concat(results, ignore_index=True)
            all_chunks.append(chunk_df)
            if "entry_gate" in chunk_df.columns:
                gate_stats += chunk_df["entry_gate"].sum()
                total_bars += len(chunk_df)

        del chunk_data, results
        gc.collect()
        print(f"  Chunk {i+1}/{n_chunks}: {len(chunk_syms)} symbols processed "
              f"({sum(len(c) for c in all_chunks):,} total rows)")

    combined = pd.concat(all_chunks, ignore_index=True)
    combined = combined.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    fc = get_feature_columns(combined)

    print(f"\n[Summary]")
    print(f"  Rows: {len(combined):,} | Symbols: {combined['Symbol'].nunique()} | Features: {len(fc)}")
    print(f"  Entry gates active: {gate_stats:,} ({100*gate_stats/max(total_bars,1):.2f}% of bars)")

    return combined, fc


def split_data_2000(df, train_ratio=0.7, val_ratio=0.15):
    train_parts, val_parts, test_parts = [], [], []
    for sym in sorted(df["Symbol"].unique()):
        sd = df[df["Symbol"] == sym].sort_values("Date")
        n = len(sd)
        n1 = int(n * train_ratio)
        n2 = int(n * (train_ratio + val_ratio))
        train_parts.append(sd.iloc[:n1])
        val_parts.append(sd.iloc[n1:n2])
        test_parts.append(sd.iloc[n2:])
    return (pd.concat(train_parts, ignore_index=True),
            pd.concat(val_parts, ignore_index=True),
            pd.concat(test_parts, ignore_index=True))


def evaluate_full(model, eval_df, feature_cols, mean, std, max_steps=5000):
    env = DummyVecEnv([lambda: InsideBarTradingEnv(
        df=eval_df, window_size=30, feature_columns=feature_cols,
        feature_mean=mean, feature_std=std,
        risk_per_trade_pct=0.10,
        require_entry_gate=True, use_action_masking=True,
        random_start=False, episode_max_steps=None,
    )])
    obs = env.reset()
    eq_curve = []
    while len(eq_curve) < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(action)
        if len(step_out) == 4:
            obs, _, dones, infos = step_out
            done = dones[0] if hasattr(dones, '__getitem__') else dones
        else:
            obs, _, terminated, truncated, infos = step_out
            done = (terminated[0] | truncated[0]) if hasattr(terminated, '__getitem__') else (terminated | truncated)
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq_curve.append(info.get("equity", env.get_attr("equity")[0]))
        if done:
            break
    trades = env.get_attr("trade_history")[0] if hasattr(env, "get_attr") else []
    env.close()
    return np.array(eq_curve), trades


def compute_metrics(eq_curve, trades, initial=100000.0):
    eq = np.array(eq_curve)
    final = eq[-1]
    ret_pct = (final / initial - 1) * 100
    rets = np.diff(eq) / eq[:-1]
    rets = rets[~np.isnan(rets) & ~np.isinf(rets)]
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if len(rets) > 0 and np.std(rets) > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    max_dd = np.max((peak - eq) / peak) * 100
    n_trades = len(trades)
    if n_trades > 0:
        pnls = [t.get("pnl_usd", 0) / initial * 100 for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n_trades * 100
        avg_win = np.mean([p for p in pnls if p > 0]) if wins > 0 else 0
        avg_loss = np.mean([p for p in pnls if p < 0]) if (n_trades - wins) > 0 else 0
    else:
        wins = 0; wr = 0; avg_win = 0; avg_loss = 0; pnls = []
    return {
        "final_equity": float(final),
        "return_pct": float(ret_pct),
        "sharpe": float(sharpe),
        "max_dd_pct": float(max_dd),
        "num_trades": n_trades,
        "win_rate": float(wr),
        "avg_win_pct": float(avg_win),
        "avg_loss_pct": float(avg_loss),
    }


def train_2000_pipeline(run_optuna=False, n_optuna_trials=20, pretrain_steps=10000,
                        finetune_steps=200000, max_symbols=2000,
                        device="cuda" if torch.cuda.is_available() else "cpu"):
    df, fc = load_and_prepare_2000(max_symbols=max_symbols)
    print("\n[Split] Creating train/val/test splits...")
    train_df, val_df, test_df = split_data_2000(df)
    print(f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} | Test: {len(test_df):,}")
    del df; gc.collect()

    print("\n[Normalize] Computing stats from training data...")
    mean = train_df[fc].values.astype(np.float32).mean(axis=0)
    std = train_df[fc].values.astype(np.float32).std(axis=0)
    std[std == 0] = 1.0

    def make_env(data, random_start=True, ep_max=1500):
        return InsideBarTradingEnv(
            df=data, window_size=30, feature_columns=fc,
            feature_mean=mean, feature_std=std,
            risk_per_trade_pct=0.10,
            require_entry_gate=True, use_action_masking=True,
            random_start=random_start, min_episode_steps=200, episode_max_steps=ep_max,
        )

    expert_env = DummyVecEnv([lambda: make_env(train_df, True, 500)])
    train_env = DummyVecEnv([lambda: make_env(train_df, True, 1500)])
    val_env = DummyVecEnv([lambda: make_env(val_df, False, None)])
    test_env = DummyVecEnv([lambda: make_env(test_df, False, None)])

    print(f"\n[Pretrain] Expert demonstrations ({pretrain_steps:,} steps)...")
    expert = ExpertPolicy(require_gate=True)
    obs_data, act_data = collect_expert_demonstrations(expert_env, expert, pretrain_steps, fc)
    counts = np.bincount(act_data, minlength=2)
    print(f"  Actions: HOLD={counts[0]} BUY={counts[1]}")

    best_params = {}
    if run_optuna:
        print(f"\n[Optuna] Hyperparameter optimization ({n_optuna_trials} trials)...")
        study = optimize_hyperparameters(
            lambda: DummyVecEnv([lambda: make_env(train_df, True, 800)]),
            lambda: DummyVecEnv([lambda: make_env(val_df, False, None)]),
            n_trials=n_optuna_trials, n_finetune_steps=30000,
        )
        best_params = study.best_params
    else:
        best_params = {
            "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 128,
            "gamma": 0.995, "gae_lambda": 0.98, "ent_coef": 0.01,
            "clip_range": 0.2, "n_epochs": 10,
            "pi_layers": [256, 256, 128], "vf_layers": [256, 256, 128],
        }

    print(f"\n[Model] PPO with params: {best_params}")
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
    )

    print(f"\n[Pretrain] Behavioral cloning ({len(obs_data):,} samples)...")
    pretrain_hist = pretrain_policy(model, obs_data, act_data, epochs=15, batch_size=128, lr=1e-3, device=device)
    os.makedirs("experiment_7/models", exist_ok=True)
    model.save("experiment_7/models/pretrained_2000")

    print(f"\n[Finetune] PPO training {finetune_steps:,} steps on {device}...")

    eval_history = []

    class EvalCallback:
        def __init__(self, model_ref, val_env, fc, mean, std, freq=20000):
            self.model = model_ref
            self.val_env = val_env
            self.fc = fc; self.m = mean; self.s = std
            self.freq = freq
            self.history = []

        def evaluate(self, step):
            eq, trades = evaluate_full(self.model, test_df if step % 40000 == 0 else val_df,
                                       self.fc, self.m, self.s, max_steps=3000)
            m = compute_metrics(eq, trades, 100000.0)
            self.history.append({"step": step, **m})
            label = "TEST" if step % 40000 == 0 else "VAL"
            print(f"  [{step:>7d}] {label} | Ret={m['return_pct']:+.2f}% "
                  f"Sharpe={m['sharpe']:.2f} DD={m['max_dd_pct']:.1f}% "
                  f"Trades={m['num_trades']}")

    eval_cb = EvalCallback(model, val_env, fc, mean, std, freq=20000)

    os.makedirs("experiment_7/checkpoints", exist_ok=True)
    t0 = time.time()

    chunk_size = 20000
    for chunk_start in range(0, finetune_steps, chunk_size):
        steps_this_chunk = min(chunk_size, finetune_steps - chunk_start)
        model.learn(total_timesteps=steps_this_chunk, reset_num_timesteps=False)
        eval_cb.evaluate(chunk_start + steps_this_chunk)

    train_time = time.time() - t0
    print(f"\n[Finetune] Complete in {train_time:.0f}s ({train_time/60:.1f} min)")

    print("\n[Final Evaluation] Testing on unseen data...")
    test_eq, test_trades = evaluate_full(model, test_df, fc, mean, std, max_steps=10000)
    val_eq, val_trades = evaluate_full(model, val_df, fc, mean, std, max_steps=5000)

    test_m = compute_metrics(test_eq, test_trades, 100000.0)
    val_m = compute_metrics(val_eq, val_trades, 100000.0)

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS (2000 stocks)")
    print(f"{'='*70}")
    print(f"  Train time: {train_time:.0f}s | Device: {device} | Symbols: {max_symbols}")
    print(f"  Val Return:  {val_m['return_pct']:+.2f}% | Sharpe: {val_m['sharpe']:.2f} | DD: {val_m['max_dd_pct']:.2f}% | Trades: {val_m['num_trades']}")
    print(f"  Test Return: {test_m['return_pct']:+.2f}% | Sharpe: {test_m['sharpe']:.2f} | DD: {test_m['max_dd_pct']:.2f}% | Trades: {test_m['num_trades']}")

    os.makedirs("experiment_7/results", exist_ok=True)
    results = {
        "timestamp": datetime.now().isoformat(),
        "train_time_seconds": train_time, "device": device,
        "num_symbols": max_symbols,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "pretrain_steps": pretrain_steps,
        "finetune_steps": finetune_steps,
        "best_params": best_params,
        "pretrain_history": pretrain_hist,
        "training_progress": eval_cb.history,
    }
    with open("experiment_7/results/final_results_2000.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    model.save("experiment_7/models/exp7_2000_final")
    print("\nModel saved: experiment_7/models/exp7_2000_final.zip")

    plot_2000_results(pretrain_hist, eval_cb.history, test_eq, val_eq, test_trades, val_trades)
    return model, results


def plot_2000_results(pretrain_hist, eval_hist, test_eq, val_eq, test_trades, val_trades, init=100000.0):
    fig, axes = plt.subplots(3, 3, figsize=(22, 16))

    ax = axes[0, 0]
    if pretrain_hist:
        epochs = [h["epoch"] for h in pretrain_hist]
        ax2 = ax.twinx()
        ax.plot(epochs, [h["loss"] for h in pretrain_hist], "b-o", ms=4, label="Loss")
        ax2.plot(epochs, [h["accuracy"] * 100 for h in pretrain_hist], "g-s", ms=4, label="Acc %")
        ax.set_title("Pretraining: Loss & Accuracy"); ax.set_xlabel("Epoch")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if eval_hist:
        steps = [h["step"] for h in eval_hist]
        rets = [h["return_pct"] for h in eval_hist]
        ax.plot(steps, rets, "b-", lw=2)
        ax.axhline(y=0, color="r", ls="--", alpha=0.5)
        ax.set_title("Validation Return vs Steps"); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    if eval_hist:
        ax.plot(steps, [h["sharpe"] for h in eval_hist], "g-", lw=2)
        ax.axhline(y=0, color="r", ls="--", alpha=0.5)
        ax.set_title("Validation Sharpe vs Steps"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(val_eq, label=f"Val ({len(val_trades)} tr)", alpha=0.8, lw=1.5, color="orange")
    ax.plot(test_eq, label=f"Test ({len(test_trades)} tr)", alpha=0.8, lw=1.5, color="blue")
    ax.axhline(y=init, color="gray", ls="--", alpha=0.5)
    ax.set_title("Equity Curves"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    peak = np.maximum.accumulate(test_eq)
    dd = (peak - test_eq) / peak * 100
    ax.fill_between(range(len(dd)), 0, -dd, color="red", alpha=0.3)
    ax.plot(-dd, "r-", lw=1)
    ax.set_title(f"Test Drawdown (Max: {np.max(dd):.2f}%)"); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    if test_trades:
        pnls = [t.get("pnl_usd", 0) / init * 100 for t in test_trades]
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, edgecolor="black", lw=0.3)
        ax.axhline(y=0, color="black", lw=0.5)
        wins = sum(1 for p in pnls if p > 0)
        ax.set_title(f"Test Trades ({len(pnls)}, WR: {wins/len(pnls)*100:.1f}%)"); ax.grid(alpha=0.3)

    axes[2, 0].set_title("Experiment 7"); axes[2, 0].text(0.5, 0.5, "Inside Bar Trend-Following", ha="center", fontsize=14)
    axes[2, 1].set_title("See full report"); axes[2, 2].set_title("experiment_7/results/")

    plt.suptitle("Experiment 7: 2000-Stock Inside Bar Trend-Following RL Agent", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("experiment_7/results/comprehensive_2000.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plot] Saved: experiment_7/results/comprehensive_2000.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 7 - 2000 Stock Training")
    parser.add_argument("--optuna", action="store_true")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--pretrain", type=int, default=10000)
    parser.add_argument("--finetune", type=int, default=200000)
    parser.add_argument("--symbols", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download", action="store_true", help="Download data first")
    args = parser.parse_args()

    if args.download:
        from experiment_7.download_2000 import download_sp500, read_tickers, download_stocks_bulk
        os.makedirs("experiment_7/data", exist_ok=True)
        download_sp500("experiment_7/data/SP500_daily.csv")
        tickers = read_tickers(max_tickers=args.symbols)
        download_stocks_bulk(tickers, "experiment_7/data/stocks_2000_daily.parquet", max_workers=12)

    train_2000_pipeline(
        run_optuna=args.optuna,
        n_optuna_trials=args.trials,
        pretrain_steps=args.pretrain,
        finetune_steps=args.finetune,
        max_symbols=args.symbols,
        device=args.device,
    )
