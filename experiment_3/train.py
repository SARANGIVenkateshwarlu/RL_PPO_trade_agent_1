"""
Experiment 3: Market Regime Filtered RL Stock Trading Agent
S&P 500 regime → BULL=BUY only, BEAR=SELL only, NEUTRAL=HOLD
Strategy: 20/50 EMA crossover + BB Squeeze, 1:3 Risk/Reward
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from experiment_3.src.indicators import compute_indicators, get_feature_columns
from experiment_3.src.market_regime import compute_sp500_regime, merge_regime_to_stocks, get_regime_stats
from experiment_3.src.trading_env import RegimeStockEnv


def load_and_prepare():
    """Load SPX + stock data, compute indicators, merge regimes."""
    print("=" * 60)
    print("Experiment 3: Market Regime Filtered Trading Agent")
    print("=" * 60)

    # Load SP500
    sp_path = "experiment_3/data/SP500_daily.csv"
    if not os.path.exists(sp_path):
        sp_path = "data/SP500_daily.csv"
    sp_df = pd.read_csv(sp_path, parse_dates=["Date"])
    sp_df = compute_sp500_regime(sp_df)
    stats = get_regime_stats(sp_df)
    print(f"\n[S&P 500 Regime Distribution]")
    print(f"  BULL:    {stats['bull_pct']:.1f}%")
    print(f"  BEAR:    {stats['bear_pct']:.1f}%")
    print(f"  NEUTRAL: {stats['neutral_pct']:.1f}%")

    # Load stocks
    st_path = "experiment_3/data/stocks_daily.parquet"
    if not os.path.exists(st_path):
        st_path = "data/stocks_daily.parquet"
        if not os.path.exists(st_path):
            st_path = "data/stocks/combined_stocks.parquet"
    st_df = pd.read_parquet(st_path)
    print(f"\n[Stocks] {len(st_df)} rows, {st_df['Symbol'].nunique()} symbols")

    # Merge regime into stocks
    st_df = merge_regime_to_stocks(st_df, sp_df)
    print(f"  Merged SPX regime: {st_df['regime_label'].value_counts().to_dict()}")

    # Compute indicators per symbol
    print("\n[Indicators] Computing 20/50 EMA + BB Squeeze per symbol...")
    results = []
    for sym in sorted(st_df["Symbol"].unique()):
        sym_df = st_df[st_df["Symbol"] == sym].copy()
        if len(sym_df) < 200:
            continue
        sym_df = compute_indicators(sym_df)
        sym_df["Symbol"] = sym
        results.append(sym_df)

    combined = pd.concat(results, ignore_index=True).sort_values(["Symbol", "Date"]).reset_index(drop=True)
    feature_cols = get_feature_columns(combined)
    print(f"  Processed: {len(combined)} rows, {len(feature_cols)} features")
    print(f"  Features: {feature_cols}")

    return combined, feature_cols


def split_data(df, train_ratio=0.7):
    train, test = [], []
    for sym in sorted(df["Symbol"].unique()):
        sdf = df[df["Symbol"] == sym].sort_values("Date")
        n = int(len(sdf) * train_ratio)
        train.append(sdf.iloc[:n])
        test.append(sdf.iloc[n:])
    return pd.concat(train).reset_index(drop=True), pd.concat(test).reset_index(drop=True)


def evaluate(model, eval_df, feature_cols, mean, std, SL_OPTS):
    env = DummyVecEnv([lambda: RegimeStockEnv(
        df=eval_df, window_size=30, sl_options_pct=SL_OPTS,
        feature_columns=feature_cols,
        feature_mean=mean, feature_std=std,
        random_start=False, episode_max_steps=None,
        hold_reward_weight=0.0, trade_penalty_pct=0.0, time_penalty_pct=0.0,
    )])
    obs = env.reset()
    eq = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(action)
        if len(step_out) == 4:
            obs, _, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, _, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq.append(info.get("equity", env.get_attr("equity")[0]))
        if done:
            break
    return np.array(eq), env.get_attr("trade_history")[0]


def main():
    df, feature_cols = load_and_prepare()

    # Split
    train_df, test_df = split_data(df, 0.7)
    print(f"\n[Split] Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    # Normalization
    mean = train_df[feature_cols].values.astype(np.float32).mean(axis=0)
    std = train_df[feature_cols].values.astype(np.float32).std(axis=0)
    std[std == 0] = 1.0

    # SL options → TP = 3x SL (1:3 RR)
    SL_OPTS = [2, 3, 5, 7, 10]
    print(f"\n[Config] SL: {SL_OPTS}% | TP: {[s*3 for s in SL_OPTS]}% (1:3 RR)")

    # Env
    env_fn = lambda: RegimeStockEnv(
        df=train_df, window_size=30, sl_options_pct=SL_OPTS,
        feature_columns=feature_cols,
        feature_mean=mean, feature_std=std,
        random_start=True, min_episode_steps=200, episode_max_steps=1500,
        hold_reward_weight=0.02, trade_penalty_pct=0.1, time_penalty_pct=0.005,
        max_drawdown_pct=0.25, drawdown_penalty_weight=2.0,
    )
    train_env = DummyVecEnv([env_fn])

    # Model
    model = PPO(
        "MlpPolicy", train_env, verbose=1,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
    )

    os.makedirs("experiment_3/checkpoints", exist_ok=True)
    ckpt = CheckpointCallback(save_freq=50_000, save_path="experiment_3/checkpoints/", name_prefix="exp3")

    TOTAL = 200_000
    print(f"\n[Training] {TOTAL:,} timesteps...")
    model.learn(total_timesteps=TOTAL, callback=ckpt)
    print("Training complete!")

    os.makedirs("experiment_3/models", exist_ok=True)
    model.save("experiment_3/models/exp3_best")

    # Evaluate
    print("\n[Evaluating]...")
    train_eq, train_tr = evaluate(model, train_df, feature_cols, mean, std, SL_OPTS)
    test_eq, test_tr = evaluate(model, test_df, feature_cols, mean, std, SL_OPTS)

    init = 100000.0
    train_ret = (train_eq[-1] / init - 1) * 100
    test_ret = (test_eq[-1] / init - 1) * 100

    peak = np.maximum.accumulate(test_eq)
    max_dd = np.max((peak - test_eq) / peak) * 100

    rets = np.diff(test_eq) / test_eq[:-1]
    rets = rets[~np.isnan(rets) & ~np.isinf(rets)]
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if len(rets) > 0 and np.std(rets) > 0 else 0.0

    # Trade stats
    def trade_stats(trades):
        if not trades:
            return {"n": 0, "wr": 0, "net": 0, "longs": 0, "shorts": 0}
        pnls = [t.get("net_pct", 0) for t in trades]
        dirs = [t.get("direction", 0) for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        return {
            "n": len(trades), "wr": wins / len(trades) * 100,
            "net": sum(pnls), "longs": sum(1 for d in dirs if d == 1),
            "shorts": sum(1 for d in dirs if d == -1),
        }

    tr_s = trade_stats(train_tr)
    te_s = trade_stats(test_tr)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Train: ${train_eq[-1]:,.2f} ({train_ret:+.2f}%) | {tr_s['n']} trades | WR: {tr_s['wr']:.1f}%")
    print(f"  Test:  ${test_eq[-1]:,.2f} ({test_ret:+.2f}%) | {te_s['n']} trades | WR: {te_s['wr']:.1f}%")
    print(f"  Max DD: {max_dd:.2f}% | Sharpe: {sharpe:.2f}")
    print(f"  Test Longs: {te_s['longs']} | Shorts: {te_s['shorts']}")

    # Save
    os.makedirs("experiment_3/results", exist_ok=True)
    results = {
        "timestamp": datetime.now().isoformat(),
        "train": {"final": float(train_eq[-1]), "return_pct": float(train_ret), "trades": tr_s},
        "test": {"final": float(test_eq[-1]), "return_pct": float(test_ret), "trades": te_s,
                 "max_dd": float(max_dd), "sharpe": float(sharpe)},
        "config": {"sl_opts": SL_OPTS, "tp_opts": [s*3 for s in SL_OPTS], "rr": "1:3",
                   "timesteps": TOTAL, "features": feature_cols},
    }
    with open("experiment_3/results/summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(train_eq, label=f"Train ${train_eq[-1]:,.0f}", alpha=0.8)
    ax1.plot(test_eq, label=f"Test ${test_eq[-1]:,.0f}", alpha=0.8)
    ax1.axhline(y=init, color="gray", ls="--", alpha=0.5)
    ax1.set_title("Equity — Market Regime Filtered Agent")
    ax1.legend(); ax1.grid(alpha=0.3)

    dd_pct = (peak - test_eq) / peak * 100
    ax2.fill_between(range(len(dd_pct)), 0, -dd_pct, color="red", alpha=0.3)
    ax2.plot(-dd_pct, "r-", lw=1)
    ax2.set_title(f"Test Drawdown (Max: {max_dd:.2f}%)")
    ax2.grid(alpha=0.3)

    plt.suptitle("Experiment 3: 20/50 EMA + BB Squeeze, 1:3 RR, Regime Filtered", fontweight="bold")
    plt.tight_layout()
    plt.savefig("experiment_3/results/equity_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plots saved.")

    # Save trades
    if test_tr:
        pd.DataFrame(test_tr).to_csv("experiment_3/results/test_trades.csv", index=False)

    return model, results


if __name__ == "__main__":
    main()
