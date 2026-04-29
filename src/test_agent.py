import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.indicators import load_and_preprocess_data
from src.trading_env import ForexTradingEnv


def run_full_evaluation(model, vec_env, deterministic=True):
    """Run a complete evaluation episode collecting all trade information."""
    obs = vec_env.reset()
    equity_curve = []
    closed_trades = []
    positions = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = vec_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
            info = infos[0]
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
            info = infos[0] if isinstance(infos, (list, tuple)) else infos

        eq = info.get("equity_usd", vec_env.get_attr("equity_usd")[0])
        pos = info.get("position", vec_env.get_attr("position")[0])
        equity_curve.append(eq)
        positions.append(pos)

        trade_info = vec_env.get_attr("last_trade_info")[0]
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)

        if done:
            break

    return equity_curve, positions, closed_trades


def compute_trade_statistics(closed_trades, initial_equity=10000.0):
    """Compute comprehensive trade statistics from closed trades."""
    if not closed_trades:
        return {
            "num_trades": 0,
            "win_rate": 0,
            "avg_win_pips": 0,
            "avg_loss_pips": 0,
            "profit_factor": 0,
            "total_net_pips": 0,
        }

    trades_df = pd.DataFrame(closed_trades)
    net_pips = trades_df["net_pips"].values
    num_trades = len(net_pips)
    win_trades = np.sum(net_pips > 0)
    loss_trades = np.sum(net_pips < 0)

    win_rate = win_trades / num_trades * 100 if num_trades > 0 else 0

    avg_win = np.mean(net_pips[net_pips > 0]) if win_trades > 0 else 0
    avg_loss = np.mean(net_pips[net_pips < 0]) if loss_trades > 0 else 0

    gross_profit = np.sum(net_pips[net_pips > 0]) if win_trades > 0 else 0
    gross_loss = abs(np.sum(net_pips[net_pips < 0])) if loss_trades > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_net_pips = np.sum(net_pips)

    # Average trade duration
    avg_duration = np.mean(trades_df["time_in_trade"].values) if "time_in_trade" in trades_df.columns else 0

    # Reason distribution
    reason_counts = trades_df["reason"].value_counts().to_dict() if "reason" in trades_df.columns else {}

    return {
        "num_trades": num_trades,
        "win_trades": int(win_trades),
        "loss_trades": int(loss_trades),
        "win_rate": win_rate,
        "avg_win_pips": float(avg_win),
        "avg_loss_pips": float(avg_loss),
        "profit_factor": float(min(profit_factor, 999.99)),
        "total_net_pips": float(total_net_pips),
        "avg_duration_bars": float(avg_duration),
        "trade_reasons": reason_counts,
    }


def test_agent(data_path, model_path="models/forex_trader_best"):
    """Comprehensive evaluation of the trained agent."""
    print("=" * 60)
    print("Reinforcement Learning Forex Trading Agent - Evaluation")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/5] Loading and preprocessing data...")
    df, feature_cols = load_and_preprocess_data(csv_path=data_path)
    print(f"  Data shape: {df.shape}")
    print(f"  Feature columns: {len(feature_cols)}")

    # Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"  Training bars: {len(train_df)}")
    print(f"  Testing bars : {len(test_df)}")

    # --- Compute normalization ---
    print("\n[2/5] Computing normalization...")
    feature_mean = train_df[feature_cols].values.astype(np.float32).mean(axis=0)
    feature_std = train_df[feature_cols].values.astype(np.float32).std(axis=0)

    # --- Build environment ---
    print("\n[3/5] Building evaluation environment...")
    SL_OPTS = [5, 10, 15, 20, 30, 40, 60, 90, 120]
    TP_OPTS = [5, 10, 15, 20, 30, 40, 60, 90, 120]
    WIN = 30

    def make_test_env():
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
            feature_mean=feature_mean,
            feature_std=feature_std,
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0,
        )

    test_env = DummyVecEnv([make_test_env])

    # --- Load model and evaluate ---
    print("\n[4/5] Loading model and evaluating...")
    model_path_full = model_path + ".zip" if not model_path.endswith(".zip") else model_path
    if not os.path.exists(model_path_full):
        print(f"  Model not found at {model_path_full}. Cannot evaluate.")
        return None, None

    model = PPO.load(model_path, env=test_env)
    equity_curve, positions, closed_trades = run_full_evaluation(
        model, test_env, deterministic=True
    )

    # --- Compute statistics ---
    print("\n[5/5] Computing statistics...")
    eq = np.array(equity_curve)
    initial_equity = 10000.0

    # Equity metrics
    returns = np.diff(eq) / eq[:-1]
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

    total_return_pct = ((eq[-1] - initial_equity) / initial_equity) * 100
    sharpe = (
        np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
        if len(returns) > 0 and np.std(returns) > 0
        else 0.0
    )

    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = float(np.max(dd)) * 100 if len(dd) > 0 else 0.0

    # Compute Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252 * 24)
    else:
        sortino = 0.0

    # Trade statistics
    trade_stats = compute_trade_statistics(closed_trades, initial_equity)

    # --- Print results ---
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Out-of-Sample)")
    print("=" * 60)
    print(f"  Initial Equity : ${initial_equity:,.2f}")
    print(f"  Final Equity   : ${eq[-1]:,.2f}")
    print(f"  Total Return   : {total_return_pct:.2f}%")
    print(f"  Sharpe Ratio   : {sharpe:.2f}")
    print(f"  Sortino Ratio  : {sortino:.2f}")
    print(f"  Max Drawdown   : {max_dd:.2f}%")
    print(f"  Total Trades   : {trade_stats['num_trades']}")
    print(f"  Win Rate       : {trade_stats['win_rate']:.1f}%")
    print(f"  Avg Win        : {trade_stats['avg_win_pips']:.2f} pips")
    print(f"  Avg Loss       : {trade_stats['avg_loss_pips']:.2f} pips")
    print(f"  Profit Factor  : {trade_stats['profit_factor']:.2f}")
    print(f"  Total Net Pips : {trade_stats['total_net_pips']:.2f}")

    # --- Save results ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Save trades
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        trades_csv = os.path.join(results_dir, "trade_history.csv")
        trades_df.to_csv(trades_csv, index=False)
        print(f"\nTrade history saved to: {trades_csv}")

    # Save metrics JSON
    evaluation_results = {
        "timestamp": datetime.now().isoformat(),
        "initial_equity": initial_equity,
        "final_equity": float(eq[-1]),
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd,
        "trade_statistics": trade_stats,
    }

    metrics_json = os.path.join(results_dir, "evaluation_results.json")
    with open(metrics_json, "w") as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    print(f"Evaluation results saved to: {metrics_json}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Equity curve
    ax = axes[0, 0]
    ax.plot(equity_curve, color="steelblue", linewidth=1)
    ax.fill_between(range(len(equity_curve)), equity_curve, initial_equity,
                     where=(np.array(equity_curve) >= initial_equity),
                     alpha=0.3, color="green", label="Profit")
    ax.fill_between(range(len(equity_curve)), equity_curve, initial_equity,
                     where=(np.array(equity_curve) < initial_equity),
                     alpha=0.3, color="red", label="Loss")
    ax.axhline(y=initial_equity, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Equity Curve (Out-of-Sample)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[0, 1]
    drawdown_pct = dd * 100
    ax.fill_between(range(len(drawdown_pct)), 0, -drawdown_pct,
                     color="red", alpha=0.3)
    ax.plot(-drawdown_pct, color="red", linewidth=1)
    ax.set_title(f"Drawdown (Max: {max_dd:.2f}%)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)

    # 3. Returns distribution
    ax = axes[1, 0]
    if len(returns) > 0:
        ax.hist(returns * 100, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
        ax.set_title("Returns Distribution")
        ax.set_xlabel("Return (%)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    # 4. Trade PnL scatter
    ax = axes[1, 1]
    if closed_trades:
        trade_pnls = [t["net_pips"] for t in closed_trades]
        colors = ["green" if p > 0 else "red" for p in trade_pnls]
        ax.bar(range(len(trade_pnls)), trade_pnls, color=colors, alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(f"Trade PnL ({len(trade_pnls)} trades)")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Net Pips")
        ax.grid(True, alpha=0.3)

    plt.suptitle("RL Forex Trading Agent - Comprehensive Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plot_path = os.path.join(results_dir, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to: {plot_path}")

    return equity_curve, evaluation_results


if __name__ == "__main__":
    test_agent("data/EURUSD_Hourly.csv", "models/forex_trader_best")
