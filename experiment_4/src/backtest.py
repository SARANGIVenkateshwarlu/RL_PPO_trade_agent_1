"""
Experiment 4: Backtest Framework
Compares RL agent vs Buy & Hold vs Random Breakout baseline.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def buy_and_hold(df, initial_equity=100000.0):
    """Buy & Hold: buy equal weight at first close per symbol, hold to end."""
    syms = sorted(df["Symbol"].unique())
    if not syms:
        return np.array([initial_equity]), {}

    alloc = initial_equity / len(syms)
    total_eq = np.zeros(len(df))
    symbol_equities = {}

    for sym in syms:
        sdf = df[df["Symbol"] == sym].sort_values("Date").reset_index(drop=True)
        if len(sdf) < 2:
            continue
        close = sdf["Close"].values
        shares = alloc / close[0]
        eq = shares * close
        symbol_equities[sym] = eq

    # Align by index: sum equities at each row
    # Simplification: average equity across symbols at each time step
    if symbol_equities:
        min_len = min(len(v) for v in symbol_equities.values())
        total_equity = np.zeros(min_len)
        for v in symbol_equities.values():
            total_equity += v[:min_len]
        avg_equity = total_equity
    else:
        avg_equity = np.array([initial_equity])

    return avg_equity, {
        "final": float(avg_equity[-1]),
        "return_pct": (avg_equity[-1] / initial_equity - 1) * 100,
        "max_dd": _max_drawdown(avg_equity),
        "trades": len(syms),
    }


def random_breakout(df, initial_equity=100000.0, cash_frac=0.20, risk=0.01, reward=0.03):
    """Random breakout: random BUY/SELL only on breakout days, per-symbol."""
    syms = sorted(df["Symbol"].unique())
    if not syms:
        return np.array([initial_equity]), {}

    alloc = initial_equity / len(syms)
    all_equities = []

    for sym in syms:
        sdf = df[df["Symbol"] == sym].sort_values("Date").reset_index(drop=True)
        n = len(sdf)
        if n < 50:
            continue

        equity = np.full(n, alloc)
        position = 0
        entry = 0.0
        shares = 0
        cash = alloc

        for i in range(1, n):
            bo_up = sdf["High"].iloc[i] > sdf["High"].iloc[i - 1]
            bo_dn = sdf["Low"].iloc[i] < sdf["Low"].iloc[i - 1]
            price = sdf["Close"].iloc[i]
            atr = sdf["atr_14"].iloc[i] if "atr_14" in sdf.columns else price * 0.02

            if position != 0:
                h = sdf["High"].iloc[i]
                l = sdf["Low"].iloc[i]
                if position == 1:
                    sl = entry - atr
                    tp = entry + atr * (reward / risk)
                    if l <= sl or h >= tp:
                        exit_p = sl if l <= sl else tp
                        if l <= sl and h >= tp:
                            exit_p = sl
                        cash += shares * exit_p * 0.999
                        position = 0

            if position == 0 and (bo_up or bo_dn):
                if np.random.random() < 0.3:
                    direction = 1 if bo_up else -1
                    max_cost = cash * cash_frac
                    stop = atr
                    s = min(int((cash * risk) / stop), int(max_cost / price))
                    if s > 0:
                        position = direction
                        entry = price
                        shares = s
                        cash -= shares * price * 1.001

            mtm = shares * (price - entry) * position if position != 0 else 0
            equity[i] = cash + mtm

        all_equities.append(equity)

    if all_equities:
        min_len = min(len(e) for e in all_equities)
        avg_eq = np.mean([e[:min_len] for e in all_equities], axis=0)
    else:
        avg_eq = np.array([initial_equity])

    return avg_eq, {
        "final": float(avg_eq[-1]),
        "return_pct": (avg_eq[-1] / initial_equity - 1) * 100,
        "max_dd": _max_drawdown(avg_eq),
        "trades": 0,
    }


def evaluate_model(model, env):
    """Run one episode with the model and return equity + trades."""
    obs = env.reset()
    eq_curve = []
    trades = []

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
        eq_curve.append(info.get("equity", 0))
        if done:
            break

    # Collect trades from env
    env_trades = env.get_attr("trade_history")[0] if hasattr(env, "get_attr") else []
    return np.array(eq_curve), env_trades


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(np.max(dd) * 100)


def compute_metrics(equity: np.ndarray, trades: list, initial: float = 100000.0) -> dict:
    """Compute comprehensive performance metrics."""
    eq = np.array(equity)
    final = eq[-1]
    ret = (final / initial - 1) * 100

    daily_r = np.diff(eq) / eq[:-1]
    daily_r = daily_r[~np.isnan(daily_r) & ~np.isinf(daily_r)]

    sharpe = np.mean(daily_r) / np.std(daily_r) * np.sqrt(252) if len(daily_r) > 0 and np.std(daily_r) > 0 else 0.0

    downside = daily_r[daily_r < 0]
    sortino = np.mean(daily_r) / np.std(downside) * np.sqrt(252) if len(downside) > 0 and np.std(downside) > 0 else 0.0

    max_dd = _max_drawdown(eq)

    n_trades = len(trades)
    if n_trades > 0:
        pnls = [t.get("pnl_pct", t.get("pnl", 0)) for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n_trades * 100
        avg_win = np.mean([p for p in pnls if p > 0]) if wins > 0 else 0
        avg_loss = np.mean([p for p in pnls if p < 0]) if (n_trades - wins) > 0 else 0

        gross_p = sum(p for p in pnls if p > 0)
        gross_l = abs(sum(p for p in pnls if p < 0))
        pf = gross_p / gross_l if gross_l > 0 else float("inf")
    else:
        pnls = []; wins = 0; wr = 0; avg_win = 0; avg_loss = 0; pf = 0

    return {
        "final_equity": float(final),
        "total_return_pct": float(ret),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown_pct": float(max_dd),
        "num_trades": n_trades,
        "win_rate": float(wr),
        "avg_win_pct": float(avg_win),
        "avg_loss_pct": float(avg_loss),
        "profit_factor": float(min(pf, 999)),
        "total_net_pct": float(sum(pnls)),
    }


def run_backtest_comparison(model, test_env, test_df, init_equity=100000.0):
    """Run full backtest comparison: RL vs B&H vs Random Breakout."""
    print("\n" + "=" * 70)
    print("BACKTEST COMPARISON: RL Agent vs Baselines")
    print("=" * 70)

    # RL Agent
    rl_eq, rl_trades = evaluate_model(model, test_env)
    rl_metrics = compute_metrics(rl_eq, rl_trades, init_equity)

    # Buy & Hold
    bh_eq, bh_metrics = buy_and_hold(test_df, init_equity)

    # Random Breakout
    rb_eq, rb_metrics = random_breakout(test_df, init_equity)

    # Results table
    print(f"\n{'Metric':<22} {'RL Agent':>14} {'Buy&Hold':>14} {'Random BO':>14}")
    print("-" * 64)
    for label, key, fmt in [
        ("Final Equity ($)", "final_equity", ",.0f"),
        ("Total Return (%)", "total_return_pct", ".2f"),
        ("Sharpe Ratio", "sharpe_ratio", ".2f"),
        ("Sortino Ratio", "sortino_ratio", ".2f"),
        ("Max Drawdown (%)", "max_drawdown_pct", ".2f"),
        ("Number of Trades", "num_trades", ".0f"),
        ("Win Rate (%)", "win_rate", ".1f"),
    ]:
        rl_v = rl_metrics.get(key, 0)
        bh_v = bh_metrics.get(key.replace("num_trades", "trades"), 0) if key != "num_trades" else bh_metrics["trades"]
        rb_v = rb_metrics.get(key.replace("num_trades", "trades"), 0) if key != "num_trades" else rb_metrics["trades"]
        rl_s = f"${rl_v:{fmt}}" if "equity" in key else f"{rl_v:{fmt}}"
        bh_s = f"${bh_v:{fmt}}" if "equity" in key else f"{bh_v:{fmt}}"
        rb_s = f"${rb_v:{fmt}}" if "equity" in key else f"{rb_v:{fmt}}"
        print(f"{label:<22} {rl_s:>14} {bh_s:>14} {rb_s:>14}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(rl_eq, label=f"RL Agent ({rl_metrics['total_return_pct']:+.1f}%)", lw=1.5, color="blue")
    ax1.plot(bh_eq, label=f"Buy&Hold ({bh_metrics['return_pct']:+.1f}%)", lw=1, color="green", alpha=0.7)
    ax1.plot(rb_eq, label=f"Random BO ({rb_metrics['return_pct']:+.1f}%)", lw=1, color="gray", alpha=0.5)
    ax1.axhline(y=init_equity, color="black", ls="--", alpha=0.3)
    ax1.set_title("Equity Curves: RL vs Baselines")
    ax1.set_xlabel("Days"); ax1.set_ylabel("Equity ($)")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Drawdown
    for label, eq in [("RL", rl_eq), ("B&H", bh_eq), ("Rand", rb_eq)]:
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak * 100
        alpha = 0.8 if label == "RL" else 0.4
        lw = 1.5 if label == "RL" else 0.8
        ax2.plot(-dd, label=f"{label} (Max: {_max_drawdown(eq):.1f}%)", lw=lw, alpha=alpha)
    ax2.set_title("Drawdown Comparison")
    ax2.set_xlabel("Days"); ax2.set_ylabel("Drawdown (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Experiment 4: Breakout-Constrained RL Trading Agent", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig("experiment_4/results/backtest_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nPlot saved to experiment_4/results/backtest_comparison.png")

    return {
        "rl": rl_metrics,
        "buy_hold": bh_metrics,
        "random_breakout": rb_metrics,
        "rl_equity": rl_eq.tolist(),
        "bh_equity": bh_eq.tolist(),
        "rb_equity": rb_eq.tolist(),
    }
