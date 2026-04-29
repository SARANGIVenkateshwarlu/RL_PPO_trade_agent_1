"""
Experiment 5: Backtest Framework
Compares RL Squeeze-Breakout Agent vs Buy & Hold vs Random Breakout baseline.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def buy_and_hold(df, initial_equity=100000.0):
    syms = sorted(df["Symbol"].unique())
    if not syms:
        return np.array([initial_equity]), {}
    alloc = initial_equity / len(syms)
    eqs = {}
    for sym in syms:
        sdf = df[df["Symbol"] == sym].sort_values("Date").reset_index(drop=True)
        if len(sdf) < 2: continue
        close = sdf["Close"].values
        eqs[sym] = (alloc / close[0]) * close
    if eqs:
        ml = min(len(v) for v in eqs.values())
        total = sum(v[:ml] for v in eqs.values())
    else:
        total = np.array([initial_equity])
    return total, {
        "final": float(total[-1]), "return_pct": (total[-1]/initial_equity-1)*100,
        "max_dd": _max_dd(total), "trades": len(syms),
    }


def random_breakout_squeeze(df, initial_equity=100000.0):
    syms = sorted(df["Symbol"].unique())
    if not syms: return np.array([initial_equity]), {}
    alloc = initial_equity / len(syms)
    all_eqs = []
    for sym in syms:
        sdf = df[df["Symbol"] == sym].sort_values("Date").reset_index(drop=True)
        n = len(sdf); pos = 0; entry = 0.0; sh = 0; cash = alloc; eq = np.full(n, alloc)
        for i in range(1, n):
            bo_up = sdf["High"].iloc[i] > sdf["High"].iloc[i-1]
            bo_dn = sdf["Low"].iloc[i] < sdf["Low"].iloc[i-1]
            sqz = sdf.get("squeeze_signal", pd.Series(np.zeros(n))).iloc[i] >= 1
            close = sdf["Close"].iloc[i]
            ema9 = sdf["ema_9"].iloc[i] if "ema_9" in sdf.columns else close
            lo = sdf["Low"].iloc[i]; hi = sdf["High"].iloc[i]

            if pos == 1 and (lo <= entry_candle_lo if pos == 1 else False):
                cash += sh * close * 0.999; pos = 0
            if pos == -1 and (hi >= entry_candle_hi if pos == -1 else False):
                cash += sh * close * 0.999; pos = 0
            if pos == 1 and close < ema9:
                cash += sh * close * 0.999; pos = 0
            if pos == -1 and close > ema9:
                cash += sh * close * 0.999; pos = 0

            if pos == 0 and (bo_up or bo_dn) and sqz and np.random.random() < 0.2:
                direction = 1 if bo_up else -1
                max_cost = cash * 0.20
                stop = abs(close - (lo if direction == 1 else hi)) + 0.01
                s = int(min((cash*0.01)/stop, max_cost/close))
                if s > 0:
                    pos = direction; entry = close; sh = s
                    cash -= sh * close * 1.001
                    if direction == 1: entry_candle_lo = lo
                    else: entry_candle_hi = hi

            mtm = sh * (close - entry) * pos if pos != 0 else 0
            eq[i] = cash + mtm
        all_eqs.append(eq)
    ml = min(len(e) for e in all_eqs) if all_eqs else 1
    avg = np.mean([e[:ml] for e in all_eqs], axis=0) if all_eqs else np.array([initial_equity])
    return avg, {"final": float(avg[-1]), "return_pct": (avg[-1]/initial_equity-1)*100, "max_dd": _max_dd(avg), "trades": 0}


def evaluate_model(model, env):
    obs = env.reset(); eq = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(action)
        if len(step_out) == 4:
            obs, _, dones, infos = step_out; done = bool(dones[0])
        else:
            obs, _, terminated, truncated, infos = step_out; done = bool(terminated[0] or truncated[0])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq.append(info.get("equity", 0))
        if done: break
    trades = env.get_attr("trade_history")[0] if hasattr(env, "get_attr") else []
    return np.array(eq), trades


def _max_dd(equity): peak = np.maximum.accumulate(equity); return float(np.max((peak - equity) / peak) * 100)


def compute_metrics(equity, trades, initial=100000.0):
    eq = np.array(equity); final = eq[-1]; ret = (final/initial-1)*100
    dr = np.diff(eq)/eq[:-1]; dr = dr[~np.isnan(dr)&~np.isinf(dr)]
    sharpe = np.mean(dr)/np.std(dr)*np.sqrt(252) if len(dr)>0 and np.std(dr)>0 else 0
    ddr = dr[dr<0]; sortino = np.mean(dr)/np.std(ddr)*np.sqrt(252) if len(ddr)>0 and np.std(ddr)>0 else 0
    max_dd = _max_dd(eq)
    n = len(trades)
    if n>0:
        pnls=[t.get("pnl_pct",0) for t in trades]; wins=sum(1 for p in pnls if p>0)
        wr=wins/n*100; avg_w=np.mean([p for p in pnls if p>0]) if wins>0 else 0
        avg_l=np.mean([p for p in pnls if p<0]) if n-wins>0 else 0
        gp=sum(p for p in pnls if p>0); gl=abs(sum(p for p in pnls if p<0))
        pf=gp/gl if gl>0 else 999
    else: pnls=[]; wins=0; wr=0; avg_w=0; avg_l=0; pf=0
    reasons = {}
    for t in trades: reasons[t.get("reason","?")] = reasons.get(t.get("reason","?"), 0) + 1
    return {"final_equity":float(final),"total_return_pct":float(ret),"sharpe":float(sharpe),
            "sortino":float(sortino),"max_dd":float(max_dd),"num_trades":n,"win_rate":float(wr),
            "avg_win_pct":float(avg_w),"avg_loss_pct":float(avg_l),"profit_factor":float(min(pf,999)),
            "exit_reasons":reasons}


def run_backtest(model, test_env, test_df, init=100000.0):
    print("\n" + "="*70)
    print("BACKTEST: RL Squeeze-Breakout vs Baselines")
    print("="*70)

    rl_eq, rl_tr = evaluate_model(model, test_env)
    rl_m = compute_metrics(rl_eq, rl_tr, init)

    bh_eq, bh_m = buy_and_hold(test_df, init)
    rb_eq, rb_m = random_breakout_squeeze(test_df, init)

    print(f"\n{'Metric':<24} {'RL Agent':>13} {'Buy&Hold':>13} {'Rand SqBO':>13}")
    print("-"*63)
    for lbl,key,fmt in [("Final Equity", "final_equity", ",.0f"), ("Return %", "total_return_pct", ".2f"),
                          ("Sharpe", "sharpe", ".2f"), ("Max DD %", "max_dd", ".2f"),
                          ("Trades", "num_trades", ".0f"), ("Win Rate %", "win_rate", ".1f")]:
        rv,bv,rv2=rl_m.get(key,0),bh_m.get(key,0) if key!="num_trades" else bh_m["trades"],rb_m.get(key,0) if key!="num_trades" else rb_m["trades"]
        rs=f"${rv:{fmt}}" if "equity" in key else f"{rv:{fmt}}"
        bs=f"${bv:{fmt}}" if "equity" in key else f"{bv:{fmt}}"
        r2s=f"${rv2:{fmt}}" if "equity" in key else f"{rv2:{fmt}}"
        print(f"{lbl:<24} {rs:>13} {bs:>13} {r2s:>13}")

    if rl_m.get("exit_reasons"):
        print(f"\nExit reasons: {rl_m['exit_reasons']}")

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18,6))
    ax1.plot(rl_eq,label=f"RL ({rl_m['total_return_pct']:+.1f}%)",lw=1.5,color="blue")
    ax1.plot(bh_eq,label=f"B&H ({bh_m['return_pct']:+.1f}%)",lw=1,color="green",alpha=0.7)
    ax1.plot(rb_eq,label=f"Rand ({rb_m['return_pct']:+.1f}%)",lw=1,color="gray",alpha=0.5)
    ax1.axhline(y=init,color="black",ls="--",alpha=0.3)
    ax1.set_title("Equity Curves"); ax1.legend(); ax1.grid(alpha=0.3)
    for lbl,eq in [("RL",rl_eq),("B&H",bh_eq),("Rand",rb_eq)]:
        peak=np.maximum.accumulate(eq); dd=(peak-eq)/peak*100
        alpha=0.8 if lbl=="RL" else 0.4; lw=1.5 if lbl=="RL" else 0.8
        ax2.plot(-dd,label=f"{lbl} (Max: {_max_dd(eq):.1f}%)",lw=lw,alpha=alpha)
    ax2.set_title("Drawdown"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle("Experiment 5: BB Squeeze + Breakout + 9 EMA Exit",fontweight="bold")
    plt.tight_layout(); plt.savefig("experiment_5/results/backtest.png",dpi=150,bbox_inches="tight"); plt.close()
    print("Plot saved to experiment_5/results/backtest.png")
    return {"rl":rl_m,"buy_hold":bh_m,"random":rb_m}
