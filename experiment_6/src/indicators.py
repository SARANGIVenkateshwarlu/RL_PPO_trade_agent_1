"""
Experiment 6: Feature Engineering — Enterprise Grade
BB Squeeze + Breakout + Full Technical Suite
"""
import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators for Experiment 6."""
    df = df.sort_values(["Symbol", "Date"]).copy()
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    eps, n = 1e-8, len(df)

    # --- BB Squeeze ---
    df["sma_20"] = close.rolling(20).mean()
    df["sigma_20"] = close.rolling(20).std()
    df["bb_upper"] = df["sma_20"] + 2 * df["sigma_20"]
    df["bb_lower"] = df["sma_20"] - 2 * df["sigma_20"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["sma_20"] + eps)
    df["sigma_ratio"] = df["sigma_20"] / (df["sma_20"] + eps)
    df["bb_pct_b"] = ((close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + eps)).clip(-0.5, 1.5)

    vol_sma = vol.rolling(20).mean()
    df["vol_ratio"] = vol / (vol_sma + eps)
    df["squeeze_primary"] = (df["sigma_ratio"] < 0.030).astype(int)
    df["squeeze_width"] = (df["bb_width"] < 0.12).astype(int)
    df["squeeze_width_tight"] = (df["bb_width"] < 0.08).astype(int)
    df["squeeze_volume"] = (df["vol_ratio"] < 0.7).astype(int)
    df["squeeze_moderate"] = (df["squeeze_primary"] & df["squeeze_width"] & df["squeeze_volume"]).astype(int)
    df["squeeze_full"] = (df["squeeze_primary"] & df["squeeze_width_tight"] & (df["vol_ratio"] < 0.5)).astype(int)
    df["squeeze_signal"] = df["squeeze_moderate"] + df["squeeze_full"]

    # --- Breakout ---
    df["breakout_up"] = (high > high.shift(1)).astype(int)
    df["breakout_down"] = (low < low.shift(1)).astype(int)

    # --- EMAs ---
    df["ema_9"] = close.ewm(span=9, adjust=False).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["close_vs_ema9"] = (close - df["ema_9"]) / (close + eps) * 100
    df["close_vs_ema20"] = (close - df["ema_20"]) / (close + eps) * 100
    df["close_vs_ema50"] = (close - df["ema_50"]) / (close + eps) * 100
    df["ema_20_50_spread"] = (df["ema_20"] - df["ema_50"]) / (close + eps) * 100
    df["ema_20_50_spread_slope"] = df["ema_20_50_spread"].diff()

    # --- ATR ---
    tr = np.zeros(n)
    for i in range(1, n): tr[i] = max(high.iloc[i]-low.iloc[i], abs(high.iloc[i]-close.iloc[i-1]), abs(low.iloc[i]-close.iloc[i-1]))
    atr = np.full(n, np.nan)
    if n > 14: atr[13] = np.mean(tr[1:14])
    for i in range(14, n): atr[i] = (atr[i-1]*13 + tr[i])/14
    df["atr_14"] = pd.Series(atr, index=df.index)
    df["atr_pct"] = df["atr_14"] / (close + eps) * 100

    # --- RSI ---
    delta = close.diff().values; g, l = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    ag, al = np.full(n, np.nan), np.full(n, np.nan)
    if n > 14: ag[13], al[13] = np.mean(g[1:14]), np.mean(l[1:14])
    for i in range(14, n): ag[i] = (ag[i-1]*13 + g[i])/14; al[i] = (al[i-1]*13 + l[i])/14
    df["rsi_14"] = pd.Series(100 - 100/(1 + ag/np.where(al==0,np.nan,al)), index=df.index)

    # --- MACD ---
    e12, e26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df["macd_line"] = e12 - e26; df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist_norm"] = (df["macd_line"] - df["macd_signal"]) / (close + eps) * 100

    # --- ADX ---
    h, l, c = high.values, low.values, close.values
    tr2, pdm, mdm = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(1, n):
        tr2[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        up, down = h[i]-h[i-1], l[i-1]-l[i]
        pdm[i] = up if up>down and up>0 else 0; mdm[i] = down if down>up and down>0 else 0
    ats, ps, ms = np.full(n,np.nan), np.full(n,np.nan), np.full(n,np.nan)
    if n > 14: ats[13], ps[13], ms[13] = np.mean(tr2[1:14]), np.mean(pdm[1:14]), np.mean(mdm[1:14])
    for i in range(14, n):
        ats[i] = (ats[i-1]*13 + tr2[i])/14; ps[i] = (ps[i-1]*13 + pdm[i])/14; ms[i] = (ms[i-1]*13 + mdm[i])/14
    dx = 100 * np.abs(ps - ms) / (ps + ms + eps)
    adx_arr = np.full(n, np.nan)
    if n > 27: adx_arr[26] = np.mean(dx[14:27])
    for i in range(27, n): adx_arr[i] = (adx_arr[i-1]*13 + dx[i])/14
    df["adx_14"] = pd.Series(adx_arr / 100.0, index=df.index)

    # --- Returns & Volatility ---
    df["ret_1d"] = close.pct_change() * 100
    df["ret_5d"] = close.pct_change(5) * 100
    df["ret_20d"] = close.pct_change(20) * 100
    df["volatility_20d"] = df["ret_1d"].rolling(20).std()

    # --- Volume ---
    df["volume_sma_20"] = vol.rolling(20).mean()
    df["volume_ratio"] = vol / (df["volume_sma_20"] + eps)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

    # --- Pivot Points ---
    hp, lp, cp = high.shift(1), low.shift(1), close.shift(1)
    pp = (hp + lp + cp) / 3
    df["pivot_r1"] = 2*pp - lp; df["pivot_s1"] = 2*pp - hp
    df["pivot_r1_dist"] = (df["pivot_r1"] - close) / (close + eps) * 100
    df["pivot_s1_dist"] = (close - df["pivot_s1"]) / (close + eps) * 100

    # --- SPX regime ---
    if "sp500_close" in df.columns and "spx_regime" not in df.columns:
        # Only compute if sp500_close has valid data
        if df["sp500_close"].notna().sum() > 0:
            df["sp500_ema_50"] = df["sp500_close"].ewm(span=50, adjust=False).mean()
            df["sp500_ema_150"] = df["sp500_close"].ewm(span=150, adjust=False).mean()
            df["spx_regime"] = 0
            df.loc[(df["sp500_close"] > df["sp500_ema_50"]) & (df["sp500_close"] > df["sp500_ema_150"]), "spx_regime"] = 1
            df.loc[(df["sp500_close"] < df["sp500_ema_50"]) & (df["sp500_close"] < df["sp500_ema_150"]), "spx_regime"] = -1
        else:
            df["spx_regime"] = 0

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    base = [
        "Open", "High", "Low", "Close",
        "bb_width", "bb_pct_b", "sigma_ratio",  # BB Squeeze
        "squeeze_moderate", "squeeze_full", "squeeze_signal",  # Squeeze states
        "breakout_up", "breakout_down",  # Breakout
        "close_vs_ema9", "close_vs_ema20", "close_vs_ema50",  # EMA distances
        "ema_20_50_spread", "ema_20_50_spread_slope",  # EMA crossover
        "atr_pct", "rsi_14", "macd_hist_norm", "adx_14",
        "ret_1d", "ret_5d", "ret_20d", "volatility_20d",
        "volume_ratio", "pivot_r1_dist", "pivot_s1_dist",
    ]
    if "spx_regime" in df.columns: base.append("spx_regime")
    return [c for c in base if c in df.columns]
