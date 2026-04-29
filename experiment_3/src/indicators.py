"""
Stock technical indicators for Experiment 3:
  - 20/50 EMA crossover (primary trend signal)
  - Bollinger Band squeeze (volatility contraction)
  - RSI, ATR, MACD, Volume

All features are scale-invariant for the RL agent.
"""
import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for stock OHLCV data.
    Core: 20/50 EMA crossover + BB Squeeze.
    """
    df = df.sort_values("Date").copy()
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    vol = df["Volume"].values
    n = len(df)
    eps = 1e-8

    # ---- EMA (20 and 50 crossover) ----
    df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # EMA slopes (% daily change)
    df["ema_20_slope"] = df["ema_20"].pct_change() * 100
    df["ema_50_slope"] = df["ema_50"].pct_change() * 100

    # EMA spread (20 vs 50) — the core crossover signal
    df["ema_20_50_spread"] = (df["ema_20"] - df["ema_50"]) / (df["Close"] + eps) * 100
    df["ema_spread_slope"] = df["ema_20_50_spread"].diff()

    # Crossover binary signals
    df["ema_cross_up"] = (
        (df["ema_20"] > df["ema_50"]) &
        (df["ema_20"].shift(1) <= df["ema_50"].shift(1))
    ).astype(int)

    df["ema_cross_down"] = (
        (df["ema_20"] < df["ema_50"]) &
        (df["ema_20"].shift(1) >= df["ema_50"].shift(1))
    ).astype(int)

    # ---- Price relative to EMAs ----
    df["close_vs_ema20"] = (df["Close"] - df["ema_20"]) / (df["Close"] + eps) * 100
    df["close_vs_ema50"] = (df["Close"] - df["ema_50"]) / (df["Close"] + eps) * 100

    # ---- Bollinger Bands (20, 2) ----
    bb_len = 20
    df["bb_mid"] = df["Close"].rolling(bb_len).mean()
    bb_std = df["Close"].rolling(bb_len).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    # BB width & squeeze detection
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + eps)
    df["bb_width_min_125"] = df["bb_width"].rolling(125).min()
    df["bb_width_rank"] = df["bb_width"] / (df["bb_width_min_125"] + eps)
    df["bb_squeeze"] = ((df["bb_width_rank"] < 1.10) & (df["bb_width"] < 0.05)).astype(int)

    # BB %B (position within bands)
    df["bb_pct_b"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + eps)
    df["bb_pct_b"] = df["bb_pct_b"].clip(-0.5, 1.5)

    # ---- ATR (Average True Range) ----
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr_val = np.full(n, np.nan)
    alen = 14
    if n > alen:
        atr_val[alen - 1] = np.mean(tr[1:alen])
    for i in range(alen, n):
        atr_val[i] = (atr_val[i - 1] * (alen - 1) + tr[i]) / alen
    df["atr_14"] = pd.Series(atr_val, index=df.index)
    df["atr_pct"] = df["atr_14"] / (df["Close"] + eps) * 100

    # ---- RSI (14) ----
    delta = df["Close"].diff().values
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    avg_g = np.full(n, np.nan)
    avg_l = np.full(n, np.nan)
    rlen = 14
    if n > rlen:
        avg_g[rlen - 1] = np.mean(gain[1:rlen])
        avg_l[rlen - 1] = np.mean(loss[1:rlen])
    for i in range(rlen, n):
        avg_g[i] = (avg_g[i - 1] * (rlen - 1) + gain[i]) / rlen
        avg_l[i] = (avg_l[i - 1] * (rlen - 1) + loss[i]) / rlen
    rs = avg_g / np.where(avg_l == 0, np.nan, avg_l)
    df["rsi_14"] = pd.Series(100 - 100 / (1 + rs), index=df.index)

    # ---- MACD (12, 26, 9) ----
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    df["macd_hist_norm"] = df["macd_hist"] / (df["Close"] + eps) * 100

    # ---- Returns ----
    df["ret_1d"] = df["Close"].pct_change() * 100
    df["ret_5d"] = df["Close"].pct_change(5) * 100
    df["ret_20d"] = df["Close"].pct_change(20) * 100
    df["volatility_20d"] = df["ret_1d"].rolling(20).std()

    # ---- Volume ----
    df["vol_sma_20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / (df["vol_sma_20"] + eps)
    df["vol_ratio"] = df["vol_ratio"].fillna(1.0)

    # ---- Trend strength (ADX 14) ----
    p_dm = np.zeros(n)
    m_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        p_dm[i] = up if (up > down and up > 0) else 0
        m_dm[i] = down if (down > up and down > 0) else 0

    atr_s = np.full(n, np.nan)
    p_s = np.full(n, np.nan)
    m_s = np.full(n, np.nan)
    alen2 = 14
    if n > alen2:
        atr_s[alen2 - 1] = np.mean(tr[1:alen2])
        p_s[alen2 - 1] = np.mean(p_dm[1:alen2])
        m_s[alen2 - 1] = np.mean(m_dm[1:alen2])
    for i in range(alen2, n):
        atr_s[i] = (atr_s[i - 1] * (alen2 - 1) + tr[i]) / alen2
        p_s[i] = (p_s[i - 1] * (alen2 - 1) + p_dm[i]) / alen2
        m_s[i] = (m_s[i - 1] * (alen2 - 1) + m_dm[i]) / alen2

    pdi = 100 * p_s / atr_s
    mdi = 100 * m_s / atr_s
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi + eps)
    adx_arr = np.full(n, np.nan)
    if n > alen2 * 2 - 1:
        adx_arr[alen2 * 2 - 2] = np.mean(dx[alen2:alen2 * 2 - 1])
    for i in range(alen2 * 2 - 1, n):
        adx_arr[i] = (adx_arr[i - 1] * (alen2 - 1) + dx[i]) / alen2
    df["adx_14"] = pd.Series(adx_arr / 100, index=df.index)

    # ---- SPX relative strength (if available) ----
    if "sp500_close" in df.columns:
        df["spx_ret_1d"] = df["sp500_close"].pct_change() * 100
        df["stock_vs_spx"] = df["ret_1d"] - df["spx_ret_1d"]

    # ---- Combined entry signal (20/50 cross + BB squeeze) ----
    df["combined_signal"] = (
        ((df["ema_cross_up"] == 1) | (df["ema_cross_down"] == 1)) &
        (df["bb_squeeze"] == 1) &
        (df["adx_14"] > 0.15)
    ).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of feature columns for the RL agent."""
    base = [
        "ema_20_slope", "ema_50_slope",
        "ema_20_50_spread", "ema_spread_slope",
        "close_vs_ema20", "close_vs_ema50",
        "bb_pct_b", "bb_width", "bb_width_rank", "bb_squeeze",
        "atr_pct",
        "rsi_14",
        "macd_hist_norm",
        "adx_14",
        "ret_1d", "ret_5d", "ret_20d",
        "volatility_20d",
        "vol_ratio",
        "combined_signal",
    ]
    # Add SPX features if available
    if "spx_ret_1d" in df.columns:
        base.append("spx_ret_1d")
    if "stock_vs_spx" in df.columns:
        base.append("stock_vs_spx")
    # Add regime as a feature so the agent knows the market context
    if "regime" in df.columns:
        base.append("regime")

    return [c for c in base if c in df.columns]
