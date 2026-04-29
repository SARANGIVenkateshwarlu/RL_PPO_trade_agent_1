"""
Experiment 5: Bollinger Band Squeeze Detector

Mathematical Conditions:
  Primary:   sigma_20_t < 0.015 × SMA_20           (1.5% volatility)
  Width:     BB_Width = (BBU - BBL) / BBM < 0.10
  Volume:    Volume_t < SMA_Volume_20 × 0.7        (70% of avg)

Squeeze States:
  Full Squeeze: Width < 0.08 AND Volume < 0.6 × Avg → MAX SETUP
  Moderate:     Width < 0.10 AND Volume < 0.7 × Avg
  None:         Otherwise
"""
import numpy as np
import pandas as pd


def compute_bb_squeeze(df: pd.DataFrame, bb_period: int = 20,
                       vol_threshold: float = 0.030,
                       width_threshold: float = 0.12,
                       width_tight: float = 0.08,
                       vol_ratio_threshold: float = 0.7,
                       vol_ratio_tight: float = 0.5) -> pd.DataFrame:
    """
    Compute Bollinger Band Squeeze features.

    Returns df with added columns:
      - bb_mid, bb_upper, bb_lower: Bollinger Band components
      - bb_width: (Upper - Lower) / Mid
      - sigma_20: rolling 20-period standard deviation
      - sigma_ratio: sigma / SMA (volatility % of price)
      - vol_ratio: Volume / SMA_Volume_20
      - squeeze_signal: 2=Full, 1=Moderate, 0=None
      - squeeze_primary: sigma < 1.5% AND width < 0.10
    """
    df = df.copy()
    close = df["Close"]

    # BB components
    df["bb_mid"] = close.rolling(bb_period).mean()
    df["sigma_20"] = close.rolling(bb_period).std()
    df["bb_upper"] = df["bb_mid"] + 2.0 * df["sigma_20"]
    df["bb_lower"] = df["bb_mid"] - 2.0 * df["sigma_20"]

    eps = 1e-8
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + eps)

    # Primary squeeze condition: σ < 1.5% of price
    df["sigma_ratio"] = df["sigma_20"] / (df["bb_mid"] + eps)
    df["squeeze_primary"] = (df["sigma_ratio"] < vol_threshold).astype(int)

    # Width condition
    df["squeeze_width"] = (df["bb_width"] < width_threshold).astype(int)
    df["squeeze_width_tight"] = (df["bb_width"] < width_tight).astype(int)

    # Volume contraction
    vol_sma = df["Volume"].rolling(bb_period).mean()
    df["vol_ratio"] = df["Volume"] / (vol_sma + eps)
    df["squeeze_volume"] = (df["vol_ratio"] < vol_ratio_threshold).astype(int)
    df["squeeze_volume_tight"] = (df["vol_ratio"] < vol_ratio_tight).astype(int)

    # Combined squeeze signal
    df["squeeze_moderate"] = (
        (df["squeeze_primary"] == 1) &
        (df["squeeze_width"] == 1) &
        (df["squeeze_volume"] == 1)
    ).astype(int)

    df["squeeze_full"] = (
        (df["squeeze_primary"] == 1) &
        (df["squeeze_width_tight"] == 1) &
        (df["squeeze_volume_tight"] == 1)
    ).astype(int)

    # Graded squeeze signal: 2=Full, 1=Moderate, 0=None
    df["squeeze_signal"] = 0
    df.loc[df["squeeze_moderate"] == 1, "squeeze_signal"] = 1
    df.loc[df["squeeze_full"] == 1, "squeeze_signal"] = 2

    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicators for Experiment 5.
    Includes: BB squeeze, breakout flags, 9 EMA, RSI, ATR, pivot points.
    """
    df = df.sort_values(["Symbol", "Date"]).copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    eps = 1e-8
    n = len(df)

    # ---- BB Squeeze (CORE) ----
    df = compute_bb_squeeze(df)

    # ---- 9 EMA (exit condition) ----
    df["ema_9"] = close.ewm(span=9, adjust=False).mean()
    df["close_vs_ema9"] = (close - df["ema_9"]) / (close + eps) * 100

    # ---- 20 SMA (for BB and trend) ----
    df["sma_20"] = close.rolling(20).mean()

    # ---- Breakout flags (from Exp 4) ----
    df["breakout_up"] = (high > high.shift(1)).astype(int)
    df["breakout_down"] = (low < low.shift(1)).astype(int)
    df["breakout_flag"] = 0
    df.loc[df["breakout_up"] == 1, "breakout_flag"] = 1
    df.loc[df["breakout_down"] == 1, "breakout_flag"] = -1

    # ---- ATR ----
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high.iloc[i] - low.iloc[i],
                    abs(high.iloc[i] - close.iloc[i - 1]),
                    abs(low.iloc[i] - close.iloc[i - 1]))
    atr_arr = np.full(n, np.nan)
    alen = 14
    if n > alen:
        atr_arr[alen - 1] = np.mean(tr[1:alen])
    for i in range(alen, n):
        atr_arr[i] = (atr_arr[i - 1] * (alen - 1) + tr[i]) / alen
    df["atr_14"] = pd.Series(atr_arr, index=df.index)

    # ---- RSI ----
    delta = close.diff().values
    gain, loss = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    avg_g, avg_l = np.full(n, np.nan), np.full(n, np.nan)
    rlen = 14
    if n > rlen:
        avg_g[rlen - 1] = np.mean(gain[1:rlen])
        avg_l[rlen - 1] = np.mean(loss[1:rlen])
    for i in range(rlen, n):
        avg_g[i] = (avg_g[i - 1] * (rlen - 1) + gain[i]) / rlen
        avg_l[i] = (avg_l[i - 1] * (rlen - 1) + loss[i]) / rlen
    rs = avg_g / np.where(avg_l == 0, np.nan, avg_l)
    df["rsi_14"] = pd.Series(100.0 - 100.0 / (1.0 + rs), index=df.index)

    # ---- Pivot Points ----
    h_prev, l_prev, c_prev = high.shift(1), low.shift(1), close.shift(1)
    pp = (h_prev + l_prev + c_prev) / 3.0
    df["pivot_r1"] = 2.0 * pp - l_prev
    df["pivot_s1"] = 2.0 * pp - h_prev
    df["pivot_r1_dist"] = (df["pivot_r1"] - close) / (close + eps) * 100
    df["pivot_s1_dist"] = (close - df["pivot_s1"]) / (close + eps) * 100

    # ---- MACD ----
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist_norm"] = (df["macd_line"] - df["macd_signal"]) / (close + eps) * 100

    # ---- Returns & Volatility ----
    df["ret_1d"] = close.pct_change() * 100
    df["ret_5d"] = close.pct_change(5) * 100
    df["volatility_20d"] = df["ret_1d"].rolling(20).std()

    # ---- Volume ----
    df["vol_sma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / (df["vol_sma_20"] + eps)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

    # ---- ADX ----
    h, l, c = high.values, low.values, close.values
    tr_arr2 = np.zeros(n); pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        tr_arr2[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up, down = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm[i] = up if (up > down and up > 0) else 0
        mdm[i] = down if (down > up and down > 0) else 0
    atr_s = np.full(n, np.nan); p_s = np.full(n, np.nan); m_s = np.full(n, np.nan)
    alen2 = 14
    if n > alen2:
        atr_s[alen2 - 1] = np.mean(tr_arr2[1:alen2])
        p_s[alen2 - 1] = np.mean(pdm[1:alen2])
        m_s[alen2 - 1] = np.mean(mdm[1:alen2])
    for i in range(alen2, n):
        atr_s[i] = (atr_s[i - 1] * (alen2 - 1) + tr_arr2[i]) / alen2
        p_s[i] = (p_s[i - 1] * (alen2 - 1) + pdm[i]) / alen2
        m_s[i] = (m_s[i - 1] * (alen2 - 1) + mdm[i]) / alen2
    pdi, mdi = 100 * p_s / atr_s, 100 * m_s / atr_s
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi + eps)
    adx_arr = np.full(n, np.nan)
    if n > alen2 * 2 - 1:
        adx_arr[alen2 * 2 - 2] = np.mean(dx[alen2:alen2 * 2 - 1])
    for i in range(alen2 * 2 - 1, n):
        adx_arr[i] = (adx_arr[i - 1] * (alen2 - 1) + dx[i]) / alen2
    df["adx_14"] = pd.Series(adx_arr / 100.0, index=df.index)

    # ---- EMA 20/50 spread (from previous experiments) ----
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["ema_20_50_spread"] = (df["ema_20"] - df["ema_50"]) / (close + eps) * 100

    # ---- SPX regime if available ----
    if "spx_regime" not in df.columns and "sp500_close" in df.columns:
        df["sp500_ema_50"] = df["sp500_close"].ewm(span=50, adjust=False).mean()
        df["sp500_ema_150"] = df["sp500_close"].ewm(span=150, adjust=False).mean()
        df["spx_regime"] = 0
        df.loc[(df["sp500_close"] > df["sp500_ema_50"]) & (df["sp500_close"] > df["sp500_ema_150"]), "spx_regime"] = 1
        df.loc[(df["sp500_close"] < df["sp500_ema_50"]) & (df["sp500_close"] < df["sp500_ema_150"]), "spx_regime"] = -1

    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature columns for the RL agent."""
    cols = [
        "Open", "High", "Low", "Close",
        # BB Squeeze features
        "bb_width", "sigma_ratio", "vol_ratio",
        "squeeze_primary", "squeeze_moderate", "squeeze_full", "squeeze_signal",
        # Breakout
        "breakout_flag",
        # EMA
        "close_vs_ema9",
        # Core technicals
        "rsi_14", "atr_14",
        "pivot_r1_dist", "pivot_s1_dist",
        "macd_hist_norm", "adx_14",
        "ema_20_50_spread",
        "ret_1d", "ret_5d", "volatility_20d",
        "volume_ratio",
    ]
    if "spx_regime" in df.columns:
        cols.append("spx_regime")
    return [c for c in cols if c in df.columns]
