"""
Experiment 4: Breakout-Constrained Trading Indicators

Computes the mandatory state space features:
  S_t = [OHLC_t, RSI_14_t, SMA_20_t, Pivot_R1/S1_t, Volume_t, Position_t, Breakout_Flag_t]

All features are computed per-symbol on daily timeframe.
"""
import numpy as np
import pandas as pd


def compute_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard Pivot Points based on previous day's H, L, C.

    Pivot (PP) = (H_prev + L_prev + C_prev) / 3
    Resistance 1 (R1) = 2 × PP - L_prev
    Support 1 (S1) = 2 × PP - H_prev
    """
    df = df.copy()
    h_prev = df["High"].shift(1)
    l_prev = df["Low"].shift(1)
    c_prev = df["Close"].shift(1)

    pp = (h_prev + l_prev + c_prev) / 3.0
    df["pivot_pp"] = pp
    df["pivot_r1"] = 2.0 * pp - l_prev
    df["pivot_s1"] = 2.0 * pp - h_prev

    # Normalized distance to R1/S1 as % of price
    eps = 1e-8
    df["pivot_r1_dist"] = (df["pivot_r1"] - df["Close"]) / (df["Close"] + eps) * 100
    df["pivot_s1_dist"] = (df["Close"] - df["pivot_s1"]) / (df["Close"] + eps) * 100

    return df


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = close.diff().values
    n = len(delta)
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)

    avg_g = np.full(n, np.nan)
    avg_l = np.full(n, np.nan)

    if n > length:
        avg_g[length - 1] = np.mean(gain[1:length])
        avg_l[length - 1] = np.mean(loss[1:length])

    for i in range(length, n):
        avg_g[i] = (avg_g[i - 1] * (length - 1) + gain[i]) / length
        avg_l[i] = (avg_l[i - 1] * (length - 1) + loss[i]) / length

    rs = avg_g / np.where(avg_l == 0, np.nan, avg_l)
    return pd.Series(100.0 - 100.0 / (1.0 + rs), index=close.index)


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder-smoothed ATR."""
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(h)

    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    atr_arr = np.full(n, np.nan)
    if n > length:
        atr_arr[length - 1] = np.mean(tr[1:length])
    for i in range(length, n):
        atr_arr[i] = (atr_arr[i - 1] * (length - 1) + tr[i]) / length

    return pd.Series(atr_arr, index=df.index)


def compute_breakout_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute breakout flags based on current vs previous day high/low.

    BREAKOUT_UP   = 1 if High_t > High_{t-1} else 0
    BREAKOUT_DOWN = 1 if Low_t < Low_{t-1} else 0

    These are the MANDATORY filters per the specification.
    """
    df = df.copy()
    df["breakout_up"] = (df["High"] > df["High"].shift(1)).astype(int)
    df["breakout_down"] = (df["Low"] < df["Low"].shift(1)).astype(int)
    df["breakout_any"] = ((df["breakout_up"] == 1) | (df["breakout_down"] == 1)).astype(int)

    # Combined flag as a single feature (-1=down breakout, 0=none, 1=up breakout)
    df["breakout_flag"] = 0
    df.loc[df["breakout_up"] == 1, "breakout_flag"] = 1
    df.loc[df["breakout_down"] == 1, "breakout_flag"] = -1

    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the complete state-space features for Experiment 4.

    S_t = [OHLC_t, RSI_14_t, SMA_20_t, Pivot_R1/S1_t, Volume_t, Breakout_Flag_t]
    Plus derived features for the RL agent.
    """
    df = df.sort_values(["Symbol", "Date"]).copy()
    close = df["Close"]
    eps = 1e-8

    # --- Core indicators ---
    df["rsi_14"] = compute_rsi(df["Close"], 14)

    df["sma_20"] = df["Close"].rolling(20).mean()
    df["close_vs_sma20"] = (close - df["sma_20"]) / (df["sma_20"] + eps) * 100

    # ATR for position sizing
    df["atr_14"] = compute_atr(df, 14)
    df["atr_pct"] = df["atr_14"] / (close + eps) * 100

    # Pivot Points
    df = compute_pivot_points(df)

    # Breakout flags
    df = compute_breakout_flag(df)

    # --- Additional features for richer state ---
    # EMA 20/50 (from experiment_3 constraint)
    df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # EMA spread
    df["ema_20_50_spread"] = (df["ema_20"] - df["ema_50"]) / (close + eps) * 100

    # Bollinger Bands
    bb_std = df["Close"].rolling(20).std()
    df["bb_upper"] = df["sma_20"] + 2 * bb_std
    df["bb_lower"] = df["sma_20"] - 2 * bb_std
    df["bb_pct_b"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + eps)
    df["bb_pct_b"] = df["bb_pct_b"].clip(-0.5, 1.5)

    # BB squeeze
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["sma_20"] + eps)
    df["bb_width_min_125"] = df["bb_width"].rolling(125).min()
    df["bb_width_rank"] = df["bb_width"] / (df["bb_width_min_125"] + eps)
    df["bb_squeeze"] = ((df["bb_width_rank"] < 1.10) & (df["bb_width"] < 0.05)).astype(int)

    # Volume features
    df["volume_sma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_sma_20"] + eps)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

    # Returns
    df["ret_1d"] = df["Close"].pct_change() * 100
    df["ret_5d"] = df["Close"].pct_change(5) * 100
    df["volatility_20d"] = df["ret_1d"].rolling(20).std()

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist_norm"] = (df["macd_line"] - df["macd_signal"]) / (close + eps) * 100

    # ADX
    df["adx_14"] = _compute_adx(df) / 100.0

    # Market regime from SPX (if available)
    if "sp500_close" in df.columns:
        df["sp500_ema_50"] = df["sp500_close"].ewm(span=50, adjust=False).mean()
        df["sp500_ema_150"] = df["sp500_close"].ewm(span=150, adjust=False).mean()
        df["spx_regime"] = 0
        df.loc[(df["sp500_close"] > df["sp500_ema_50"]) & (df["sp500_close"] > df["sp500_ema_150"]), "spx_regime"] = 1
        df.loc[(df["sp500_close"] < df["sp500_ema_50"]) & (df["sp500_close"] < df["sp500_ema_150"]), "spx_regime"] = -1

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    return df


def _compute_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Compute ADX."""
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    n = len(h)
    tr_arr = np.zeros(n)
    pdm = np.zeros(n)
    mdm = np.zeros(n)
    for i in range(1, n):
        tr_arr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        pdm[i] = up if (up > down and up > 0) else 0
        mdm[i] = down if (down > up and down > 0) else 0

    atr_s = np.full(n, np.nan)
    p_s = np.full(n, np.nan)
    m_s = np.full(n, np.nan)
    if n > length:
        atr_s[length - 1] = np.mean(tr_arr[1:length])
        p_s[length - 1] = np.mean(pdm[1:length])
        m_s[length - 1] = np.mean(mdm[1:length])
    for i in range(length, n):
        atr_s[i] = (atr_s[i - 1] * (length - 1) + tr_arr[i]) / length
        p_s[i] = (p_s[i - 1] * (length - 1) + pdm[i]) / length
        m_s[i] = (m_s[i - 1] * (length - 1) + mdm[i]) / length

    eps = 1e-8
    pdi = 100 * p_s / atr_s
    mdi = 100 * m_s / atr_s
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi + eps)
    adx_arr = np.full(n, np.nan)
    if n > length * 2 - 1:
        adx_arr[length * 2 - 2] = np.mean(dx[length:length * 2 - 1])
    for i in range(length * 2 - 1, n):
        adx_arr[i] = (adx_arr[i - 1] * (length - 1) + dx[i]) / length
    return pd.Series(adx_arr, index=df.index)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the mandatory + auxiliary feature columns for RL agent."""
    mandatory = [
        "Open", "High", "Low", "Close",          # OHLC
        "rsi_14",                                  # RSI(14)
        "close_vs_sma20",                          # SMA_20 price position
        "pivot_r1_dist", "pivot_s1_dist",          # Pivot R1/S1 distance
        "volume_ratio",                            # Volume (normalized)
        "breakout_flag",                           # Breakout flag
    ]
    auxiliary = [
        "ema_20_50_spread",
        "bb_pct_b", "bb_width_rank", "bb_squeeze",
        "atr_pct",
        "macd_hist_norm",
        "adx_14",
        "ret_1d", "ret_5d",
        "volatility_20d",
    ]
    if "spx_regime" in df.columns:
        auxiliary.append("spx_regime")

    all_cols = mandatory + auxiliary
    return [c for c in all_cols if c in df.columns]
