import pandas as pd
import numpy as np


def rsi(close, length=14):
    """Calculate RSI using Wilder's smoothing method."""
    n = len(close)
    delta = close.diff().values
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    # Initial SMA
    if n > length:
        avg_gain[length - 1] = np.mean(gain[1:length])
        avg_loss[length - 1] = np.mean(loss[1:length])

    # Wilder smoothing
    for i in range(length, n):
        avg_gain[i] = (avg_gain[i - 1] * (length - 1) + gain[i]) / length
        avg_loss[i] = (avg_loss[i - 1] * (length - 1) + loss[i]) / length

    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    return pd.Series(100.0 - 100.0 / (1.0 + rs), index=close.index)


def atr(high, low, close, length=14):
    """Calculate Average True Range (Wilder smoothing)."""
    n = len(close)
    h = high.values
    l = low.values
    c = close.values

    tr_vals = np.zeros(n)
    tr_vals[0] = h[0] - l[0]
    for i in range(1, n):
        tr1 = h[i] - l[i]
        tr2 = abs(h[i] - c[i - 1])
        tr3 = abs(l[i] - c[i - 1])
        tr_vals[i] = max(tr1, tr2, tr3)

    atr_arr = np.full(n, np.nan)
    if n > length:
        atr_arr[length - 1] = np.mean(tr_vals[1:length])

    for i in range(length, n):
        atr_arr[i] = (atr_arr[i - 1] * (length - 1) + tr_vals[i]) / length

    return pd.Series(atr_arr, index=close.index)


def ema(close, length):
    """Calculate Exponential Moving Average."""
    return close.ewm(span=length, adjust=False).mean()


def sma(close, length):
    """Calculate Simple Moving Average."""
    return close.rolling(window=length, min_periods=length).mean()


def macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD, Signal line, and Histogram."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(close, length=20, std=2):
    """Calculate Bollinger Bands."""
    middle = sma(close, length)
    std_dev = close.rolling(window=length, min_periods=length).std()
    upper = middle + std * std_dev
    lower = middle - std * std_dev
    bandwidth = upper - lower
    percent_b = (close - lower) / (upper - lower)
    return upper, middle, lower, bandwidth, percent_b


def stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    k = 100.0 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


def obv(close, volume):
    """Calculate On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def adx(high, low, close, length=14):
    """Calculate Average Directional Index."""
    n = len(close)
    h = high.values
    l = low.values
    c = close.values

    # True Range
    tr_vals = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        tr_vals[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    # Wilder smoothing for TR, +DM, -DM
    atr_smooth = np.full(n, np.nan)
    plus_dm_smooth = np.full(n, np.nan)
    minus_dm_smooth = np.full(n, np.nan)

    if n > length:
        atr_smooth[length - 1] = np.mean(tr_vals[1:length])
        plus_dm_smooth[length - 1] = np.mean(plus_dm[1:length])
        minus_dm_smooth[length - 1] = np.mean(minus_dm[1:length])

    for i in range(length, n):
        atr_smooth[i] = (atr_smooth[i - 1] * (length - 1) + tr_vals[i]) / length
        plus_dm_smooth[i] = (plus_dm_smooth[i - 1] * (length - 1) + plus_dm[i]) / length
        minus_dm_smooth[i] = (minus_dm_smooth[i - 1] * (length - 1) + minus_dm[i]) / length

    plus_di = 100.0 * plus_dm_smooth / atr_smooth
    minus_di = 100.0 * minus_dm_smooth / atr_smooth

    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = np.full(n, np.nan)

    if n > length * 2 - 1:
        adx_val[length * 2 - 2] = np.mean(dx[length:length * 2 - 1])
    for i in range(length * 2 - 1, n):
        adx_val[i] = (adx_val[i - 1] * (length - 1) + dx[i]) / length

    idx = close.index
    return pd.Series(adx_val / 100.0, index=idx), pd.Series(plus_di / 100.0, index=idx), pd.Series(minus_di / 100.0, index=idx)


def load_and_preprocess_data(csv_path: str = None, df: pd.DataFrame = None):
    """
    Loads EURUSD data from CSV or DataFrame and adds RELATIVE technical indicators.

    Features added (all scale-invariant for RL agent):
      - RSI (14)
      - ATR normalized as % of close
      - EMA 9, 21, 50 with slopes and cross signals
      - MACD (12, 26, 9)
      - Bollinger Bands (20, 2)
      - Stochastic (14, 3)
      - Volume features (OBV slope, volume ratio)
      - ADX trend strength
      - Volatility ratio

    The EMA crossover strategy is captured via:
      - EMA spreads (short - long MA, normalized)
      - EMA spread slopes (trend direction changes)
      - Price distance from EMAs

    Returns (df, feature_cols) for the RL agent.
    """
    # ---- Load data ----
    if df is None and csv_path is not None:
        df = pd.read_csv(
            csv_path,
            parse_dates=["Time (EET)"],
            dayfirst=False,
        )
        df.columns = df.columns.str.strip()
        df = df.set_index("Time (EET)")
        df.sort_index(inplace=True)
    elif df is not None:
        df = df.copy()
    else:
        raise ValueError("Either csv_path or df must be provided.")

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"] if "Volume" in df.columns else pd.Series(1.0, index=df.index)

    # ---- Core indicators ----
    df["rsi_14"] = rsi(close, length=14)

    # ATR (raw and % of close)
    df["atr_14"] = atr(high, low, close, length=14)
    df["atr_pct"] = df["atr_14"] / close

    # ---- EMA crossover features (the core strategy) ----
    df["ema_9"] = ema(close, 9)
    df["ema_21"] = ema(close, 21)
    df["ema_50"] = ema(close, 50)

    # EMA slopes (normalized by price for scale invariance)
    eps = 1e-8
    df["ema_9_slope"] = df["ema_9"].diff() / (close + eps)
    df["ema_21_slope"] = df["ema_21"].diff() / (close + eps)
    df["ema_50_slope"] = df["ema_50"].diff() / (close + eps)

    # Price distance from EMAs (normalized by ATR)
    atr_safe = df["atr_14"].replace(0, np.nan)
    df["close_ema9_diff"] = (close - df["ema_9"]) / atr_safe
    df["close_ema21_diff"] = (close - df["ema_21"]) / atr_safe
    df["close_ema50_diff"] = (close - df["ema_50"]) / atr_safe

    # EMA spread signals (core cross-over indicators)
    df["ema_9_21_spread"] = (df["ema_9"] - df["ema_21"]) / atr_safe
    df["ema_21_50_spread"] = (df["ema_21"] - df["ema_50"]) / atr_safe
    df["ema_9_50_spread"] = (df["ema_9"] - df["ema_50"]) / atr_safe

    # Spread slopes (direction of EMA convergence/divergence)
    df["ema_9_21_spread_slope"] = df["ema_9_21_spread"].diff()
    df["ema_21_50_spread_slope"] = df["ema_21_50_spread"].diff()

    # ---- MACD ----
    macd_line, signal_line, histogram = macd(close, 12, 26, 9)
    df["macd_line"] = macd_line / atr_safe
    df["macd_signal"] = signal_line / atr_safe
    df["macd_hist"] = histogram / atr_safe

    # ---- Bollinger Bands ----
    bb_upper, bb_middle, bb_lower, bb_bandwidth, bb_percent_b = bollinger_bands(close, 20, 2)
    df["bb_percent_b"] = bb_percent_b
    df["bb_bandwidth"] = bb_bandwidth / atr_safe

    # ---- Stochastic ----
    stoch_k, stoch_d = stochastic(high, low, close, 14, 3)
    df["stoch_k"] = stoch_k / 100.0
    df["stoch_d"] = stoch_d / 100.0

    # ---- Volatility features ----
    returns = close.pct_change()
    df["volatility_14"] = returns.rolling(14).std()
    df["volatility_ratio"] = df["volatility_14"] / (df["volatility_14"].rolling(50).mean() + eps)

    # ---- Volume features (handle zero-volume forex data) ----
    if (vol == 0).all():
        # Forex data typically has no volume from Yahoo; fill with neutral values
        df["volume_ratio"] = 1.0
        df["obv"] = 0.0
        df["obv_slope"] = 0.0
    else:
        vol_safe = vol.replace(0, np.nan)
        df["volume_sma_14"] = vol.rolling(14).mean()
        df["volume_ratio"] = vol / df["volume_sma_14"].replace(0, np.nan)
        df["obv"] = obv(close, vol)
        df["obv_slope"] = df["obv"].diff() / (vol_safe + eps)

    # ---- Trend strength (ADX) ----
    adx_val, plus_di, minus_di = adx(high, low, close, 14)
    df["adx_14"] = adx_val / 100.0

    # ---- EMA relative crossover signal ----
    df["ema_cross_signal"] = (close - df["ema_21"]) / (close + eps)

    # ---- Drop initial NaN rows from indicators ----
    df.dropna(inplace=True)

    if len(df) < 100:
        raise ValueError(f"Dataset too small after preprocessing: {len(df)} rows. Need at least 100.")

    # ---- Feature columns for the RL agent ----
    # ONLY scale-invariant / relative features (no raw prices)
    feature_cols = [
        "rsi_14",                   # RSI 0-100
        "atr_pct",                  # ATR as % of price
        "ema_9_slope",              # EMA 9 slope normalized
        "ema_21_slope",             # EMA 21 slope normalized
        "ema_50_slope",             # EMA 50 slope normalized
        "close_ema9_diff",          # Price distance from EMA 9
        "close_ema21_diff",         # Price distance from EMA 21
        "close_ema50_diff",         # Price distance from EMA 50
        "ema_9_21_spread",          # EMA 9-21 spread (fast cross)
        "ema_21_50_spread",         # EMA 21-50 spread (slow cross)
        "ema_9_50_spread",          # EMA 9-50 spread (trend)
        "ema_9_21_spread_slope",    # Spread direction change
        "ema_21_50_spread_slope",   # Spread direction change
        "macd_line",                # MACD line (normalized)
        "macd_signal",              # MACD signal (normalized)
        "macd_hist",                # MACD histogram (normalized)
        "bb_percent_b",             # Bollinger %B
        "bb_bandwidth",             # Bollinger bandwidth
        "stoch_k",                  # Stochastic %K
        "stoch_d",                  # Stochastic %D
        "volatility_ratio",         # Volatility regime
        "volume_ratio",             # Volume anomaly
        "obv_slope",                # OBV momentum
        "adx_14",                   # Trend strength 0-1
        "ema_cross_signal",         # EMA derived signal
    ]

    # Keep only features that exist in the dataframe
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"[Indicators] Computed {len(feature_cols)} features on {len(df)} rows")
    print(f"[Indicators] Features: {feature_cols}")

    return df, feature_cols


def download_eurusd_data(save_path: str = "data/EURUSD_Hourly.csv"):
    """
    Download EURUSD forex data using yfinance.
    Saves to CSV with the expected format.
    """
    import yfinance as yf

    print(f"[Data] Downloading EURUSD hourly data (10 years)...")
    ticker = yf.Ticker("EURUSD=X")
    df = ticker.history(period="10y", interval="1h")

    if df.empty:
        raise ValueError("No data downloaded. Check yfinance / internet connectivity.")

    df = df.reset_index()
    df = df.rename(
        columns={
            "Datetime": "Time (EET)",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
    )

    df = df[["Time (EET)", "Open", "High", "Low", "Close", "Volume"]]
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[Data] EURUSD hourly data saved to {save_path} ({len(df)} rows)")

    return df
