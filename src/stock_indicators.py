"""
Stock-specific technical indicators with:
  - EMA crossover features (EMA 9/21/50)
  - Bollinger Band squeeze detection (low volatility breakout predictor)
  - Trend strength (ADX)
  - Momentum (RSI, MACD)
  - Volume anomalies

All features are scale-invariant for RL agent consumption.
"""
import numpy as np
import pandas as pd


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def sma(series, length):
    return series.rolling(window=length, min_periods=length).mean()


def compute_stock_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for stock OHLCV data.
    Core strategy: EMA crossover + BB Squeeze for buy-only entries.

    Returns DataFrame with all indicators added.
    """
    df = df.copy()
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    vol = df["Volume"].values
    n = len(df)

    # ---- EMA (core crossover signals) ----
    df["ema_9"] = ema(df["Close"], 9)
    df["ema_21"] = ema(df["Close"], 21)
    df["ema_50"] = ema(df["Close"], 50)

    # EMA slopes normalized by price (%)
    eps = 1e-8
    df["ema_9_slope"] = df["ema_9"].pct_change() * 100
    df["ema_21_slope"] = df["ema_21"].pct_change() * 100

    # EMA spread (short - long), normalized by price %
    df["ema_9_21_spread"] = (df["ema_9"] - df["ema_21"]) / (df["Close"] + eps) * 100
    df["ema_9_50_spread"] = (df["ema_9"] - df["ema_50"]) / (df["Close"] + eps) * 100
    df["ema_21_50_spread"] = (df["ema_21"] - df["ema_50"]) / (df["Close"] + eps) * 100

    # EMA spread slope (acceleration/deceleration)
    df["ema_spread_slope"] = df["ema_9_21_spread"].diff()

    # EMA crossover binary signal
    df["ema_cross_up"] = ((df["ema_9"] > df["ema_21"]) & (df["ema_9"].shift(1) <= df["ema_21"].shift(1))).astype(int)
    df["ema_cross_down"] = ((df["ema_9"] < df["ema_21"]) & (df["ema_9"].shift(1) >= df["ema_21"].shift(1))).astype(int)

    # ---- Bollinger Bands (BB squeeze = volatility contraction) ----
    bb_period = 20
    bb_std = 2.0
    df["bb_middle"] = sma(df["Close"], bb_period)
    bb_std_val = df["Close"].rolling(window=bb_period, min_periods=bb_period).std()
    df["bb_upper"] = df["bb_middle"] + bb_std * bb_std_val
    df["bb_lower"] = df["bb_middle"] - bb_std * bb_std_val

    # BB width (volatility measure)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + eps)

    # BB squeeze: width at its lowest in the last 125 days (~6 months)
    squeeze_lookback = 125
    df["bb_width_min"] = df["bb_width"].rolling(window=squeeze_lookback, min_periods=squeeze_lookback).min()
    df["bb_width_rank"] = df["bb_width"] / (df["bb_width_min"] + eps)

    # BB squeeze signal: width < 1.1x historical minimum AND width < 5%
    df["bb_squeeze"] = ((df["bb_width_rank"] < 1.10) & (df["bb_width"] < 0.05)).astype(int)

    # BB %B (position within bands: 0=lower, 1=upper)
    df["bb_percent_b"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + eps)
    df["bb_percent_b"] = df["bb_percent_b"].clip(-0.5, 1.5)

    # ---- ATR (volatility in $ terms) ----
    tr_arr = np.zeros(n)
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr_arr[i] = max(tr1, tr2, tr3)
    atr_smooth = np.full(n, np.nan)
    atr_len = 14
    if n > atr_len:
        atr_smooth[atr_len - 1] = np.mean(tr_arr[1:atr_len])
    for i in range(atr_len, n):
        atr_smooth[i] = (atr_smooth[i - 1] * (atr_len - 1) + tr_arr[i]) / atr_len
    df["atr_14"] = pd.Series(atr_smooth, index=df.index)
    df["atr_pct"] = df["atr_14"] / (df["Close"] + eps) * 100  # ATR as % of price

    # ---- RSI ----
    delta = df["Close"].diff().values
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rsi_len = 14
    if n > rsi_len:
        avg_gain[rsi_len - 1] = np.mean(gain[1:rsi_len])
        avg_loss[rsi_len - 1] = np.mean(loss[1:rsi_len])
    for i in range(rsi_len, n):
        avg_gain[i] = (avg_gain[i - 1] * (rsi_len - 1) + gain[i]) / rsi_len
        avg_loss[i] = (avg_loss[i - 1] * (rsi_len - 1) + loss[i]) / rsi_len
    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    df["rsi_14"] = pd.Series(100.0 - 100.0 / (1.0 + rs), index=df.index)

    # ---- MACD ----
    ema_12 = ema(df["Close"], 12)
    ema_26 = ema(df["Close"], 26)
    df["macd_line"] = ema_12 - ema_26
    df["macd_signal"] = ema(df["macd_line"], 9)
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    df["macd_hist_norm"] = df["macd_hist"] / (df["Close"] + eps) * 100

    # ---- Volume ----
    df["volume_sma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_sma_20"].replace(0, np.nan) + eps)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

    # ---- ADX (trend strength) ----
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    atr_dm = np.full(n, np.nan)
    plus_dm_s = np.full(n, np.nan)
    minus_dm_s = np.full(n, np.nan)
    adx_len = 14
    if n > adx_len:
        atr_dm[adx_len - 1] = np.mean(tr_arr[1:adx_len])
        plus_dm_s[adx_len - 1] = np.mean(plus_dm[1:adx_len])
        minus_dm_s[adx_len - 1] = np.mean(minus_dm[1:adx_len])
    for i in range(adx_len, n):
        atr_dm[i] = (atr_dm[i - 1] * (adx_len - 1) + tr_arr[i]) / adx_len
        plus_dm_s[i] = (plus_dm_s[i - 1] * (adx_len - 1) + plus_dm[i]) / adx_len
        minus_dm_s[i] = (minus_dm_s[i - 1] * (adx_len - 1) + minus_dm[i]) / adx_len

    pdi = 100.0 * plus_dm_s / atr_dm
    mdi = 100.0 * minus_dm_s / atr_dm
    dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi + eps)
    adx_arr = np.full(n, np.nan)
    if n > adx_len * 2 - 1:
        adx_arr[adx_len * 2 - 2] = np.mean(dx[adx_len:adx_len * 2 - 1])
    for i in range(adx_len * 2 - 1, n):
        adx_arr[i] = (adx_arr[i - 1] * (adx_len - 1) + dx[i]) / adx_len
    df["adx_14"] = pd.Series(adx_arr / 100.0, index=df.index)

    # ---- Returns-based features ----
    df["returns_1d"] = df["Close"].pct_change() * 100
    df["returns_5d"] = df["Close"].pct_change(periods=5) * 100
    df["returns_20d"] = df["Close"].pct_change(periods=20) * 100

    # Volatility (20-day rolling std of daily returns)
    df["volatility_20"] = df["returns_1d"].rolling(20).std()

    # ---- Price relative to moving averages ----
    df["close_vs_ema21"] = (df["Close"] - df["ema_21"]) / (df["Close"] + eps) * 100
    df["close_vs_ema50"] = (df["Close"] - df["ema_50"]) / (df["Close"] + eps) * 100

    # ---- Combined entry signal (EMA cross up + BB squeeze) ----
    df["combined_buy_signal"] = (
        (df["ema_cross_up"] == 1) &
        (df["bb_squeeze"] == 1) &
        (df["adx_14"] > 0.15)  # Require minimal trend
    ).astype(int)

    # Drop NaN rows from rolling windows
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    return df


def prepare_agent_features(df: pd.DataFrame) -> tuple:
    """
    Extract scale-invariant feature columns for the RL agent.
    Returns (feature_df, feature_cols)
    """
    feature_cols = [
        # EMA features (core strategy)
        "ema_9_slope",
        "ema_21_slope",
        "ema_9_21_spread",
        "ema_9_50_spread",
        "ema_21_50_spread",
        "ema_spread_slope",
        # Bollinger Band features (squeeze)
        "bb_percent_b",
        "bb_width",
        "bb_width_rank",
        "bb_squeeze",
        # Volatility
        "atr_pct",
        "volatility_20",
        # Momentum
        "rsi_14",
        "macd_hist_norm",
        # Trend
        "adx_14",
        # Returns
        "returns_1d",
        "returns_5d",
        "returns_20d",
        # Price position
        "close_vs_ema21",
        "close_vs_ema50",
        # Volume
        "volume_ratio",
        # Combined signal (informational only)
        "combined_buy_signal",
    ]

    available = [c for c in feature_cols if c in df.columns]
    return df[available], available


def load_and_process_stocks(data_path="data/stocks/combined_stocks.parquet"):
    """
    Load combined stock data, compute indicators per symbol,
    return a single DataFrame indexed by (Symbol, Date) with features.
    """
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    results = []
    symbols = sorted(df["Symbol"].unique())
    print(f"Processing {len(symbols)} symbols...")

    for sym in symbols:
        sym_df = df[df["Symbol"] == sym].sort_values("Date").copy()
        if len(sym_df) < 200:
            continue  # Skip symbols with insufficient data

        sym_df = compute_stock_indicators(sym_df)
        sym_df["Symbol"] = sym
        results.append(sym_df)

    combined = pd.concat(results, ignore_index=True)
    combined = combined.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    print(f"Processed: {len(combined)} rows across {combined['Symbol'].nunique()} symbols")

    # Get feature columns
    _, feature_cols = prepare_agent_features(combined)

    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    return combined, feature_cols
