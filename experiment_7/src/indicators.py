"""
Experiment 7: Feature Engineering — Inside Bar Trend-Following Strategy
Constraints: SP500 EMA regime, 52wk high proximity, low vol, strong uptrend,
             weekly/daily inside bars, low-holding, entry above prev high.
"""
import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators for Experiment 7."""
    df = df.sort_values(["Symbol", "Date"]).copy()
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    eps, n = 1e-8, len(df)

    # --- S&P 500 regime: must be above 50/100/200 EMA ---
    if "sp500_close" in df.columns and df["sp500_close"].notna().sum() > 0:
        df["sp500_ema_50"] = df["sp500_close"].ewm(span=50, adjust=False).mean()
        df["sp500_ema_100"] = df["sp500_close"].ewm(span=100, adjust=False).mean()
        df["sp500_ema_200"] = df["sp500_close"].ewm(span=200, adjust=False).mean()
        df["spx_bull"] = ((df["sp500_close"] > df["sp500_ema_50"]) &
                          (df["sp500_close"] > df["sp500_ema_100"]) &
                          (df["sp500_close"] > df["sp500_ema_200"])).astype(int)
        df["spx_above_ema50"] = (df["sp500_close"] / (df["sp500_ema_50"] + eps) - 1) * 100
        df["spx_above_ema200"] = (df["sp500_close"] / (df["sp500_ema_200"] + eps) - 1) * 100
    else:
        df["spx_bull"] = 1
        df["spx_above_ema50"] = 0.0
        df["spx_above_ema200"] = 0.0

    # --- 52-week high: stock within 25% of 52-week high ---
    df["high_52w"] = high.rolling(252, min_periods=1).max()
    df["pct_below_52w_high"] = ((df["high_52w"] - close) / (df["high_52w"] + eps) * 100).clip(0, 100)
    df["near_52w_high"] = (df["pct_below_52w_high"] < 25).astype(int)

    # --- Returns ---
    df["ret_1d"] = close.pct_change() * 100
    df["ret_5d"] = close.pct_change(5) * 100
    df["ret_1m"] = close.pct_change(21) * 100
    df["ret_3m"] = close.pct_change(63) * 100
    df["ret_6m"] = close.pct_change(126) * 100
    df["ret_12m"] = close.pct_change(252) * 100

    # --- RTI (Range Tightening Index): 21-day range contraction ---
    # RTI_t = 100 * (Range_t - Min_Range_21d) / (Max_Range_21d - Min_Range_21d)
    # BUY constraint: RTI_t <= 15 OR RTI_{t-1} <= 15 (bottom 15% of range)
    df["daily_range"] = high - low
    df["rti_max_range_21"] = df["daily_range"].rolling(21, min_periods=5).max()
    df["rti_min_range_21"] = df["daily_range"].rolling(21, min_periods=5).min()
    rti_denom = df["rti_max_range_21"] - df["rti_min_range_21"]
    df["rti"] = np.where(rti_denom > 0, 100 * (df["daily_range"] - df["rti_min_range_21"]) / rti_denom, 50.0)
    df["rti"] = df["rti"].clip(0, 100)
    df["rti_prev"] = df["rti"].shift(1).fillna(100)
    df["low_vol"] = ((df["rti"] <= 15) | (df["rti_prev"] <= 15)).astype(int)

    # --- Strong prior uptrend ---
    df["uptrend_1m"] = (df["ret_1m"] > 25).astype(int)
    df["uptrend_3m"] = (df["ret_3m"] > 30).astype(int)
    df["uptrend_6m"] = (df["ret_6m"] > 30).astype(int)
    df["uptrend_12m"] = (df["ret_12m"] > 30).astype(int)
    # Uptrend: >25% over 1 month OR >30% over 3-12 months
    df["strong_uptrend"] = (
        df["uptrend_1m"] |
        df["uptrend_3m"] |
        df["uptrend_6m"] |
        df["uptrend_12m"]
    ).astype(int)

    # --- Weekly inside bar: current week's candle inside previous week's ---
    df["week_high"] = high.rolling(5).max()
    df["week_low"] = low.rolling(5).min()
    df["prev_week_high"] = df["week_high"].shift(5)
    df["prev_week_low"] = df["week_low"].shift(5)
    df["weekly_inside_bar"] = ((df["week_high"] <= df["prev_week_high"]) &
                               (df["week_low"] >= df["prev_week_low"])).astype(int)

    # --- Daily inside bar: current day inside previous day's range ---
    df["daily_inside_bar"] = ((high <= high.shift(1)) &
                              (low >= low.shift(1))).astype(int)

    # --- Current day low higher than previous day low ---
    df["low_holding"] = (low > low.shift(1)).astype(int)

    # --- Entry trigger: price crosses above previous day high ---
    df["cross_above_prev_high"] = (high > high.shift(1)).astype(int)

    # --- Entry candle within 2% of previous day close ---
    df["entry_near_prev_close"] = (
        (np.abs(close - close.shift(1)) / (close.shift(1) + eps) < 0.02)
    ).fillna(0).astype(int)

    # --- Composite entry gate: all conditions must be true ---
    df["entry_gate"] = (
        (df["spx_bull"] == 1) &
        (df["near_52w_high"] == 1) &
        (df["low_vol"] == 1) &
        (df["strong_uptrend"] == 1) &
        (df["weekly_inside_bar"] == 1) &
        (df["daily_inside_bar"] == 1) &
        (df["low_holding"] == 1) &
        (df["cross_above_prev_high"] == 1) &
        (df["entry_near_prev_close"] == 1)
    ).astype(int)

    # --- EMAs for exit rules ---
    df["ema_10"] = close.ewm(span=10, adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["close_vs_ema10"] = (close - df["ema_10"]) / (close + eps) * 100
    df["close_vs_ema21"] = (close - df["ema_21"]) / (close + eps) * 100
    df["close_below_ema10"] = (close < df["ema_10"]).astype(int)
    df["close_below_ema21"] = (close < df["ema_21"]).astype(int)

    # --- ATR ---
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high.iloc[i] - low.iloc[i],
                    abs(high.iloc[i] - close.iloc[i - 1]),
                    abs(low.iloc[i] - close.iloc[i - 1]))
    atr = np.full(n, np.nan)
    if n > 14:
        atr[13] = np.mean(tr[1:14])
    for i in range(14, n):
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    df["atr_14"] = pd.Series(atr, index=df.index)
    df["atr_pct"] = df["atr_14"] / (close + eps) * 100

    # --- RSI ---
    delta = close.diff().values
    g, l = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    ag, al = np.full(n, np.nan), np.full(n, np.nan)
    if n > 14:
        ag[13], al[13] = np.mean(g[1:14]), np.mean(l[1:14])
    for i in range(14, n):
        ag[i] = (ag[i - 1] * 13 + g[i]) / 14
        al[i] = (al[i - 1] * 13 + l[i]) / 14
    df["rsi_14"] = pd.Series(100 - 100 / (1 + ag / np.where(al == 0, np.nan, al)), index=df.index)

    # --- MACD ---
    e12, e26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df["macd_line"] = e12 - e26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist_norm"] = (df["macd_line"] - df["macd_signal"]) / (close + eps) * 100

    # --- Volume ---
    df["volume_sma_20"] = vol.rolling(20).mean()
    df["volume_ratio"] = vol / (df["volume_sma_20"] + eps)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)

    # --- ADX ---
    h, l_arr, c = high.values, low.values, close.values
    tr2, pdm, mdm = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(1, n):
        tr2[i] = max(h[i] - l_arr[i], abs(h[i] - c[i - 1]), abs(l_arr[i] - c[i - 1]))
        up, down = h[i] - h[i - 1], l_arr[i - 1] - l_arr[i]
        pdm[i] = up if up > down and up > 0 else 0
        mdm[i] = down if down > up and down > 0 else 0
    ats, ps, ms = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    if n > 14:
        ats[13], ps[13], ms[13] = np.mean(tr2[1:14]), np.mean(pdm[1:14]), np.mean(mdm[1:14])
    for i in range(14, n):
        ats[i] = (ats[i - 1] * 13 + tr2[i]) / 14
        ps[i] = (ps[i - 1] * 13 + pdm[i]) / 14
        ms[i] = (ms[i - 1] * 13 + mdm[i]) / 14
    dx = 100 * np.abs(ps - ms) / (ps + ms + eps)
    adx_arr = np.full(n, np.nan)
    if n > 27:
        adx_arr[26] = np.mean(dx[14:27])
    for i in range(27, n):
        adx_arr[i] = (adx_arr[i - 1] * 13 + dx[i]) / 14
    df["adx_14"] = pd.Series(adx_arr / 100.0, index=df.index)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    base = [
        "Open", "High", "Low", "Close",
        "spx_above_ema50", "spx_above_ema200",  # SPX regime
        "pct_below_52w_high",  # 52-week proximity
        "rti",  # Range Tightening Index (21-day)
        "ret_1m", "ret_3m", "ret_6m",  # Uptrend momentum
        "weekly_inside_bar", "daily_inside_bar",  # Inside bars
        "low_holding", "cross_above_prev_high", "entry_near_prev_close",  # Entry signals
        "entry_gate",  # Composite entry gate
        "close_vs_ema10", "close_vs_ema21",  # EMA distances for exit
        "atr_pct", "rsi_14", "macd_hist_norm", "adx_14",
        "ret_1d", "ret_5d", "volume_ratio",
    ]
    return [c for c in base if c in df.columns]
