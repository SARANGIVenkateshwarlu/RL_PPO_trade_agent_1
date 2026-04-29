"""
Market Regime Detector: S&P 500 index trend classification.

Rules:
  - BULL:  SPX Close > 50 EMA AND SPX Close > 150 EMA  (uptrend)
  - BEAR:  SPX Close < 50 EMA AND SPX Close < 150 EMA  (downtrend)
  - NEUTRAL: Otherwise (price between EMAs = consolidation/transition)

The regime dictates which side the RL agent can trade:
  - BULL regime  → BUY only (long)
  - BEAR regime  → SELL only (short)
  - NEUTRAL regime → HOLD only (no new positions)
"""
import pandas as pd
import numpy as np


def compute_sp500_regime(sp500_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market regime columns to the S&P 500 dataframe.

    Returns sp500_df with added columns:
      - ema_50, ema_150: EMA values
      - regime: 1=BULL, -1=BEAR, 0=NEUTRAL
      - regime_label: string label
    """
    df = sp500_df.sort_values("Date").copy()

    df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["ema_150"] = df["Close"].ewm(span=150, adjust=False).mean()

    above_50 = df["Close"] > df["ema_50"]
    above_150 = df["Close"] > df["ema_150"]

    conditions = [
        above_50 & above_150,
        (~above_50) & (~above_150),
    ]
    choices = [1, -1]
    df["regime"] = np.select(conditions, choices, default=0)
    df["regime_label"] = df["regime"].map({1: "BULL", -1: "BEAR", 0: "NEUTRAL"})

    # Forward-fill regime on initial NaN rows (before EMAs are computed)
    df["regime"] = df["regime"].fillna(0).astype(int)
    df["regime_label"] = df["regime_label"].fillna("NEUTRAL")

    return df


def merge_regime_to_stocks(stocks_df: pd.DataFrame, sp500_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge S&P 500 regime information into the stocks dataframe by date.
    Each stock row gets the prevailing market regime for that date.
    """
    # Ensure Date columns are compatible datetime types
    stocks_df = stocks_df.copy()
    sp500_df = sp500_df.copy()
    if "Date" in stocks_df.columns:
        stocks_df["Date"] = pd.to_datetime(stocks_df["Date"]).dt.normalize()
    if "Date" in sp500_df.columns:
        sp500_df["Date"] = pd.to_datetime(sp500_df["Date"]).dt.normalize()

    regime_cols = sp500_df[["Date", "regime", "regime_label", "ema_50", "ema_150", "Close"]].copy()
    regime_cols = regime_cols.rename(columns={
        "Close": "sp500_close",
        "ema_50": "sp500_ema_50",
        "ema_150": "sp500_ema_150",
    })

    merged = stocks_df.merge(regime_cols, on="Date", how="left")
    merged["regime"] = merged["regime"].fillna(0).astype(int)
    merged["regime_label"] = merged["regime_label"].fillna("NEUTRAL")

    # Forward fill any remaining NaN SPX columns
    for c in ["sp500_close", "sp500_ema_50", "sp500_ema_150"]:
        if c in merged.columns:
            merged[c] = merged[c].ffill()

    return merged


def get_regime_stats(sp500_df: pd.DataFrame) -> dict:
    """Compute regime distribution statistics."""
    df = compute_sp500_regime(sp500_df)
    total = len(df[df["ema_150"].notna()])
    bull = (df["regime"] == 1).sum()
    bear = (df["regime"] == -1).sum()
    neutral = (df["regime"] == 0).sum()
    return {
        "total_days": total,
        "bull_pct": bull / total * 100,
        "bear_pct": bear / total * 100,
        "neutral_pct": neutral / total * 100,
    }
