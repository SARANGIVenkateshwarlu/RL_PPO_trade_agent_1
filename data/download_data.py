import yfinance as yf
import pandas as pd
import os


def download_eurusd_data(save_path: str = "data/EURUSD_Hourly.csv"):
    """
    Download EURUSD forex data using yfinance.
    Tries multiple intervals to get the best data coverage.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    print("[Data] Attempting to download EURUSD hourly data...")

    df = None
    for interval, period in [("60m", "2y"), ("1h", "2y"), ("1d", "10y")]:
        try:
            print(f"[Data] Trying interval={interval}, period={period}...")
            raw = yf.download("EURUSD=X", period=period, interval=interval, progress=False)
            if raw is not None and len(raw) > 0:
                # Flatten multi-index columns (yfinance adds ticker level)
                df = raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()
                print(f"[Data] Success — {len(df)} rows, columns: {list(df.columns)}")
                break
        except Exception as e:
            print(f"[Data] Failed: {e}")

    if df is None or len(df) == 0:
        raise ValueError("Failed to download any EURUSD data.")

    # Find and rename datetime column
    date_col = None
    for c in df.columns:
        if isinstance(c, str) and c.lower() in ("datetime", "date", "index", ""):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]  # Assume first column is date
    df = df.rename(columns={date_col: "Time (EET)"})

    # Standardize OHLCV column names
    rename_map = {}
    for col in df.columns:
        cl = col.lower() if isinstance(col, str) else str(col).lower()
        if "open" in cl:
            rename_map[col] = "Open"
        elif "high" in cl:
            rename_map[col] = "High"
        elif "low" in cl:
            rename_map[col] = "Low"
        elif "close" in cl:
            rename_map[col] = "Close"
        elif "volume" in cl:
            rename_map[col] = "Volume"
    df = df.rename(columns=rename_map)

    # Keep only needed columns, add Volume if missing
    needed = ["Time (EET)", "Open", "High", "Low", "Close", "Volume"]
    available = [c for c in needed if c in df.columns]
    df = df[available]
    if "Volume" not in df.columns:
        df["Volume"] = 1.0

    df.to_csv(save_path, index=False)
    print(f"[Data] EURUSD data saved to {save_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    download_eurusd_data("data/EURUSD_Hourly.csv")
