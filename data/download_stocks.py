"""
Stock data downloader: pulls 5 years of daily OHLCV data from yfinance
for tickers listed in Tickers_list_USA.csv
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def read_tickers(csv_path: str = "data/Tickers_list_USA.csv", max_tickers: int = 30):
    """Read ticker symbols from the CSV file. Returns top N by position."""
    df = pd.read_csv(csv_path)
    # The tickers appear in order of importance (market cap / technical rating)
    # Filter out problematic tickers (BRK.A has weird pricing)
    symbols = df["Symbol"].dropna().tolist()
    # Remove common problematic tickers
    skip = {"BRK.A", "BRK-B", "BF.A", "BF.B"}
    symbols = [s for s in symbols if s not in skip]
    return symbols[:max_tickers]


def download_stock_data(
    symbols: list,
    save_dir: str = "data/stocks",
    years: int = 5,
):
    """
    Download 5 years of daily data for each symbol.
    Saves individual CSV files and a combined parquet file.
    Returns combined DataFrame.
    """
    os.makedirs(save_dir, exist_ok=True)

    end = datetime.now()
    start = end - timedelta(days=years * 365 + 30)

    all_dfs = []
    failed = []

    for i, sym in enumerate(symbols):
        save_path = os.path.join(save_dir, f"{sym}.csv")
        if os.path.exists(save_path):
            df = pd.read_csv(save_path, parse_dates=["Date"])
            df["Symbol"] = sym
            all_dfs.append(df)
            continue

        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(start=start, end=end, interval="1d")
            if df.empty:
                failed.append(sym)
                continue

            # Flatten multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()

            # Standardize columns
            rename_map = {}
            for col in df.columns:
                cl = str(col).lower()
                if "open" in cl:
                    rename_map[col] = "Open"
                elif "high" in cl:
                    rename_map[col] = "High"
                elif "low" in cl:
                    rename_map[col] = "Low"
                elif "close" in cl:
                    rename_map[col] = "Close"
                elif "volume" in cl and "date" not in cl:
                    rename_map[col] = "Volume"
                elif "date" in cl or "time" in cl:
                    rename_map[col] = "Date"
            df = df.rename(columns=rename_map)

            # Ensure required columns
            needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
            for c in needed:
                if c not in df.columns:
                    if c == "Volume":
                        df[c] = 0
                    else:
                        raise KeyError(f"Missing column {c} in {sym}")

            df = df[needed]
            df = df.dropna(subset=["Close"])
            df["Symbol"] = sym

            df.to_csv(save_path, index=False)
            all_dfs.append(df)
            print(f"  [{i+1}/{len(symbols)}] {sym} — {len(df)} days")

        except Exception as e:
            print(f"  [{i+1}/{len(symbols)}] {sym} — FAILED: {e}")
            failed.append(sym)

    if not all_dfs:
        raise RuntimeError(f"No data downloaded. All symbols failed: {failed}")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = combined.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # Save combined
    combined_path = os.path.join(save_dir, "combined_stocks.parquet")
    combined.to_parquet(combined_path)
    print(f"\nCombined data saved: {combined_path}")
    print(f"Total: {len(symbols)} requested, {len(combined['Symbol'].unique())} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")

    return combined


if __name__ == "__main__":
    tickers = read_tickers(max_tickers=25)
    print(f"Selected {len(tickers)} tickers: {tickers[:10]}...")
    df = download_stock_data(tickers)
    print(f"Downloaded {len(df)} total rows across {df['Symbol'].nunique()} symbols")
