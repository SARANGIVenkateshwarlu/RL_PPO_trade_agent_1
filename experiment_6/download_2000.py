"""
Download S&P 500 + up to 2000 US stocks (5 years daily data) for Experiment 6.
Handles batching, rate limiting, and resume-from-cache.
"""
import os
import sys
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_sp500(save_path: str = "experiment_6/data/SP500_daily.csv", years: int = 5):
    """Download S&P 500 index daily data."""
    if os.path.exists(save_path):
        df = pd.read_csv(save_path, parse_dates=["Date"])
        print(f"[SP500] Loaded from cache: {len(df)} days")
        return df

    print("[SP500] Downloading ^GSPC daily data...")
    end = datetime.now()
    start = end - timedelta(days=years * 365 + 30)

    df = yf.download("^GSPC", start=start, end=end, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    rename = {}
    for c in df.columns:
        cl = str(c).lower()
        if "open" in cl: rename[c] = "Open"
        elif "high" in cl: rename[c] = "High"
        elif "low" in cl: rename[c] = "Low"
        elif "close" in cl: rename[c] = "Close"
        elif "volume" in cl: rename[c] = "Volume"
        elif "date" in cl: rename[c] = "Date"
    df = df.rename(columns=rename)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"], utc=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[SP500] Saved: {save_path} ({len(df)} days)")
    return df


def read_tickers(max_tickers: int = 2000) -> list:
    """Read ticker symbols from CSV, filter problematic ones."""
    ticker_paths = [
        "data/Tickers_list_USA.csv",
        "../data/Tickers_list_USA.csv",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Tickers_list_USA.csv"),
    ]
    path = None
    for p in ticker_paths:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError("Tickers_list_USA.csv not found")

    df = pd.read_csv(path)
    symbols = df["Symbol"].dropna().tolist()

    # Filter out problematic tickers
    skip_prefixes = ("BRK", "BF.", "BF-", "^", "$")
    skip_exact = {"BRK.A", "BRK-B", "BF.A", "BF.B", "DISCK", "DISCA", "FOXA", "FOX"}
    symbols = [s for s in symbols
               if s not in skip_exact
               and not any(s.startswith(p) for p in skip_prefixes)]

    return symbols[:max_tickers]


def download_one_stock(sym: str, start, end, cache_dir: str) -> dict:
    """Download a single stock. Returns dict with data or error."""
    cache_path = os.path.join(cache_dir, f"{sym}.csv")
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, parse_dates=["Date"])
            return {"symbol": sym, "data": df, "status": "cached"}
        except:
            pass

    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(start=start, end=end, interval="1d")
        if df.empty:
            return {"symbol": sym, "data": None, "status": "empty"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()

        rename = {}
        for c in df.columns:
            cl = str(c).lower()
            if "open" in cl: rename[c] = "Open"
            elif "high" in cl: rename[c] = "High"
            elif "low" in cl: rename[c] = "Low"
            elif "close" in cl: rename[c] = "Close"
            elif "volume" in cl: rename[c] = "Volume"
            elif "date" in cl: rename[c] = "Date"
        df = df.rename(columns=rename)

        needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for c in needed:
            if c not in df.columns:
                if c == "Volume":
                    df[c] = 0.0
                else:
                    return {"symbol": sym, "data": None, "status": f"missing_col_{c}"}

        df = df[needed]
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df["Symbol"] = sym

        df.to_csv(cache_path, index=False)
        return {"symbol": sym, "data": df, "status": "downloaded"}

    except Exception as e:
        return {"symbol": sym, "data": None, "status": str(e)[:50]}


def download_stocks_bulk(symbols: list, save_path: str = "experiment_6/data/stocks_2000_daily.parquet",
                         years: int = 5, max_workers: int = 8):
    """
    Download stock data in parallel using ThreadPoolExecutor.

    Args:
        symbols: List of ticker symbols
        save_path: Output parquet file path
        years: Years of historical data
        max_workers: Number of concurrent download threads
    """
    if os.path.exists(save_path):
        print(f"[Stocks] Loading from cache: {save_path}")
        return pd.read_parquet(save_path)

    end = datetime.now()
    start = end - timedelta(days=years * 365 + 30)

    cache_dir = os.path.join(os.path.dirname(save_path), "stock_cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[Stocks] Downloading {len(symbols)} symbols ({max_workers} workers)...")
    print(f"[Stocks] Period: {start.date()} to {end.date()}")

    all_data = []
    completed = 0
    failed = 0
    cached = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one_stock, sym, start, end, cache_dir): sym
                   for sym in symbols}

        for future in as_completed(futures):
            result = future.result()
            if result["data"] is not None:
                all_data.append(result["data"])
                if result["status"] == "cached":
                    cached += 1
                else:
                    completed += 1
            else:
                failed += 1

            total_done = completed + failed + cached
            if total_done % 100 == 0 or total_done == len(symbols):
                print(f"  [{total_done}/{len(symbols)}] "
                      f"New={completed} Cached={cached} Failed={failed}", end="\r")

    print(f"\n[Stocks] Complete: {len(all_data)} succeeded, {failed} failed")

    if not all_data:
        raise RuntimeError("No stock data downloaded!")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # Filter out symbols with too few data points
    min_bars = 200
    counts = combined.groupby("Symbol").size()
    valid_syms = counts[counts >= min_bars].index
    combined = combined[combined["Symbol"].isin(valid_syms)]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined.to_parquet(save_path)
    print(f"[Stocks] Saved: {save_path}")
    print(f"  {len(combined):,} rows | {combined['Symbol'].nunique()} symbols | "
          f"{combined['Date'].min().date()} to {combined['Date'].max().date()}")

    return combined


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_stocks", type=int, default=2000, help="Max stocks to download")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download threads")
    parser.add_argument("--years", type=int, default=5, help="Years of history")
    parser.add_argument("--sp500_only", action="store_true", help="Only download S&P 500")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 6: Bulk Data Download")
    print("=" * 60)

    # S&P 500
    sp500 = download_sp500("experiment_6/data/SP500_daily.csv", args.years)
    print(f"  S&P 500: {len(sp500)} days | {sp500['Date'].min().date()} to {sp500['Date'].max().date()}")

    if not args.sp500_only:
        # Stocks
        tickers = read_tickers(max_tickers=args.max_stocks)
        print(f"\n  Tickers: {len(tickers)} (from Tickers_list_USA.csv)")

        stocks = download_stocks_bulk(
            tickers,
            save_path="experiment_6/data/stocks_2000_daily.parquet",
            years=args.years,
            max_workers=args.workers,
        )
        print(f"  Stocks: {len(stocks):,} rows | {stocks['Symbol'].nunique()} symbols")

    print("\nDone! Data ready for training.")
