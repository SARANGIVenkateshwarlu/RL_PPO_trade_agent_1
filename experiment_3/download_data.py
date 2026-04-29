"""
Download S&P 500 index + top 25 US stocks daily data (5 years)
for Experiment 3: Market Regime Filtered Trading Agent.
"""
import os
import pandas as pd
import yfinance as yf


def download_sp500(save_path="experiment_3/data/SP500_daily.csv", years=5):
    """Download S&P 500 index daily data."""
    import datetime
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=years * 365 + 30)

    if os.path.exists(save_path):
        df = pd.read_csv(save_path, parse_dates=["Date"])
        print(f"SP500 data already exists: {save_path} ({len(df)} days)")
        return df

    print("Downloading S&P 500 index (^GSPC) daily data...")
    ticker = yf.Ticker("^GSPC")
    df = ticker.history(start=start, end=end, interval="1d")

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
    df["Date"] = pd.to_datetime(df["Date"])
    df.to_csv(save_path, index=False)
    print(f"SP500 data saved: {save_path} ({len(df)} days)")
    return df


def download_stocks(save_path="experiment_3/data/stocks_daily.parquet", years=5):
    """Download top 25 US stocks daily data from tickers file."""
    import datetime
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=years * 365 + 30)

    if os.path.exists(save_path):
        print(f"Stock data already exists: {save_path}")
        return pd.read_parquet(save_path)

    tickers_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Tickers_list_USA.csv")
    if not os.path.exists(tickers_path):
        tickers_path = "data/Tickers_list_USA.csv"
    if not os.path.exists(tickers_path):
        # Go up to parent
        tickers_path = "../data/Tickers_list_USA.csv"

    tk_df = pd.read_csv(tickers_path)
    symbols = tk_df["Symbol"].dropna().tolist()
    skip = {"BRK.A", "BRK-B", "BF.A", "BF.B"}
    symbols = [s for s in symbols if s not in skip][:25]

    print(f"Downloading {len(symbols)} stocks daily data (5 years)...")
    all_data = []
    for i, sym in enumerate(symbols):
        try:
            t = yf.Ticker(sym)
            d = t.history(start=start, end=end, interval="1d")
            if d.empty:
                continue
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d.reset_index()
            rename = {}
            for c in d.columns:
                cl = str(c).lower()
                if "open" in cl: rename[c] = "Open"
                elif "high" in cl: rename[c] = "High"
                elif "low" in cl: rename[c] = "Low"
                elif "close" in cl: rename[c] = "Close"
                elif "volume" in cl: rename[c] = "Volume"
                elif "date" in cl: rename[c] = "Date"
            d = d.rename(columns=rename)
            d = d[["Date", "Open", "High", "Low", "Close", "Volume"]]
            d["Date"] = pd.to_datetime(d["Date"])
            d["Symbol"] = sym
            all_data.append(d)
            print(f"  [{i+1}/{len(symbols)}] {sym}: {len(d)} days")
        except Exception as e:
            print(f"  [{i+1}/{len(symbols)}] {sym}: FAILED ({e})")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined.to_parquet(save_path)
    print(f"Saved {len(combined)} rows across {combined['Symbol'].nunique()} symbols")
    return combined


if __name__ == "__main__":
    sp500 = download_sp500()
    stocks = download_stocks()
    print("Data download complete!")
