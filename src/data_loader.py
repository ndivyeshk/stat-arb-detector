# src/data_loader.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ── Stock universe ─────────────────────────────────────────────────────────
# 50 large-cap NSE stocks. ".NS" suffix = National Stock Exchange of India
STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS", "ONGC.NS",
    "NTPC.NS", "POWERGRID.NS", "TECHM.NS", "HCLTECH.NS", "NESTLEIND.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS",
    "BPCL.NS", "IOC.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "APOLLOHOSP.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "VEDL.NS", "INDUSINDBK.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "BRITANNIA.NS", "DABUR.NS", "MARICO.NS", "PIDILITIND.NS", "BERGEPAINT.NS",
]


def download_prices(tickers=STOCKS, years=5):
    """
    Downloads daily closing prices for all tickers over the last 2 years.
    Returns a DataFrame — rows are dates, columns are stocks.
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    print(f"Downloading {len(tickers)} stocks from {start_date.date()} to {end_date.date()}...")

    raw = yf.download(
        tickers     = tickers,
        start       = start_date,
        end         = end_date,
        auto_adjust = True,   # adjusts for stock splits & dividends
        progress    = True,
    )

    # yfinance gives us many price types (Open, High, Low, Close, Volume)
    # we only want the daily closing price
    prices = raw["Close"]

    # drop any stock with more than 5% missing data
    threshold = int(0.95 * len(prices))
    prices = prices.dropna(axis=1, thresh=threshold)

    # fill tiny gaps (e.g. trading halts) by carrying the last known price forward
    prices = prices.ffill().dropna()

    print(f"✅ Done — {prices.shape[0]} trading days x {prices.shape[1]} stocks")
    return prices


def save_prices(prices, path="data/prices.csv"):
    prices.to_csv(path)
    print(f"💾 Saved to {path}")


def load_prices(path="data/prices.csv"):
    """Load saved prices instead of re-downloading every time."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"📂 Loaded {df.shape[0]} rows x {df.shape[1]} stocks")
    return df

def split_prices(prices, train_ratio=0.7):
    """
    Splits price data into train and test sets.
    
    Train set : first 70% of dates → used to find pairs and hedge ratios
    Test set  : last 30% of dates  → used to generate signals and backtest
    
    This prevents overfitting — pairs are found on data the backtest never sees.
    """
    split_index = int(len(prices) * train_ratio)
    
    train = prices.iloc[:split_index]
    test  = prices.iloc[split_index:]
    
    print(f"Train set : {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
    print(f"Test set  : {test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)")
    
    return train, test

# ── Run this file directly to download and save ────────────────────────────
if __name__ == "__main__":
    prices = download_prices()
    save_prices(prices)