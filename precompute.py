# precompute.py
# Run this once locally before deploying.
# Saves all backtest results to data/results.pkl

import pickle
import pandas as pd
from src.data_loader import load_prices, split_prices
from src.cointegration import find_cointegrated_pairs
from src.signals import get_pair_data
from src.backtest import run_backtest, run_lstm_backtest

def precompute_all():
    print("Loading data...")
    prices      = load_prices()
    train, test = split_prices(prices)

    print("\nFinding cointegrated pairs...")
    pairs = find_cointegrated_pairs(train)

    raw_summaries  = []
    lstm_summaries = []
    all_trades     = {}

    total = len(pairs)
    print(f"\nRunning backtests for {total} pairs...\n")

    for i in range(total):
        row     = pairs.iloc[i]
        stock_a = row["stock_a"]
        stock_b = row["stock_b"]
        pair_key = f"{stock_a} / {stock_b}"

        print(f"[{i+1}/{total}] {pair_key}")

        # raw backtest
        pair_data, _            = get_pair_data(test, stock_a, stock_b)
        raw_summary, raw_trades = run_backtest(pair_data, stock_a, stock_b)

        # lstm backtest
        lstm_summary, lstm_trades = run_lstm_backtest(
            prices, train, test, stock_a, stock_b
        )

        if raw_summary:
            raw_summary["pair"] = pair_key
            raw_summaries.append(raw_summary)

        if lstm_summary:
            lstm_summary["pair"] = pair_key
            lstm_summaries.append(lstm_summary)

        all_trades[pair_key] = {
            "pair_data"   : pair_data,
            "raw_trades"  : raw_trades,
            "lstm_trades" : lstm_trades,
        }

    # build DataFrames
    raw_df  = pd.DataFrame(raw_summaries).sort_values(
        "total_return_pct", ascending=False).reset_index(drop=True)
    lstm_df = pd.DataFrame(lstm_summaries).sort_values(
        "total_return_pct", ascending=False).reset_index(drop=True)

    # merged comparison
    merged = raw_df.merge(lstm_df, on="pair", suffixes=("_raw", "_lstm"))
    merged["improvement"]   = merged["total_return_pct_lstm"] - merged["total_return_pct_raw"]
    merged["lstm_improved"] = merged["improvement"] > 0

    # save everything to disk
    results = {
        "raw_df"     : raw_df,
        "lstm_df"    : lstm_df,
        "merged"     : merged,
        "all_trades" : all_trades,
        "train_period": f"{train.index[0].date()} to {train.index[-1].date()}",
        "test_period" : f"{test.index[0].date()} to {test.index[-1].date()}",
    }

    with open("data/results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\n✅ Results saved to data/results.pkl")
    print(f"   Raw profitable pairs  : {len(raw_df[raw_df['total_return_pct'] > 0])}")
    print(f"   LSTM profitable pairs : {len(lstm_df[lstm_df['total_return_pct'] > 0])}")


if __name__ == "__main__":
    precompute_all()