# src/backtest.py

import pandas as pd
import numpy as np


def run_backtest(pair_data, stock_a, stock_b):
    """
    Simulates trades based on the signals in pair_data.
    Returns a summary of performance metrics and a log of every trade.

    pair_data : DataFrame from get_pair_data() in signals.py
    stock_a   : name of stock A (string)
    stock_b   : name of stock B (string)
    """

    trades      = []       # log of every completed trade
    in_trade    = False    # are we currently in a trade?
    entry_day   = None     # when did we enter?
    entry_price_a = None
    entry_price_b = None
    trade_side  = None     # +1 or -1

    for date, row in pair_data.iterrows():

        signal  = row["signal"]
        price_a = row["price_a"]
        price_b = row["price_b"]
        zscore  = row["zscore"]

        # ── Entry logic ───────────────────────────────────────────────────
        if not in_trade and signal != 0:
            in_trade      = True
            entry_day     = date
            entry_price_a = price_a
            entry_price_b = price_b
            trade_side    = signal

        # ── Exit logic ────────────────────────────────────────────────────
        # exit when zscore crosses back to 0
        # ── Exit logic ────────────────────────────────────────────────────
        # exit when zscore crosses back to 0 OR hits stop loss at ±3
        elif in_trade:
            exit_now   = False
            stop_loss  = False

            # normal exit — spread reverted to mean
            if trade_side == 1  and zscore >= 0:
                exit_now = True
            elif trade_side == -1 and zscore <= 0:
                exit_now = True

            # stop loss — spread kept diverging beyond 3σ
            if trade_side == 1  and zscore < -3:
                exit_now  = True
                stop_loss = True
            elif trade_side == -1 and zscore >  3:
                exit_now  = True
                stop_loss = True

            if exit_now:
                # calculate return
                # if trade_side = +1 : we bought A and shorted B
                #   profit from A = (exit - entry) / entry
                #   profit from B = (entry - exit) / entry  (short)
                #   total return  = average of both legs

                return_a = (price_a - entry_price_a) / entry_price_a
                return_b = (price_b - entry_price_b) / entry_price_b

                if trade_side == 1:
                    trade_return = (return_a - return_b) / 2
                else:
                    trade_return = (return_b - return_a) / 2

                # deduct transaction costs — 0.1% per trade (entry + exit)
                TRANSACTION_COST = 0.001
                trade_return = trade_return - TRANSACTION_COST

                holding_period = (date - entry_day).days

                trades.append({
                    "entry_date"     : entry_day,
                    "exit_date"      : date,
                    "side"           : "Buy A" if trade_side == 1 else "Short A",
                    "return_pct"     : round(trade_return * 100, 3),
                    "holding_days"   : holding_period,
                    "win"            : trade_return > 0,
                    "stop_loss"      : stop_loss,
                })

                # reset
                in_trade    = False
                entry_day   = None
                trade_side  = None

    # ── Build trades DataFrame ─────────────────────────────────────────────
    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print("⚠️  No completed trades found for this pair.")
        return None, None

    # ── Calculate summary metrics ──────────────────────────────────────────
    total_trades      = len(trades_df)
    winning_trades    = trades_df["win"].sum()
    hit_rate          = round(winning_trades / total_trades * 100, 1)
    avg_return        = round(trades_df["return_pct"].mean(), 3)
    total_return      = round(trades_df["return_pct"].sum(), 3)
    avg_holding       = round(trades_df["holding_days"].mean(), 1)

    # sharpe ratio — using per-trade returns
    returns           = trades_df["return_pct"] / 100
    sharpe            = round(
        returns.mean() / returns.std() * np.sqrt(252), 2
    ) if returns.std() != 0 else 0.0

    summary = {
        "pair"           : f"{stock_a} / {stock_b}",
        "total_trades"   : total_trades,
        "hit_rate_pct"   : hit_rate,
        "avg_return_pct" : avg_return,
        "total_return_pct": total_return,
        "avg_holding_days": avg_holding,
        "sharpe_ratio"   : sharpe,
    }

    return summary, trades_df

def run_lstm_backtest(prices, train, test, stock_a, stock_b):
    """
    Runs backtest using LSTM predicted z-scores instead of raw z-scores.
    
    1. Gets z-score on train period → trains LSTM
    2. Predicts z-scores on full period
    3. Uses predicted z-scores to generate signals
    4. Backtests those signals on test period only
    """
    from src.lstm_model import train_lstm, predict_zscore
    from src.signals import get_pair_data, calculate_hedge_ratio, calculate_spread, generate_signals
    
    # get z-scores for train and full period
    train_data, hedge_ratio = get_pair_data(train, stock_a, stock_b)
    full_data,  _           = get_pair_data(prices, stock_a, stock_b)

    zscore_train = train_data["zscore"].dropna()
    zscore_full  = full_data["zscore"].dropna()

    # train LSTM on train period z-scores
    print(f"  Training LSTM for {stock_a} & {stock_b}...")
    model = train_lstm(zscore_train, epochs=50)

    # predict z-scores on full period
    predicted_zscore = predict_zscore(model, zscore_full)

    # get test period prices
    test_data, _ = get_pair_data(test, stock_a, stock_b)

    # align predicted z-scores to test period only
    predicted_test = predicted_zscore[predicted_zscore.index.isin(test_data.index)]

    if len(predicted_test) == 0:
        return None, None

    # build pair_data using predicted z-scores
    lstm_pair_data = test_data.copy()
    lstm_pair_data["zscore"] = predicted_test

    # generate signals from predicted z-scores
    lstm_pair_data["signal"] = generate_signals(lstm_pair_data["zscore"])

    # run backtest
    summary, trades = run_backtest(lstm_pair_data, stock_a, stock_b)
    return summary, trades

if __name__ == "__main__":
    from data_loader import load_prices, split_prices
    from cointegration import find_cointegrated_pairs
    from signals import get_pair_data

    prices      = load_prices()
    train, test = split_prices(prices)

    print("\nFinding pairs on train data...")
    pairs = find_cointegrated_pairs(train)

    print(f"\nRunning Raw + LSTM backtest on all {len(pairs)} pairs...\n")

    all_raw_summaries  = []
    all_lstm_summaries = []

    for i in range(len(pairs)):
        row     = pairs.iloc[i]
        stock_a = row["stock_a"]
        stock_b = row["stock_b"]

        # raw backtest
        pair_data, _   = get_pair_data(test, stock_a, stock_b)
        raw_summary, _ = run_backtest(pair_data, stock_a, stock_b)

        # lstm backtest
        lstm_summary, _ = run_lstm_backtest(prices, train, test, stock_a, stock_b)

        if raw_summary:
            raw_summary["pair"] = f"{stock_a} / {stock_b}"
            all_raw_summaries.append(raw_summary)

        if lstm_summary:
            lstm_summary["pair"] = f"{stock_a} / {stock_b}"
            all_lstm_summaries.append(lstm_summary)

    # build DataFrames
    raw_df  = pd.DataFrame(all_raw_summaries)
    lstm_df = pd.DataFrame(all_lstm_summaries)

    # merge on pair for comparison
    merged = raw_df.merge(lstm_df, on="pair", suffixes=("_raw", "_lstm"))

    # find pairs where LSTM improved total return
    merged["lstm_improved"] = merged["total_return_pct_lstm"] > merged["total_return_pct_raw"]

    improved     = merged[merged["lstm_improved"] == True]
    not_improved = merged[merged["lstm_improved"] == False]

    print(f"\n{'='*50}")
    print(f"RESULTS ACROSS ALL {len(merged)} PAIRS")
    print(f"{'='*50}")
    print(f"LSTM improved return   : {len(improved)} pairs ({round(len(improved)/len(merged)*100, 1)}%)")
    print(f"LSTM did not improve   : {len(not_improved)} pairs ({round(len(not_improved)/len(merged)*100, 1)}%)")

    print(f"\n--- Top 10 pairs where LSTM helped most ---\n")
    improved_sorted = improved.copy()
    improved_sorted["improvement"] = improved_sorted["total_return_pct_lstm"] - improved_sorted["total_return_pct_raw"]
    improved_sorted = improved_sorted.sort_values("improvement", ascending=False)

    cols = ["pair", "total_return_pct_raw", "total_return_pct_lstm", "improvement", "hit_rate_pct_lstm", "sharpe_ratio_lstm"]
    print(improved_sorted[cols].head(10).to_string(index=False))

    print(f"\n--- Overall performance comparison ---\n")
    print(f"  {'Metric':<30} {'Raw':>10} {'LSTM':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Avg total return %':<30} {round(raw_df['total_return_pct'].mean(), 3):>10} {round(lstm_df['total_return_pct'].mean(), 3):>10}")
    print(f"  {'Avg hit rate %':<30} {round(raw_df['hit_rate_pct'].mean(), 3):>10} {round(lstm_df['hit_rate_pct'].mean(), 3):>10}")
    print(f"  {'Avg sharpe ratio':<30} {round(raw_df['sharpe_ratio'].mean(), 3):>10} {round(lstm_df['sharpe_ratio'].mean(), 3):>10}")
    print(f"  {'Profitable pairs':<30} {len(raw_df[raw_df['total_return_pct'] > 0]):>10} {len(lstm_df[lstm_df['total_return_pct'] > 0]):>10}")