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


if __name__ == "__main__":
    from data_loader import load_prices, split_prices
    from cointegration import find_cointegrated_pairs
    from signals import get_pair_data

    prices = load_prices()

    # split into train and test
    train, test = split_prices(prices)

    # find cointegrated pairs on TRAIN data only
    print("\nFinding pairs on train data...")
    pairs = find_cointegrated_pairs(train)

    print(f"\nBacktesting all {len(pairs)} pairs on test data...\n")

    all_summaries = []

    for i in range(len(pairs)):
        row     = pairs.iloc[i]
        stock_a = row["stock_a"]
        stock_b = row["stock_b"]

        pair_data, _ = get_pair_data(test, stock_a, stock_b)
        summary, trades = run_backtest(pair_data, stock_a, stock_b)

        if summary:
            all_summaries.append(summary)

    # convert to DataFrame and sort by total return
    results_df = pd.DataFrame(all_summaries)
    results_df = results_df.sort_values("total_return_pct", ascending=False).reset_index(drop=True)

    # filter only profitable pairs
    profitable = results_df[results_df["total_return_pct"] > 0]

    print(f"Results across all pairs:")
    print(f"Total pairs tested    : {len(results_df)}")
    print(f"Profitable pairs      : {len(profitable)}")
    print(f"Loss making pairs     : {len(results_df) - len(profitable)}")

    print(f"\n--- Top 10 profitable pairs ---\n")
    print(profitable.head(10).to_string(index=False))