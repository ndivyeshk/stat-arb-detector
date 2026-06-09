    # src/signals.py

import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def calculate_hedge_ratio(series_a, series_b):
    """
    Finds the hedge ratio between two stocks using OLS regression.
    Answers: how many units of B move with 1 unit of A?
    
    We regress A on B — the coefficient is our hedge ratio.
    """
    b_with_const = add_constant(series_b)
    model        = OLS(series_a, b_with_const).fit()
    hedge_ratio  = model.params.iloc[1]
    return hedge_ratio


def calculate_spread(series_a, series_b, hedge_ratio):
    """
    Calculates the spread between two stocks.
    spread = Price_A - (hedge_ratio x Price_B)
    """
    spread = series_a - (hedge_ratio * series_b)
    return spread


def calculate_zscore(spread, window=60):
    """
    Converts the spread into a z-score using a rolling window.
    
    Z = (current_spread - rolling_mean) / rolling_std
    
    window : number of trading days to look back (60 = ~3 months)
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std  = spread.rolling(window=window).std()
    zscore       = (spread - rolling_mean) / rolling_std
    return zscore


def generate_signals(zscore, upper=2.0, lower=-2.0):
    """
    Fires buy/sell signals based on z-score thresholds.
    
    When Z > +2 : signal = -1 (short A, buy B)
    When Z < -2 : signal = +1 (buy A, short B)
    When Z crosses 0 : signal = 0 (close the trade)
    
    Returns a Series of signals — same index as zscore.
    """
    signals = pd.Series(index=zscore.index, dtype=float)
    signals[:] = 0  # default — no position

    signals[zscore >  upper] = -1   # spread too high → short A, buy B
    signals[zscore <  lower] =  1   # spread too low  → buy A, short B

    return signals


def get_pair_data(prices, stock_a, stock_b, window=60):
    """
    Master function — given two stock names, returns everything we need:
    hedge ratio, spread, z-score, and signals.
    All returned as a single clean DataFrame.
    """
    series_a = prices[stock_a]
    series_b = prices[stock_b]

    hedge_ratio = calculate_hedge_ratio(series_a, series_b)
    spread      = calculate_spread(series_a, series_b, hedge_ratio)
    zscore      = calculate_zscore(spread, window=window)
    signals     = generate_signals(zscore)

    df = pd.DataFrame({
        "price_a"     : series_a,
        "price_b"     : series_b,
        "spread"      : spread,
        "zscore"      : zscore,
        "signal"      : signals,
    })

    return df, hedge_ratio


if __name__ == "__main__":
    from data_loader import load_prices
    from cointegration import find_cointegrated_pairs

    prices = load_prices()
    pairs  = find_cointegrated_pairs(prices)

    # test on the top pair
    top_pair = pairs.iloc[0]
    stock_a  = top_pair["stock_a"]
    stock_b  = top_pair["stock_b"]

    print(f"\nGenerating signals for: {stock_a} & {stock_b}")

    pair_data, hedge_ratio = get_pair_data(prices, stock_a, stock_b)

    print(f"Hedge ratio : {hedge_ratio:.4f}")
    print(f"Total days  : {len(pair_data)}")
    print(f"Buy signals  (+1) : {(pair_data['signal'] ==  1).sum()}")
    print(f"Sell signals (-1) : {(pair_data['signal'] == -1).sum()}")
    print(f"No position  ( 0) : {(pair_data['signal'] ==  0).sum()}")

    print("\nSample of the data:")
    print(pair_data.tail(10))