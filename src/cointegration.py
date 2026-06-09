# src/cointegration.py

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from itertools import combinations

def find_cointegrated_pairs(prices, pvalue_threshold=0.05):
    """
    Tests every possible pair of stocks for cointegration.
    Returns a DataFrame of pairs that pass the test.
    
    prices            : DataFrame where columns are stocks, rows are dates
    pvalue_threshold  : pairs with p-value below this are considered cointegrated
    """
    
    stocks = prices.columns.tolist()
    all_pairs = list(combinations(stocks, 2))  # every possible pair
    
    print(f"Testing {len(all_pairs)} pairs for cointegration...")
    
    results = []
    
    for stock_a, stock_b in all_pairs:
        
        series_a = prices[stock_a]
        series_b = prices[stock_b]
        
        # coint() runs the Engle-Granger test
        # it returns 3 things — test statistic, p-value, and critical values
        # we only care about the p-value
        _, pvalue, _ = coint(series_a, series_b)
        
        if pvalue < pvalue_threshold:
            results.append({
                "stock_a"  : stock_a,
                "stock_b"  : stock_b,
                "pvalue"   : round(pvalue, 4),
            })
    
    # sort by p-value — strongest pairs at the top
    pairs_df = pd.DataFrame(results).sort_values("pvalue").reset_index(drop=True)
    
    print(f"✅ Found {len(pairs_df)} cointegrated pairs out of {len(all_pairs)} tested")
    return pairs_df


if __name__ == "__main__":
    from data_loader import load_prices
    
    prices = load_prices()
    pairs  = find_cointegrated_pairs(prices)
    
    print("\nTop 10 cointegrated pairs:")
    print(pairs.head(10))