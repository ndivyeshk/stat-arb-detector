# 📈 Statistical Arbitrage Signal Detector

A pairs trading system built on NSE (Indian) stocks that automatically detects
cointegrated pairs, generates z-score based trading signals, and backtests
the strategy on out-of-sample data.

Live Demo : https://stat-arb-detector-gzwbjuusnrvrehdnzr3f5m.streamlit.app

---

## What it does

- Downloads 5 years of daily price data for 49 large-cap NSE stocks
- Runs Engle-Granger cointegration tests on all 1,176 possible pairs
- Finds pairs that are statistically "tied" to each other long-term
- Generates buy/sell signals when the spread between a pair deviates beyond 2σ
- Backtests all signals on completely unseen out-of-sample data
- Displays results on an interactive Streamlit dashboard

---

## Results (out-of-sample test period: Dec 2024 — Jun 2026)

| Metric | Value |
|---|---|
| Total pairs tested | 100 |
| Profitable pairs | 68 (68%) |
| Best total return | 29.9% |
| Best hit rate | 100% |
| Best Sharpe ratio | 42.27 |

---

## Strategy Design

### Why cointegration and not just correlation?

Correlation only measures whether two stocks move in the same direction.
Cointegration is stronger — it means the spread between two stocks is
stationary and mean-reverting. Even if they drift apart short-term, they
always come back together. This is the mathematical foundation of pairs trading.

### Signal logic
Z = (spread - rolling_mean) / rolling_std   [60-day window]
Z > +2  →  Short Stock A, Buy Stock B
Z < -2  →  Buy Stock A, Short Stock B
Z →  0  →  Close the trade (mean reversion complete)
Z > +3 or Z < -3  →  Stop loss triggered

### Realistic backtest design

- Train/test split — pairs found on first 70% of data, signals on last 30%
- Transaction costs — 0.1% deducted per trade
- Stop loss — exit if z-score crosses ±3 (spread keeps diverging)
- 5 years of data — enough for statistically meaningful results

---

## Tech Stack

| Library | Purpose |
|---|---|
| yfinance | Download NSE stock price history |
| pandas | Data manipulation and time series |
| numpy | Z-score and rolling statistics |
| statsmodels | Engle-Granger cointegration test |
| streamlit | Interactive web dashboard |
| plotly | Interactive charts |

---

## Project Structure
stat-arb-detector/
├── data/
│   └── prices.csv          ← downloaded stock data
├── src/
│   ├── data_loader.py      ← data download and train/test split
│   ├── cointegration.py    ← Engle-Granger cointegration tests
│   ├── signals.py          ← hedge ratio, spread, z-score, signals
│   └── backtest.py         ← trade simulation and performance metrics
├── app.py                  ← Streamlit dashboard
└── requirements.txt        ← dependencies

---

## How to run locally

```bash
# clone the repo
git clone https://github.com/yourusername/stat-arb-detector.git
cd stat-arb-detector

# install dependencies
pip install -r requirements.txt

# download stock data (run once)
python src/data_loader.py

# launch dashboard
streamlit run app.py
```

---

## Key concepts

**Engle-Granger test** — a two-step statistical test for cointegration.
Tests whether the residuals of a regression between two price series are
stationary. A p-value below 0.05 indicates cointegration at 95% confidence.

**Z-score** — measures how many standard deviations the current spread is
from its rolling mean. Values beyond ±2 indicate statistically unusual
divergence — the core trading signal.

**Sharpe ratio** — return per unit of risk. Calculated as
mean(returns) / std(returns) × √252. Above 2.0 is considered excellent
in professional quantitative finance.

**Out-of-sample testing** — pairs identified on training data are tested
on a completely separate held-out period. This prevents overfitting and
gives a realistic estimate of real-world performance.