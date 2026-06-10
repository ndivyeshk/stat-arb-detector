# 📈 Statistical Arbitrage Signal Detector

A pairs trading system built on NSE (Indian) stocks that automatically detects
cointegrated pairs, generates z-score based trading signals, enhances them with
LSTM forecasting, and backtests the strategy on completely out-of-sample data.

Live Demo : https://stat-arb-detector-gzwbjuusnrvrehdnzr3f5m.streamlit.app/
GitHub    : https://github.com/ndivyeshk/stat-arb-detector

---

## What it does

- Downloads 5 years of daily price data for 49 large-cap NSE stocks
- Runs Engle-Granger cointegration tests on all 1,176 possible pairs
- Finds pairs that are statistically "tied" to each other long-term
- Generates buy/sell signals when the spread deviates beyond 2σ
- Trains individual LSTMs per pair to forecast z-score trajectories
- Compares raw signal vs LSTM-enhanced signal performance
- Backtests all signals on completely unseen out-of-sample data
- Displays full results on an interactive Streamlit dashboard

---

## Results (out-of-sample test period: Dec 2024 — Jun 2026)

### Raw Signal Baseline

| Metric | Value |
|---|---|
| Total pairs tested | 100 |
| Profitable pairs | 68 (68%) |
| Avg hit rate | 53.9% |
| Avg total return | 4.07% |
| Avg Sharpe ratio | 3.85 |

### LSTM Enhanced

| Metric | Value | vs Baseline |
|---|---|---|
| Profitable pairs | 60 | — |
| Avg hit rate | 62.7% | +8.8% |
| Avg total return | 4.37% | +0.3% |
| Avg Sharpe ratio | 6.55 | +2.7 |
| Pairs improved by LSTM | 49 / 100 | — |
| Best LSTM improvement | +30.88% | — |

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

### LSTM Enhancement

For each cointegrated pair, an individual LSTM is trained on the training
period z-score series. The model learns the temporal pattern of spread
behavior and predicts the next day's z-score. These predicted z-scores
replace raw z-scores in signal generation, improving average hit rate
from 53.9% to 62.7% across all 100 pairs.

Architecture: 2-layer LSTM (64 hidden units) → Dropout(0.2) → Linear
Input: 30-day z-score sequence
Output: predicted next-day z-score

### Realistic backtest design

- Train/test split — pairs found on first 70% of data, signals on last 30%
- Transaction costs — 0.1% deducted per trade (brokerage + STT)
- Stop loss — exit if z-score crosses ±3 (spread keeps diverging)
- 5 years of data — 1,237 trading days across 49 stocks

---

## Tech Stack

| Library | Purpose |
|---|---|
| yfinance | Download NSE stock price history |
| pandas | Data manipulation and time series |
| numpy | Z-score and rolling statistics |
| statsmodels | Engle-Granger cointegration test |
| torch (PyTorch) | LSTM model training and inference |
| streamlit | Interactive web dashboard |
| plotly | Interactive charts |

---

## Project Structure

stat-arb-detector/
├── data/
│   └── prices.csv              ← 5 years of NSE stock data
├── src/
│   ├── data_loader.py          ← data download and train/test split
│   ├── cointegration.py        ← Engle-Granger cointegration tests
│   ├── signals.py              ← hedge ratio, spread, z-score, signals
│   ├── backtest.py             ← trade simulation and performance metrics
│   └── lstm_model.py           ← LSTM architecture, training, prediction
├── app.py                      ← Streamlit dashboard
└── requirements.txt            ← dependencies

---

## How to run locally

```bash
# clone the repo
git clone https://github.com/ndivyeshk/stat-arb-detector.git
cd stat-arb-detector

# install dependencies
pip install -r requirements.txt

# download stock data (run once)
python src/data_loader.py

# launch dashboard
streamlit run app.py
```

Note: First load takes ~20 minutes as it trains 100 LSTMs.
After that Streamlit caches everything and loads instantly.

---

## Key concepts

**Engle-Granger test** — a two-step statistical test for cointegration.
Tests whether the residuals of a regression between two price series are
stationary. A p-value below 0.05 indicates cointegration at 95% confidence.

**Z-score** — measures how many standard deviations the current spread is
from its rolling mean. Values beyond ±2 indicate statistically unusual
divergence — the core trading signal.

**LSTM (Long Short-Term Memory)** — a recurrent neural network architecture
designed for sequential data. Used here to model the temporal dynamics of
z-score series and predict next-day spread behavior.

**Sharpe ratio** — return per unit of risk. Calculated as
mean(returns) / std(returns) × √252. Above 2.0 is considered excellent
in professional quantitative finance.

**Out-of-sample testing** — pairs identified on training data are tested
on a completely separate held-out period. This prevents overfitting and
gives a realistic estimate of real-world performance.

---

## Limitations and future work

- LSTM improved 49/100 pairs — predictability correlates with cointegration
  strength; spurious cross-sector pairs show limited improvement
- Small trade counts per pair limit statistical significance
- Future: Johansen test for multivariate cointegration, sector-filtered
  pair selection, portfolio-level position sizing

  