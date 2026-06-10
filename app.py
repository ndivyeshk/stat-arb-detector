# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_loader import load_prices, split_prices
from src.cointegration import find_cointegrated_pairs
from src.signals import get_pair_data, generate_signals
from src.backtest import run_backtest, run_lstm_backtest
from src.lstm_model import train_lstm, predict_zscore

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Statistical Arbitrage Signal Detector",
    page_icon  = "📈",
    layout     = "wide",
)

# ── Cache heavy computations ───────────────────────────────────────────────
@st.cache_data
def load_all_data():
    prices      = load_prices()
    train, test = split_prices(prices)
    return prices, train, test


@st.cache_data
def get_all_pairs(_train_data):
    return find_cointegrated_pairs(_train_data)


@st.cache_data
def get_all_results(_pairs_df, _prices, _train, _test):
    raw_summaries  = []
    lstm_summaries = []
    all_trades     = {}

    total = len(_pairs_df)

    for i in range(total):
        row     = _pairs_df.iloc[i]
        stock_a = row["stock_a"]
        stock_b = row["stock_b"]
        pair_key = f"{stock_a} / {stock_b}"

        # raw backtest
        pair_data, _   = get_pair_data(_test, stock_a, stock_b)
        raw_summary, raw_trades = run_backtest(pair_data, stock_a, stock_b)

        # lstm backtest
        lstm_summary, lstm_trades = run_lstm_backtest(_prices, _train, _test, stock_a, stock_b)

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

    raw_df  = pd.DataFrame(raw_summaries).sort_values("total_return_pct", ascending=False).reset_index(drop=True)
    lstm_df = pd.DataFrame(lstm_summaries).sort_values("total_return_pct", ascending=False).reset_index(drop=True)

    # merged comparison
    merged = raw_df.merge(lstm_df, on="pair", suffixes=("_raw", "_lstm"))
    merged["improvement"] = merged["total_return_pct_lstm"] - merged["total_return_pct_raw"]
    merged["lstm_improved"] = merged["improvement"] > 0

    return raw_df, lstm_df, merged, all_trades


# ── Load everything ────────────────────────────────────────────────────────
with st.spinner("Loading data and running backtests... first load takes ~20 minutes, cached after that"):
    prices, train, test         = load_all_data()
    pairs                       = get_all_pairs(train)
    raw_df, lstm_df, merged, all_trades = get_all_results(pairs, prices, train, test)

profitable_raw  = raw_df[raw_df["total_return_pct"]  > 0]
profitable_lstm = lstm_df[lstm_df["total_return_pct"] > 0]

# ── Header ─────────────────────────────────────────────────────────────────
st.title("📈 Statistical Arbitrage Signal Detector")
st.markdown("Pairs trading strategy on NSE stocks — cointegration based signal detection with LSTM z-score forecasting and out-of-sample backtesting")
st.divider()

# ── Section 1 — Overall summary ────────────────────────────────────────────
st.subheader("📊 Overall Results")

tab1, tab2 = st.tabs(["Raw Signal", "LSTM Enhanced"])

with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pairs Tested",      len(raw_df))
    c2.metric("Profitable Pairs",  len(profitable_raw))
    c3.metric("Avg Hit Rate",      f"{round(raw_df['hit_rate_pct'].mean(), 1)}%")
    c4.metric("Avg Total Return",  f"{round(raw_df['total_return_pct'].mean(), 2)}%")
    c5.metric("Avg Sharpe Ratio",  round(raw_df['sharpe_ratio'].mean(), 2))

with tab2:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pairs Tested",      len(lstm_df))
    c2.metric("Profitable Pairs",  len(profitable_lstm))
    c3.metric("Avg Hit Rate",      f"{round(lstm_df['hit_rate_pct'].mean(), 1)}%",
              delta=f"+{round(lstm_df['hit_rate_pct'].mean() - raw_df['hit_rate_pct'].mean(), 1)}%")
    c4.metric("Avg Total Return",  f"{round(lstm_df['total_return_pct'].mean(), 2)}%",
              delta=f"+{round(lstm_df['total_return_pct'].mean() - raw_df['total_return_pct'].mean(), 2)}%")
    c5.metric("Avg Sharpe Ratio",  round(lstm_df['sharpe_ratio'].mean(), 2),
              delta=round(lstm_df['sharpe_ratio'].mean() - raw_df['sharpe_ratio'].mean(), 2))

st.divider()

# ── Section 2 — LSTM improvement summary ──────────────────────────────────
st.subheader("🤖 LSTM Impact Summary")

col1, col2, col3 = st.columns(3)
improved     = merged[merged["lstm_improved"] == True]
not_improved = merged[merged["lstm_improved"] == False]

col1.metric("Pairs Improved by LSTM",     f"{len(improved)} / {len(merged)}")
col2.metric("Best LSTM Improvement",      f"+{round(improved['improvement'].max(), 2)}%")
col3.metric("Avg Improvement on Winners", f"+{round(improved['improvement'].mean(), 2)}%")

st.divider()

# ── Section 3 — Pairs tables ───────────────────────────────────────────────
st.subheader("🏆 Pair Rankings")

tab3, tab4, tab5 = st.tabs(["Top Raw Pairs", "Top LSTM Pairs", "LSTM vs Raw Comparison"])

with tab3:
    display = profitable_raw.rename(columns={
        "pair": "Pair", "total_trades": "Trades",
        "hit_rate_pct": "Hit Rate %", "avg_return_pct": "Avg Return %",
        "total_return_pct": "Total Return %", "avg_holding_days": "Avg Holding Days",
        "sharpe_ratio": "Sharpe Ratio"
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

with tab4:
    display = profitable_lstm.rename(columns={
        "pair": "Pair", "total_trades": "Trades",
        "hit_rate_pct": "Hit Rate %", "avg_return_pct": "Avg Return %",
        "total_return_pct": "Total Return %", "avg_holding_days": "Avg Holding Days",
        "sharpe_ratio": "Sharpe Ratio"
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

with tab5:
    compare = merged[[
        "pair", "total_return_pct_raw", "total_return_pct_lstm",
        "improvement", "hit_rate_pct_raw", "hit_rate_pct_lstm",
        "sharpe_ratio_raw", "sharpe_ratio_lstm", "lstm_improved"
    ]].rename(columns={
        "pair": "Pair",
        "total_return_pct_raw": "Raw Return %",
        "total_return_pct_lstm": "LSTM Return %",
        "improvement": "Improvement %",
        "hit_rate_pct_raw": "Raw Hit Rate %",
        "hit_rate_pct_lstm": "LSTM Hit Rate %",
        "sharpe_ratio_raw": "Raw Sharpe",
        "sharpe_ratio_lstm": "LSTM Sharpe",
        "lstm_improved": "LSTM Helped",
    })
    st.dataframe(compare, use_container_width=True, hide_index=True)

st.divider()

# ── Section 4 — Pair deep dive ─────────────────────────────────────────────
st.subheader("🔍 Pair Deep Dive")

all_pair_names = merged["pair"].tolist()
selected_pair  = st.selectbox("Select a pair to analyse", options=all_pair_names)

if selected_pair in all_trades:
    pair_info  = all_trades[selected_pair]
    pair_data  = pair_info["pair_data"]
    raw_trades = pair_info["raw_trades"]
    lstm_trades= pair_info["lstm_trades"]

    pair_row   = merged[merged["pair"] == selected_pair].iloc[0]

    # metrics
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Raw Return %",   f"{pair_row['total_return_pct_raw']}%")
    m2.metric("LSTM Return %",  f"{pair_row['total_return_pct_lstm']}%",
              delta=f"{round(pair_row['improvement'], 2)}%")
    m3.metric("Raw Hit Rate",   f"{pair_row['hit_rate_pct_raw']}%")
    m4.metric("LSTM Hit Rate",  f"{pair_row['hit_rate_pct_lstm']}%")
    m5.metric("Raw Sharpe",     pair_row["sharpe_ratio_raw"])
    m6.metric("LSTM Sharpe",    pair_row["sharpe_ratio_lstm"])

    stock_a, stock_b = selected_pair.split(" / ")

    # ── Chart 1 — Normalised prices ────────────────────────────────────────
    price_a_norm = (pair_data["price_a"] / pair_data["price_a"].iloc[0]) * 100
    price_b_norm = (pair_data["price_b"] / pair_data["price_b"].iloc[0]) * 100

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=pair_data.index, y=price_a_norm,
        name=stock_a, line=dict(color="#00b4d8", width=2)))
    fig1.add_trace(go.Scatter(x=pair_data.index, y=price_b_norm,
        name=stock_b, line=dict(color="#f77f00", width=2)))
    fig1.update_layout(title="Normalised Price Movement (base = 100)",
        xaxis_title="Date", yaxis_title="Normalised Price",
        hovermode="x unified", height=350)
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2 — Z-score with signals ────────────────────────────────────
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=pair_data.index, y=pair_data["zscore"],
        name="Z-score", line=dict(color="#7b2d8b", width=2)))
    for level, color, label in [(2, "red", "+2σ"), (-2, "green", "-2σ"), (0, "gray", "Mean")]:
        fig2.add_hline(y=level, line_dash="dash", line_color=color,
            opacity=0.6, annotation_text=label)
    if raw_trades is not None and len(raw_trades) > 0:
        buys   = raw_trades[raw_trades["side"] == "Buy A"]
        shorts = raw_trades[raw_trades["side"] == "Short A"]
        fig2.add_trace(go.Scatter(x=buys["entry_date"],
            y=[-2.1]*len(buys), mode="markers", name="Buy Signal",
            marker=dict(symbol="triangle-up", size=12, color="green")))
        fig2.add_trace(go.Scatter(x=shorts["entry_date"],
            y=[2.1]*len(shorts), mode="markers", name="Short Signal",
            marker=dict(symbol="triangle-down", size=12, color="red")))
    fig2.update_layout(title="Z-score with Raw Signals",
        xaxis_title="Date", yaxis_title="Z-score",
        hovermode="x unified", height=350)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3 — Cumulative return comparison ─────────────────────────────
    fig3 = go.Figure()
    if raw_trades is not None and len(raw_trades) > 0:
        raw_sorted = raw_trades.sort_values("entry_date")
        raw_sorted["cumulative"] = raw_sorted["return_pct"].cumsum()
        fig3.add_trace(go.Scatter(x=raw_sorted["exit_date"],
            y=raw_sorted["cumulative"], name="Raw Signal",
            line=dict(color="#00b4d8", width=2)))
    if lstm_trades is not None and len(lstm_trades) > 0:
        lstm_sorted = lstm_trades.sort_values("entry_date")
        lstm_sorted["cumulative"] = lstm_sorted["return_pct"].cumsum()
        fig3.add_trace(go.Scatter(x=lstm_sorted["exit_date"],
            y=lstm_sorted["cumulative"], name="LSTM Signal",
            line=dict(color="#2dc653", width=2)))
    fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig3.update_layout(title="Cumulative Return — Raw vs LSTM",
        xaxis_title="Date", yaxis_title="Cumulative Return %",
        hovermode="x unified", height=350)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Trade logs ─────────────────────────────────────────────────────────
    st.subheader("📋 Trade Logs")
    tl1, tl2 = st.tabs(["Raw Trades", "LSTM Trades"])

    def format_trades(trades):
        if trades is None or len(trades) == 0:
            return pd.DataFrame()
        t = trades.copy()
        t["win"]       = t["win"].map({True: "✅ Win", False: "❌ Loss"})
        t["stop_loss"] = t["stop_loss"].map({True: "🛑 Yes", False: "No"})
        return t.rename(columns={
            "entry_date": "Entry Date", "exit_date": "Exit Date",
            "side": "Side", "return_pct": "Return %",
            "holding_days": "Holding Days", "win": "Result",
            "stop_loss": "Stop Loss Hit"
        })

    with tl1:
        st.dataframe(format_trades(raw_trades), use_container_width=True, hide_index=True)
    with tl2:
        st.dataframe(format_trades(lstm_trades), use_container_width=True, hide_index=True)