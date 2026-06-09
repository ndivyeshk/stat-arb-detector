# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import load_prices, split_prices
from src.cointegration import find_cointegrated_pairs
from src.signals import get_pair_data
from src.backtest import run_backtest

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Statistical Arbitrage Signal Detector",
    page_icon  = "📈",
    layout     = "wide",
)

# ── Cache heavy computations ───────────────────────────────────────────────
# @st.cache_data tells Streamlit — run this once, cache the result
# so every time the user clicks something, we don't re-download data

@st.cache_data
def load_all_data():
    prices      = load_prices()
    train, test = split_prices(prices)
    return prices, train, test

@st.cache_data
def get_all_pairs(train_data):
    return find_cointegrated_pairs(train_data)

@st.cache_data
def get_all_results(pairs_df, test_data):
    all_summaries = []
    all_trades    = {}

    for i in range(len(pairs_df)):
        row     = pairs_df.iloc[i]
        stock_a = row["stock_a"]
        stock_b = row["stock_b"]

        pair_data, hedge_ratio = get_pair_data(test_data, stock_a, stock_b)
        summary, trades        = run_backtest(pair_data, stock_a, stock_b)

        if summary:
            all_summaries.append(summary)
            pair_key = f"{stock_a} / {stock_b}"
            all_trades[pair_key] = {
                "pair_data"    : pair_data,
                "trades"       : trades,
                "hedge_ratio"  : hedge_ratio,
            }

    results_df = pd.DataFrame(all_summaries)
    results_df = results_df.sort_values("total_return_pct", ascending=False).reset_index(drop=True)
    return results_df, all_trades


# ── Load everything ────────────────────────────────────────────────────────
with st.spinner("Loading data and running backtests... this takes ~60 seconds on first load"):
    prices, train, test = load_all_data()
    pairs               = get_all_pairs(train)
    results_df, all_trades = get_all_results(pairs, test)

profitable  = results_df[results_df["total_return_pct"] > 0].reset_index(drop=True)
losing      = results_df[results_df["total_return_pct"] <= 0].reset_index(drop=True)

# ── Section 1 — Header ─────────────────────────────────────────────────────
st.title("📈 Statistical Arbitrage Signal Detector")
st.markdown("Pairs trading strategy on NSE stocks — cointegration based signal detection with out-of-sample backtesting")
st.divider()

# ── Section 2 — Summary stats ──────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Pairs Tested",  len(results_df))
col2.metric("Profitable Pairs",    len(profitable))
col3.metric("Loss Making Pairs",   len(losing))
col4.metric("Best Hit Rate",       f"{profitable['hit_rate_pct'].max()}%")
col5.metric("Best Total Return",   f"{profitable['total_return_pct'].max():.1f}%")

st.divider()

# ── Section 3 — Results table ──────────────────────────────────────────────
st.subheader("🏆 All Profitable Pairs — Ranked by Total Return")

# clean up column names for display
display_df = profitable.rename(columns={
    "pair"              : "Pair",
    "total_trades"      : "Trades",
    "hit_rate_pct"      : "Hit Rate %",
    "avg_return_pct"    : "Avg Return %",
    "total_return_pct"  : "Total Return %",
    "avg_holding_days"  : "Avg Holding Days",
    "sharpe_ratio"      : "Sharpe Ratio",
})

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()

# ── Section 4 — Pair deep dive ─────────────────────────────────────────────
st.subheader("🔍 Pair Deep Dive")

selected_pair = st.selectbox(
    "Select a pair to analyse",
    options = profitable["pair"].tolist(),
)

if selected_pair in all_trades:
    pair_info  = all_trades[selected_pair]
    pair_data  = pair_info["pair_data"]
    trades     = pair_info["trades"]
    hedge_ratio = pair_info["hedge_ratio"]

    # metrics for selected pair
    pair_summary = profitable[profitable["pair"] == selected_pair].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Trades",    pair_summary["total_trades"])
    m2.metric("Hit Rate",        f"{pair_summary['hit_rate_pct']}%")
    m3.metric("Total Return",    f"{pair_summary['total_return_pct']}%")
    m4.metric("Sharpe Ratio",    pair_summary["sharpe_ratio"])

    stock_a, stock_b = selected_pair.split(" / ")

    # ── Chart 1 — Normalised price chart ──────────────────────────────────
    # we normalise both prices to 100 at the start so they're comparable
    price_a_norm = (pair_data["price_a"] / pair_data["price_a"].iloc[0]) * 100
    price_b_norm = (pair_data["price_b"] / pair_data["price_b"].iloc[0]) * 100

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=pair_data.index, y=price_a_norm,
        name=stock_a, line=dict(color="#00b4d8", width=2)
    ))
    fig1.add_trace(go.Scatter(
        x=pair_data.index, y=price_b_norm,
        name=stock_b, line=dict(color="#f77f00", width=2)
    ))
    fig1.update_layout(
        title  = "Normalised Price Movement (base = 100)",
        xaxis_title = "Date",
        yaxis_title = "Normalised Price",
        hovermode   = "x unified",
        height      = 350,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2 — Z-score chart ────────────────────────────────────────────
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=pair_data.index, y=pair_data["zscore"],
        name="Z-score", line=dict(color="#7b2d8b", width=2)
    ))

    # threshold lines
    for level, color, label in [(2, "red", "Entry +2σ"), (-2, "green", "Entry -2σ"), (0, "gray", "Mean")]:
        fig2.add_hline(
            y=level, line_dash="dash",
            line_color=color, opacity=0.6,
            annotation_text=label,
        )

    # mark entry points on z-score chart
    if trades is not None and len(trades) > 0:
        entries = trades[trades["side"] == "Buy A"]
        fig2.add_trace(go.Scatter(
            x=entries["entry_date"], y=[-2.05] * len(entries),
            mode="markers", name="Buy Signal",
            marker=dict(symbol="triangle-up", size=12, color="green")
        ))
        shorts = trades[trades["side"] == "Short A"]
        fig2.add_trace(go.Scatter(
            x=shorts["entry_date"], y=[2.05] * len(shorts),
            mode="markers", name="Short Signal",
            marker=dict(symbol="triangle-down", size=12, color="red")
        ))

    fig2.update_layout(
        title       = "Z-score Over Time with Entry Signals",
        xaxis_title = "Date",
        yaxis_title = "Z-score",
        hovermode   = "x unified",
        height      = 350,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3 — Cumulative return ────────────────────────────────────────
    if trades is not None and len(trades) > 0:
        trades_sorted = trades.sort_values("entry_date")
        trades_sorted["cumulative_return"] = trades_sorted["return_pct"].cumsum()

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=trades_sorted["exit_date"],
            y=trades_sorted["cumulative_return"],
            name="Cumulative Return %",
            line=dict(color="#2dc653", width=2),
            fill="tozeroy",
            fillcolor="rgba(45, 198, 83, 0.1)"
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig3.update_layout(
            title       = "Cumulative Return Over Test Period",
            xaxis_title = "Date",
            yaxis_title = "Cumulative Return %",
            hovermode   = "x unified",
            height      = 350,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Trade log ──────────────────────────────────────────────────────────
    st.subheader("📋 Trade Log")
    if trades is not None and len(trades) > 0:
        trades_display = trades.copy()
        trades_display["win"] = trades_display["win"].map({True: "✅ Win", False: "❌ Loss"})
        trades_display["stop_loss"] = trades_display["stop_loss"].map({True: "🛑 Yes", False: "No"})
        trades_display = trades_display.rename(columns={
            "entry_date"   : "Entry Date",
            "exit_date"    : "Exit Date",
            "side"         : "Side",
            "return_pct"   : "Return %",
            "holding_days" : "Holding Days",
            "win"          : "Result",
            "stop_loss"    : "Stop Loss Hit",
        })
        st.dataframe(trades_display, use_container_width=True, hide_index=True)