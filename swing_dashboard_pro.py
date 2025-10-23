# swing_dashboard_pro_resilient.py
# Streamlit Swing Dashboard — resilient rewrite
# - URL + session ticker persistence
# - Robust yfinance download (single or multiple tickers)
# - Safe caching (tuple-hashed args) and defensive schema checks
# - Common indicators (SMA, EMA, RSI, MACD, Bollinger)
# - Optional Plotly charts (fallback to st.line_chart)
# - Simple “bull flag” heuristic scanner

import os
import math
import datetime as dt
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np

# Optional libraries
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None

# ----------------------------- Utilities -----------------------------

def _normalize_tickers(raw) -> List[str]:
    """Accepts a comma string or list and returns a de-duped, uppercased list."""
    if isinstance(raw, str):
        raw = raw.split(",")
    ticks = [str(t).strip().upper() for t in (raw or []) if str(t).strip()]
    seen, out = set(), []
    for t in ticks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _get_query_params():
    return st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()


def _set_query_params(**kwargs):
    if hasattr(st, "query_params"):
        for k, v in kwargs.items():
            st.query_params[k] = v
    else:
        st.experimental_set_query_params(**kwargs)


# ----------------------------- Indicator functions -----------------------------

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = (d.where(d > 0, 0.0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close: pd.Series, n: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return upper, ma, lower


# ----------------------------- Data loading -----------------------------

@st.cache_data(show_spinner=False)
def fetch_prices(tickers: Tuple[str, ...], start: Optional[dt.date], end: Optional[dt.date], interval: str) -> pd.DataFrame:
    """Download prices via yfinance and return a long DataFrame with columns:
    Date, Ticker, Open, High, Low, Close, Volume
    """
    if yf is None:
        raise RuntimeError("yfinance not installed. Add 'yfinance' to requirements.txt")

    if not tickers:
        return pd.DataFrame()

    # yfinance accepts list/space-separated string
    dl = yf.download(list(tickers), start=start, end=end, interval=interval,
                     group_by="ticker", auto_adjust=False, threads=False, progress=False)

    if dl is None or getattr(dl, "empty", True):
        return pd.DataFrame()

    # If MultiIndex columns (multiple tickers), reshape to long
    if isinstance(dl.columns, pd.MultiIndex):
        # columns like ('AAPL','Open') ... we want rows per (Date, Ticker)
        long_df = (
            dl.stack(level=0)  # ticker level to rows
              .rename_axis(["Date", "Ticker"])  # index names
              .reset_index()
        )
    else:
        # Single ticker frame: Date is index, OHLCV are columns
        long_df = dl.reset_index()
        # If only one ticker, ensure a Ticker column
        t0 = tickers[0] if tickers else "TICKER"
        long_df["Ticker"] = t0

    # Standardize column names
    long_df = long_df.rename(columns={"Adj Close": "Close", "Date": "Date"})

    # Defensive: ensure required columns exist
    required = {"Date", "Open", "High", "Low", "Close", "Volume", "Ticker"}
    missing = required.difference(long_df.columns)
    if missing:
        raise KeyError(f"Missing columns after download/reshape: {sorted(missing)}")

    long_df["Date"] = pd.to_datetime(long_df["Date"], errors="coerce")
    long_df = long_df.dropna(subset=["Date"]).copy()
    return long_df


def compute_indicators(g: pd.DataFrame) -> pd.DataFrame:
    # Ensure Date column exists (in case of grouping shenanigans)
    if "Date" not in g.columns:
        g = g.reset_index()
    g = g.sort_values("Date").copy()

    # Indicators
    g["SMA20"] = sma(g["Close"], 20)
    g["SMA50"] = sma(g["Close"], 50)
    g["EMA12"] = ema(g["Close"], 12)
    g["EMA26"] = ema(g["Close"], 26)
    g["RSI14"] = rsi_wilder(g["Close"], 14)
    m, s, h = macd(g["Close"])  # MACD
    g["MACD"], g["MACDsig"], g["MACDhist"] = m, s, h
    bb_u, bb_m, bb_l = bollinger(g["Close"])  # Bollinger
    g["BBU"], g["BBM"], g["BBL"] = bb_u, bb_m, bb_l

    # Daily returns
    g["Ret"] = g["Close"].pct_change()
    return g


def bull_flag_score(df: pd.DataFrame, lookback: int = 25) -> float:
    """Very simple bull-flag heuristic: strong run-up then a tight, downward-sloping consolidation.
    Returns a score 0..1; higher ~ more flag-like. Not financial advice.
    """
    if len(df) < lookback + 5:
        return 0.0
    sub = df.tail(lookback).copy()
    # 1) prior momentum: price above SMA20 and SMA20 above SMA50
    mom = float((sub["Close"].iloc[-1] > sub["SMA20"].iloc[-1] > sub["SMA50"].iloc[-1]))
    # 2) consolidation: last 5 bars ATR-like width vs prior move
    rng = (sub["High"].max() - sub["Low"].min())
    recent = sub.tail(5)
    cons_width = (recent["High"] - recent["Low"]).mean()
    tight = 1.0 - float(cons_width / rng) if rng > 0 else 0.0
    tight = max(0.0, min(1.0, tight))
    # 3) mild downward slope in last 5 closes
    slope = np.polyfit(range(5), recent["Close"].to_numpy(), 1)[0]
    down = 1.0 if slope < 0 else 0.0
    # weighted combo
    score = 0.45 * mom + 0.35 * tight + 0.20 * down
    return round(float(max(0.0, min(1.0, score))), 3)


# ----------------------------- UI: Init State + Sidebar -----------------------------

st.set_page_config(page_title="Swing Dashboard", layout="wide")

# Initialize tickers from URL once per session
if "tickers" not in st.session_state:
    qp = _get_query_params()
    initial = qp.get("tickers", [])
    if isinstance(initial, list) and len(initial) == 1:
        initial = initial[0]
    st.session_state.tickers = _normalize_tickers(initial)

# Date defaults
if "date_start" not in st.session_state:
    st.session_state.date_start = dt.date.today() - dt.timedelta(days=365)
if "date_end" not in st.session_state:
    st.session_state.date_end = dt.date.today()

# Sidebar controls
with st.sidebar:
    st.title("⚡ Swing Dashboard")
    st.caption("URL & session-persistent watchlist")

    st.subheader("Watchlist")
    # Show current list
    st.write(", ".join(st.session_state.tickers) or "—")

    # Freeform input (comma-separated)
    if "tickers_input" not in st.session_state:
        st.session_state.tickers_input = ",".join(st.session_state.tickers)

    def _persist_tickers_to_url():
        csv = ",".join(st.session_state.tickers)
        _set_query_params(tickers=csv)

    def _on_input_change():
        new_list = _normalize_tickers(st.session_state.tickers_input)
        st.session_state.tickers = new_list
        _persist_tickers_to_url()

    st.text_input("Tickers (comma-separated)", key="tickers_input", placeholder="AAPL, MSFT, NVDA", on_change=_on_input_change)

    with st.form("ticker_add_remove", clear_on_submit=True):
        new_tick = st.text_input("Add one ticker", key="new_tick")
        to_remove = st.multiselect("Remove selected", st.session_state.tickers, key="to_remove")
        c1, c2 = st.columns(2)
        do_add = c1.form_submit_button("Add")
        do_remove = c2.form_submit_button("Remove")

    if do_add and new_tick:
        t = new_tick.strip().upper()
        if t and t not in st.session_state.tickers:
            st.session_state.tickers.append(t)
            st.session_state.tickers_input = ",".join(st.session_state.tickers)
            _persist_tickers_to_url()
        elif t:
            st.info(f"{t} is already on the list.")

    if do_remove and st.session_state.to_remove:
        st.session_state.tickers = [t for t in st.session_state.tickers if t not in st.session_state.to_remove]
        st.session_state.tickers_input = ",".join(st.session_state.tickers)
        _persist_tickers_to_url()

    st.divider()
    st.subheader("Data Settings")
    st.session_state.date_start = st.date_input("Start", value=st.session_state.date_start)
    st.session_state.date_end = st.date_input("End", value=st.session_state.date_end)
    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)

    st.divider()
    st.subheader("Indicators")
    show_ma = st.checkbox("SMA20 / SMA50", value=True)
    show_bb = st.checkbox("Bollinger Bands", value=False)
    show_macd = st.checkbox("MACD (below)", value=True)
    show_rsi = st.checkbox("RSI (below)", value=True)


# ----------------------------- Main: Data + Views -----------------------------

st.title("📈 Swing Trading Dashboard — Resilient")

# Fetch data
with st.status("Downloading & computing…", expanded=False) as status:
    tickers = st.session_state.tickers
    if not tickers:
        st.info("Add tickers in the sidebar to begin.")
        st.stop()

    try:
        df = fetch_prices(tuple(tickers), st.session_state.date_start, st.session_state.date_end, interval)
    except Exception as e:
        st.error(f"Data download failed: {e}")
        st.stop()

    if df.empty:
        st.warning("No data returned. Check tickers/date range/interval.")
        st.stop()

    try:
        df = df.groupby("Ticker", group_keys=False, sort=False).apply(compute_indicators)
    except KeyError as ke:
        st.error(f"Indicator computation failed: {ke}")
        st.write("Columns present:", list(df.columns))
        st.stop()

    status.update(label="Ready", state="complete")

# ----------------------------- Screener: Bull-flag heuristic -----------------------------

st.subheader("📌 Quick Screener — Bull Flag Heuristic (0..1)")

scores = (
    df.groupby("Ticker", group_keys=False)
      .apply(lambda g: bull_flag_score(g))
      .reset_index(name="FlagScore")
      .sort_values("FlagScore", ascending=False)
)

st.dataframe(scores, use_container_width=True)

# ----------------------------- Charts -----------------------------

selected = st.multiselect("Choose tickers to chart", tickers, default=tickers[:1])

for t in selected:
    g = df[df["Ticker"] == t].copy()
    st.markdown(f"### {t}")

    # Price chart (Plotly if available)
    if go is not None:
        fig = go.Figure()
        fig.add_candlestick(x=g["Date"], open=g["Open"], high=g["High"], low=g["Low"], close=g["Close"], name="OHLC")
        if show_ma:
            fig.add_trace(go.Scatter(x=g["Date"], y=g["SMA20"], mode="lines", name="SMA20"))
            fig.add_trace(go.Scatter(x=g["Date"], y=g["SMA50"], mode="lines", name="SMA50"))
        if show_bb:
            fig.add_trace(go.Scatter(x=g["Date"], y=g["BBU"], mode="lines", name="BB Upper"))
            fig.add_trace(go.Scatter(x=g["Date"], y=g["BBM"], mode="lines", name="BB Mid"))
            fig.add_trace(go.Scatter(x=g["Date"], y=g["BBL"], mode="lines", name="BB Lower"))
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(g.set_index("Date")["Close"], height=300)
        if show_ma:
            st.line_chart(g.set_index("Date")["SMA20"].dropna(), height=150)
            st.line_chart(g.set_index("Date")["SMA50"].dropna(), height=150)
        if show_bb:
            st.line_chart(g.set_index("Date")[["BBU", "BBM", "BBL"]].dropna(), height=200)

    # MACD / RSI panels
    if show_macd:
        if go is not None:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=g["Date"], y=g["MACD"], name="MACD"))
            fig2.add_trace(go.Scatter(x=g["Date"], y=g["MACDsig"], name="Signal"))
            fig2.add_trace(go.Bar(x=g["Date"], y=g["MACDhist"], name="Hist"))
            fig2.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.line_chart(g.set_index("Date")["MACD"].dropna(), height=150)

    if show_rsi:
        if go is not None:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=g["Date"], y=g["RSI14"], name="RSI14"))
            fig3.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.line_chart(g.set_index("Date")["RSI14"].dropna(), height=150)

st.caption("Not financial advice. Educational purposes only.")
