# swing_dashboard_simple.py
# Simple Streamlit swing dashboard with resilient data fetch & clear BUY/HOLD/SELL alerts
# - Daily data via yfinance (robust: period fallback, per‑ticker retry, repair)
# - Indicators: SMA20/50, RSI14, MACD, Bollinger width
# - Flags: Breakout (HH20 + volume), Squeeze (narrow bands), GoldenCross
# - Snapshot table + one candlestick chart (first ticker)
# No URL syncing, no email, no backtest — focused and sturdy.

import datetime as dt
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# Optional deps
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

# ---------------- Utilities ----------------

def _normalize_tickers(raw: str) -> List[str]:
    ticks = [t.strip().upper() for t in raw.split(",") if t.strip()]
    seen, out = set(), []
    for t in ticks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

# Indicators

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = (d.where(d > 0, 0.0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, width

# ---------------- Resilient Data Fetch ----------------

@st.cache_data(show_spinner=False)
def fetch_prices(tickers: Tuple[str, ...], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Robust daily OHLCV fetch.
    Strategy:
    - Use yf.download for all tickers with repair & no-raise
    - Prefer `period=1y` when the range is ~1y (sidestep tz lookup issues)
    - If bulk fetch empty: fallback to per‑ticker Ticker().history() with retry
    Returns a long DF: Date, Open, High, Low, Close, Volume, Ticker
    """
    if yf is None:
        raise RuntimeError("yfinance not installed. Add yfinance to requirements.txt")
    if not tickers:
        return pd.DataFrame()

    days = (end - start).days
    use_period = days in (364, 365, 366)
    period = "1y"

    def _bulk_download():
        if use_period:
            return yf.download(
                list(tickers),
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=False,
                progress=False,
                raise_errors=False,
                repair=True,
            )
        else:
            return yf.download(
                list(tickers),
                start=start,
                end=end,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=False,
                progress=False,
                raise_errors=False,
                repair=True,
            )

    dl = _bulk_download()

    def _to_long(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            out = df.stack(level=0).rename_axis(["Date", "Ticker"]).reset_index()
        else:
            out = df.reset_index()
            out["Ticker"] = tickers[0]
        return out

    if dl is None or getattr(dl, "empty", True):
        # Per‑ticker fallback with tiny retry
        frames = []
        for t in tickers:
            for _ in range(2):
                try:
                    if use_period:
                        hist = yf.Ticker(t).history(period=period, interval="1d", auto_adjust=False, repair=True)
                    else:
                        hist = yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=False, repair=True)
                    if hist is not None and not hist.empty:
                        hist = hist.reset_index().rename(columns={"Date": "Date"})
                        hist["Ticker"] = t
                        frames.append(hist[["Date","Open","High","Low","Close","Volume","Ticker"]])
                        break
                except Exception:
                    pass
        if not frames:
            return pd.DataFrame()
        long_df = pd.concat(frames, ignore_index=True)
    else:
        long_df = _to_long(dl)

    long_df = long_df.rename(columns={"Adj Close": "Close"})
    long_df["Date"] = pd.to_datetime(long_df["Date"], errors="coerce")
    required = {"Date","Open","High","Low","Close","Volume","Ticker"}
    miss = required.difference(long_df.columns)
    if miss:
        raise KeyError(f"Missing columns after download: {sorted(miss)}")
    return long_df.dropna(subset=["Date"]).copy()

# ---------------- Indicators + Signals ----------------

def add_indicators_and_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"]).copy()

    # Core indicators per ticker
    df["SMA20"] = df.groupby("Ticker", group_keys=False)["Close"].apply(lambda s: sma(s, 20))
    df["SMA50"] = df.groupby("Ticker", group_keys=False)["Close"].apply(lambda s: sma(s, 50))
    df["RSI14"] = df.groupby("Ticker", group_keys=False)["Close"].apply(lambda s: rsi_wilder(s, 14))

    # MACD
    def _macd_group(s: pd.Series):
        m, sig, h = macd(s)
        return pd.DataFrame({"MACD": m, "MACDsig": sig, "MACDhist": h})
    macd_df = df.groupby("Ticker", group_keys=False)["Close"].apply(_macd_group)
    df = pd.concat([df.reset_index(drop=True), macd_df.reset_index(drop=True)], axis=1)

    # Bollinger + width and a simple 'squeeze' vs 20-day avg width
    def _bb_group(s: pd.Series):
        u, m, l, w = bollinger(s)
        return pd.DataFrame({"BBU": u, "BBM": m, "BBL": l, "BBw": w})
    bb_df = df.groupby("Ticker", group_keys=False)["Close"].apply(_bb_group)
    df = pd.concat([df.reset_index(drop=True), bb_df.reset_index(drop=True)], axis=1)

    df["BBwMA20"] = df.groupby("Ticker", group_keys=False)["BBw"].apply(lambda s: sma(s, 20))

    # Flags per ticker
    df["VolSMA20"] = df.groupby("Ticker", group_keys=False)["Volume"].apply(lambda s: sma(s, 20))
    df["HH20"] = df.groupby("Ticker", group_keys=False)["High"].apply(lambda s: s.rolling(20, min_periods=20).max())
    df["breakout"] = (df["Close"] > df["HH20"]) & (df["Volume"] > df["VolSMA20"]) & df["HH20"].notna()
    df["squeeze"] = (df["BBw"] < (df["BBwMA20"] * 0.66)) & df["BBwMA20"].notna()
    df["golden_cross"] = (df["SMA20"].shift(1) <= df["SMA50"].shift(1)) & (df["SMA20"] > df["SMA50"]) 

    # Simple signal rules
    trend_up = df["SMA20"] > df["SMA50"]
    momentum_up = df["MACD"] > df["MACDsig"]

    buy_cond = (trend_up & momentum_up & (df["RSI14"] > 55)) | df["breakout"] | df["golden_cross"]
    sell_cond = (df["Close"] < df["SMA20"]) | ((df["MACD"] < df["MACDsig"]) & (df["RSI14"] < 45))
    hold_cond = (trend_up & (df["Close"] >= df["SMA20"]) & ~buy_cond & ~sell_cond)

    df["Signal"] = np.select(
        [sell_cond, buy_cond, hold_cond],
        ["SELL", "BUY", "HOLD"],
        default="NEUTRAL",
    )

    # Human-friendly flags column
    flags = []
    for bo, sq, gc in zip(df["breakout"].fillna(False), df["squeeze"].fillna(False), df["golden_cross"].fillna(False)):
        tag = []
        if bo:
            tag.append("🚩Breakout")
        if sq:
            tag.append("🎯Squeeze")
        if gc:
            tag.append("✨GoldenCross")
        flags.append(", ".join(tag))
    df["Flags"] = flags

    return df

# ---------------- UI ----------------

st.set_page_config(page_title="Simple Swing Dashboard", layout="wide")
st.title("🪄 Simple Swing Dashboard — Alerts (Resilient)")

with st.sidebar:
    st.subheader("Watchlist")
    raw = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,NVDA")
    tickers = _normalize_tickers(raw)

    st.subheader("Date range")
    start = st.date_input("Start", value=dt.date.today() - dt.timedelta(days=365))
    end = st.date_input("End", value=dt.date.today())

if not tickers:
    st.info("Add at least one ticker to begin.")
    st.stop()

try:
    df = fetch_prices(tuple(tickers), start, end)
except Exception as e:
    st.error(f"Download failed: {e}")
    st.stop()

if df.empty:
    st.warning("No data returned — try different dates or tickers.")
    st.stop()

try:
    df = add_indicators_and_signals(df)
except Exception as e:
    st.error(f"Indicator calc failed: {e}")
    st.write("Columns present:", list(df.columns))
    st.stop()

# Latest snapshot table with alerts
latest_date = df["Date"].max()
latest = df[df["Date"] == latest_date]

sig_icon = {
    "BUY": "🟢 BUY",
    "HOLD": "🟡 HOLD",
    "SELL": "🔴 SELL",
    "NEUTRAL": "⚪ NEUTRAL",
}
latest = latest.assign(Alert=latest["Signal"].map(sig_icon))

st.subheader(f"📋 Latest alerts — {latest_date.date()}")
cols = ["Ticker", "Alert", "Flags", "Close", "SMA20", "SMA50", "RSI14"]
st.dataframe(latest[cols].sort_values(["Alert", "Ticker"], ascending=[True, True]), use_container_width=True)

# One chart (first ticker)
st.subheader("📊 Chart")
first = tickers[0]
g = df[df["Ticker"] == first].sort_values("Date")
st.markdown(f"### {first}")

if go is not None:
    fig = go.Figure()
    fig.add_candlestick(x=g["Date"], open=g["Open"], high=g["High"], low=g["Low"], close=g["Close"], name="OHLC")
    fig.add_trace(go.Scatter(x=g["Date"], y=g["SMA20"], mode="lines", name="SMA20"))
    fig.add_trace(go.Scatter(x=g["Date"], y=g["SMA50"], mode="lines", name="SMA50"))
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.line_chart(g.set_index("Date")["Close"], use_container_width=True)

with st.expander("How the alerts work"):
    st.markdown(
        """
        **Signals** (simple, widely-used combos — not financial advice):
        - **BUY** when trend is up (SMA20>50) **and** momentum up (MACD>signal) **and** RSI>55, or a **🚩Breakout**, or a **✨GoldenCross** occurs.
        - **SELL** when price closes below SMA20, or momentum weakens (MACD<signal **and** RSI<45).
        - **HOLD** when trend is up and price is above SMA20, without a BUY/SELL trigger.
        - **NEUTRAL** otherwise.

        **Flags**:
        - 🚩 **Breakout**: Close > 20‑day high with volume > 20‑day average.
        - 🎯 **Squeeze**: Bollinger Band width is significantly below its 20‑day average.
        - ✨ **GoldenCross**: SMA20 crosses above SMA50 today.
        """
    )

st.caption("Not financial advice. For education only.")
