# swing_dashboard_pro_resilient.py
# Streamlit Swing Dashboard — resilient and deploy-ready
# Includes: signals, alerts, position log, risk sizing, charting, simple backtest

import math
import datetime as dt
from typing import List, Tuple, Optional, Dict

import streamlit as st
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    import smtplib, ssl
except Exception:
    smtplib = None
    ssl = None


# ----------------------------- Utility Functions -----------------------------

def _normalize_tickers(raw) -> List[str]:
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


# ----------------------------- Indicator Functions -----------------------------

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


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close: pd.Series, n=20, num_std=2.0):
    ma = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return upper, ma, lower


def atr(df: pd.DataFrame, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


# ----------------------------- Data Fetch -----------------------------
@st.cache_data(show_spinner=False)
def fetch_prices(tickers: Tuple[str, ...], start: dt.date, end: dt.date, interval: str):
    if yf is None:
        raise RuntimeError("yfinance not installed")

    if not tickers:
        return pd.DataFrame()

    dl = yf.download(
        list(tickers),
        start=start,
        end=end,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        threads=False,
        progress=False,
    )

    if dl is None or getattr(dl, "empty", True):
        return pd.DataFrame()

    if isinstance(dl.columns, pd.MultiIndex):
        long_df = dl.stack(level=0).rename_axis(["Date", "Ticker"]).reset_index()
    else:
        long_df = dl.reset_index()
        long_df["Ticker"] = tickers[0] if tickers else "TICKER"

    long_df = long_df.rename(columns={"Adj Close": "Close"})
    long_df["Date"] = pd.to_datetime(long_df["Date"], errors="coerce")
    return long_df.dropna(subset=["Date"])


# ----------------------------- Indicators -----------------------------

def compute_indicators(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("Date").copy()
    g["SMA20"] = sma(g["Close"], 20)
    g["SMA50"] = sma(g["Close"], 50)
    g["RSI14"] = rsi_wilder(g["Close"], 14)
    g["ATR14"] = atr(g, 14)
    g["MACD"], g["MACDsig"], g["MACDhist"] = macd(g["Close"])
    g["BBU"], g["BBM"], g["BBL"] = bollinger(g["Close"])

    # Entry & Exit
    g["cross_up"] = (g["SMA20"].shift(1) <= g["SMA50"].shift(1)) & (g["SMA20"] > g["SMA50"])
    g["entry_sig"] = g["cross_up"] & (g["Close"] > g["SMA20"]) & (g["RSI14"] > 50)
    g["macd_down"] = (g["MACD"].shift(1) >= g["MACDsig"].shift(1)) & (g["MACD"] < g["MACDsig"])
    g["exit_sig"] = (g["Close"] < g["SMA20"]) | (g["RSI14"] < 45) | g["macd_down"]
    return g


# ----------------------------- Email Alerts -----------------------------

def send_email_alert(subject: str, body: str, to_email: str) -> str:
    """Send email using credentials in st.secrets"""
    if smtplib is None or ssl is None:
        return "Email libs missing."
    secrets = st.secrets if hasattr(st, "secrets") else {}
    required = ["EMAIL_HOST", "EMAIL_PORT", "EMAIL_USER", "EMAIL_PASSWORD", "EMAIL_FROM"]
    if not all(k in secrets for k in required):
        return "Email secrets missing in st.secrets."

    host = secrets["EMAIL_HOST"]
    port = int(secrets["EMAIL_PORT"])
    user = secrets["EMAIL_USER"]
    pwd = secrets["EMAIL_PASSWORD"]
    from_addr = secrets["EMAIL_FROM"]

    msg = (
        f"From: {from_addr}
"
        f"To: {to_email}
"
        f"Subject: {subject}

"
        f"{body}"
    )

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as server:
            server.login(user, pwd)
            server.sendmail(from_addr, [to_email], msg.encode("utf-8"))
        return "Email sent successfully"
    except Exception as e:
        return f"Email failed: {e}"


# ----------------------------- Backtest -----------------------------

def backtest_signals(g: pd.DataFrame, cash0: float = 10000.0, slippage_bps: float = 5.0) -> Dict[str, pd.DataFrame]:
    """Very simple long-only backtest using entry_sig/exit_sig.
    - Buy next open after entry_sig if flat
    - Sell next open after exit_sig if long
    - Slippage in basis points applied on each trade
    Returns dict with: 'trades' DataFrame and 'equity' DataFrame
    """
    g = g.sort_values("Date").reset_index(drop=True).copy()
    g["Open_next"] = g["Open"].shift(-1)

    cash = cash0
    pos = 0  # shares
    eq_curve = []
    trades = []

    for i, row in g.iterrows():
        date = row["Date"]
        price = row["Close"]
        equity = cash + pos * price
        eq_curve.append({"Date": date, "Equity": equity})

        # Skip last row for entries because we use next open
        if i >= len(g) - 1:
            continue

        next_open = g.at[i, "Open_next"]
        if pd.isna(next_open):
            continue

        # Exit condition
        if pos > 0 and row["exit_sig"]:
            sell_px = float(next_open) * (1 - slippage_bps / 10000.0)
            cash += pos * sell_px
            trades.append({"Date": g.at[i+1, "Date"], "Action": "SELL", "Price": sell_px, "Shares": pos})
            pos = 0
            continue

        # Entry condition
        if pos == 0 and row["entry_sig"]:
            buy_px = float(next_open) * (1 + slippage_bps / 10000.0)
            shares = math.floor(cash / buy_px)
            if shares > 0:
                cash -= shares * buy_px
                pos += shares
                trades.append({"Date": g.at[i+1, "Date"], "Action": "BUY", "Price": buy_px, "Shares": shares})

    # Liquidate at final close for reporting
    if pos > 0:
        final_px = g.iloc[-1]["Close"]
        cash += pos * final_px
        trades.append({"Date": g.iloc[-1]["Date"], "Action": "SELL_EOD", "Price": final_px, "Shares": pos})
        pos = 0

    equity_df = pd.DataFrame(eq_curve)
    trades_df = pd.DataFrame(trades)
    return {"equity": equity_df, "trades": trades_df}


# ----------------------------- Streamlit UI -----------------------------

st.set_page_config(page_title="Swing Dashboard", layout="wide")
st.title("📈 Swing Trading Dashboard — Signals, Risk, Log, Backtest")

# Sidebar
with st.sidebar:
    st.subheader("Watchlist")
    tickers = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,NVDA")
    tickers = _normalize_tickers(tickers)
    start = st.date_input("Start", dt.date.today() - dt.timedelta(days=365))
    end = st.date_input("End", dt.date.today())
    interval = st.selectbox("Interval", ["1d", "1h", "15m"], index=0)

    st.subheader("Email Alerts")
    email = st.text_input("Alert email")
    enable_entry = st.checkbox("Send ENTRY emails")
    enable_exit = st.checkbox("Send EXIT emails")

    st.subheader("Risk Sizing")
    acct_size = st.number_input("Account size ($)", 1000.0)
    risk_pct = st.slider("Risk % per trade", 0.5, 5.0, 1.0)
    atr_mult = st.slider("ATR Stop multiple", 1.0, 5.0, 2.0)

# Download + compute
if not tickers:
    st.warning("Enter at least one ticker.")
    st.stop()

data = fetch_prices(tuple(tickers), start, end, interval)
if data.empty:
    st.error("No data returned.")
    st.stop()

data = data.groupby("Ticker", group_keys=False).apply(compute_indicators)
latest_date = data["Date"].max()

# Signals
latest = data[data["Date"] == latest_date]
st.subheader(f"🚨 Signals ({latest_date.date()})")
st.dataframe(latest[["Ticker", "Close", "SMA20", "SMA50", "RSI14", "entry_sig", "exit_sig"]])

# Risk panel
st.subheader("🎯 Risk & Position Sizing")
rows = []
for t in tickers:
    g = data[data["Ticker"] == t].tail(1)
    if g.empty:
        continue
    close, atrv = g["Close"].iloc[-1], g["ATR14"].iloc[-1]
    stop = close - atr_mult * atrv
    risk_per_share = max(close - stop, 0.01)
    dollar_risk = acct_size * (risk_pct / 100)
    qty = math.floor(dollar_risk / risk_per_share) if risk_per_share > 0 else 0
    rows.append({"Ticker": t, "Close": round(close,2), "ATR14": round(atrv,2), "Stop": round(stop,2), "Qty": int(qty)})
st.dataframe(pd.DataFrame(rows))

# Chart
st.subheader("📊 Charts")
for t in tickers[:2]:
    g = data[data["Ticker"] == t]
    if g.empty:
        continue
    if go:
        fig = go.Figure()
        fig.add_candlestick(x=g["Date"], open=g["Open"], high=g["High"], low=g["Low"], close=g["Close"])
        fig.add_trace(go.Scatter(x=g["Date"], y=g["SMA20"], name="SMA20"))
        fig.add_trace(go.Scatter(x=g["Date"], y=g["SMA50"], name="SMA50"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(g.set_index("Date")["Close"], use_container_width=True)

# ----------------------------- Backtest UI -----------------------------

st.subheader("📒 Backtest (simple, long-only)")
with st.expander("Run backtest"):
    bt_cash = st.number_input("Initial cash ($)", min_value=1000.0, value=10000.0, step=500.0)
    bt_slip = st.number_input("Slippage (bps)", min_value=0.0, value=5.0, step=1.0, help="1 bps = 0.01%")
    bt_ticker = st.selectbox("Ticker", tickers, index=0)
    run_bt = st.button("Run Backtest")

    if run_bt:
        g = data[data["Ticker"] == bt_ticker].copy()
        res = backtest_signals(g, cash0=bt_cash, slippage_bps=bt_slip)
        eq = res["equity"]
        tr = res["trades"]
        # Metrics
        total_return = (eq["Equity"].iloc[-1] / eq["Equity"].iloc[0]) - 1.0 if len(eq) > 1 else 0.0
        daily_ret = eq["Equity"].pct_change().dropna()
        sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-9)) * (252 ** 0.5) if not daily_ret.empty else 0.0
        cummax = eq["Equity"].cummax()
        drawdown = (eq["Equity"] / cummax - 1.0).min() if not eq.empty else 0.0

        m = pd.DataFrame({
            "Metric": ["Total Return", "Sharpe (daily)", "Max Drawdown", "Trades"],
            "Value": [f"{total_return*100:.2f}%", f"{sharpe:.2f}", f"{drawdown*100:.2f}%", len(tr)]
        })
        st.dataframe(m, use_container_width=True)

        if go:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq["Date"], y=eq["Equity"], name="Equity"))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(eq.set_index("Date")["Equity"], use_container_width=True)

        st.write("Trades:")
        st.dataframe(tr, use_container_width=True)

st.caption("⚠️ For educational use only — not financial advice.")
