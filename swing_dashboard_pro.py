# swing_dashboard_pro_resilient.py
# Streamlit Swing Dashboard — resilient rewrite + signals, alerts, position log, risk sizing
# - URL + session ticker persistence
# - Robust yfinance download (single or multiple tickers)
# - Safe caching (tuple-hashed args) and defensive schema checks
# - Common indicators (SMA, EMA, RSI, MACD, Bollinger, ATR)
# - Simple bull-flag heuristic screener
# - Entry/Exit signals, optional email alerts (entry & exit), Position Log with P/L
# - Risk box: ATR-based stop and position sizing by % account risk

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

try:
    import smtplib, ssl  # for optional email alerts
except Exception:
    smtplib = None
    ssl = None

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


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close: pd.Series, n: int = 20, num_std: float = 2.0):
    ma = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return upper, ma, lower


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range over n periods.
    Requires columns: High, Low, Close
    """
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


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

    dl = yf.download(list(tickers), start=start, end=end, interval=interval,
                     group_by="ticker", auto_adjust=False, threads=False, progress=False)

    if dl is None or getattr(dl, "empty", True):
        return pd.DataFrame()

    # If MultiIndex columns (multiple tickers), reshape to long
    if isinstance(dl.columns, pd.MultiIndex):
        long_df = (
            dl.stack(level=0)  # ticker level -> rows
              .rename_axis(["Date", "Ticker"])  # index names
              .reset_index()
        )
    else:
        long_df = dl.reset_index()
        t0 = tickers[0] if tickers else "TICKER"
        long_df["Ticker"] = t0

    # Standardize column names
    long_df = long_df.rename(columns={"Adj Close": "Close", "Date": "Date"})

    required = {"Date", "Open", "High", "Low", "Close", "Volume", "Ticker"}
    missing = required.difference(long_df.columns)
    if missing:
        raise KeyError(f"Missing columns after download/reshape: {sorted(missing)}")

    long_df["Date"] = pd.to_datetime(long_df["Date"], errors="coerce")
    long_df = long_df.dropna(subset=["Date"]).copy()
    return long_df


def compute_indicators(g: pd.DataFrame) -> pd.DataFrame:
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
    g["ATR14"] = atr(g, 14)

    # Daily returns
    g["Ret"] = g["Close"].pct_change()

    # Signals (very simple rules; NOT financial advice)
    # Entry when: SMA20 crosses above SMA50 AND Close > SMA20 AND RSI > 50
    g["cross_up"] = (g["SMA20"].shift(1) <= g["SMA50"].shift(1)) & (g["SMA20"] > g["SMA50"])  # golden cross today
    g["entry_sig"] = g["cross_up"] & (g["Close"] > g["SMA20"]) & (g["RSI14"] > 50)

    # Exit when: Close < SMA20 OR RSI < 45 OR MACD crosses down
    g["macd_cross_down"] = (g["MACD"].shift(1) >= g["MACDsig"].shift(1)) & (g["MACD"] < g["MACDsig"])
    g["exit_sig"] = (g["Close"] < g["SMA20"]) | (g["RSI14"] < 45) | g["macd_cross_down"]

    return g


def bull_flag_score(df: pd.DataFrame, lookback: int = 25) -> float:
    if len(df) < lookback + 5:
        return 0.0
    sub = df.tail(lookback).copy()
    mom = float((sub["Close"].iloc[-1] > sub["SMA20"].iloc[-1] > sub["SMA50"].iloc[-1]))
    rng = (sub["High"].max() - sub["Low"].min())
    recent = sub.tail(5)
    cons_width = (recent["High"] - recent["Low"]).mean()
    tight = 1.0 - float(cons_width / rng) if rng > 0 else 0.0
    tight = max(0.0, min(1.0, tight))
    slope = np.polyfit(range(5), recent["Close"].to_numpy(), 1)[0]
    down = 1.0 if slope < 0 else 0.0
    score = 0.45 * mom + 0.35 * tight + 0.20 * down
    return round(float(max(0.0, min(1.0, score))), 3)


# ----------------------------- Alerts (optional email) -----------------------------

def send_email_alert(subject: str, body: str, to_email: str) -> str:
    """Send email via creds in st.secrets. Returns status message.
    Requires secrets:
      EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASSWORD, EMAIL_FROM
    """
    if smtplib is None or ssl is None:
        return "Email libs missing. Add smtplib/ssl (stdlib) — usually available."
    secrets = st.secrets if hasattr(st, "secrets") else {}
    needed = ["EMAIL_HOST", "EMAIL_PORT", "EMAIL_USER", "EMAIL_PASSWORD", "EMAIL_FROM"]
    if not all(k in secrets for k in needed):
        return "Email secrets not configured. Add EMAIL_HOST/PORT/USER/PASSWORD/FROM to st.secrets."

    host = secrets["EMAIL_HOST"]
    port = int(secrets["EMAIL_PORT"])  # e.g., 465 for SSL
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
        return "Email sent"
    except Exception as e:
        return f"Email failed: {e}"


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

# Position log state (DataFrame)
if "positions" not in st.session_state:
    st.session_state.positions = pd.DataFrame(columns=[
        "OpenDate", "Ticker", "Side", "Qty", "Entry", "Stop", "Target", "Notes", "Closed", "CloseDate", "ExitPrice"
    ])

# Sidebar controls
with st.sidebar:
    st.title("⚡ Swing Dashboard")
    st.caption("URL & session-persistent watchlist — NOT financial advice")

    st.subheader("Watchlist")
    st.write(", ".join(st.session_state.tickers) or "—")

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

    st.divider()
    st.subheader("Alerting")
    alert_email = st.text_input("Alert email (optional)", value="")
    enable_email_entries = st.checkbox("Email me when a new ENTRY signal appears today", value=False)
    enable_email_exits = st.checkbox("Email me when a new EXIT signal appears today", value=False)

    st.divider()
    st.subheader("Risk / Position Sizing")
    acct_size = st.number_input("Account size ($)", min_value=0.0, step=100.0, value=10000.0, format="%.2f")
    risk_pct = st.slider("Risk per trade (% of account)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    atr_mult = st.slider("ATR multiple for stop", min_value=1.0, max_value=5.0, value=2.0, step=0.5)


# ----------------------------- Main: Data + Views -----------------------------

st.title("📈 Swing Trading Dashboard — Signals • Alerts • Log • Risk")

# Fetch & compute
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

# ----------------------------- Signals Today -----------------------------

st.subheader("🚨 Signals (today)")
latest_date = df["Date"].max()
latest = df[df["Date"] == latest_date].copy()
show_cols = ["Date", "Ticker", "Close", "SMA20", "SMA50", "RSI14", "MACD", "MACDsig", "ATR14", "entry_sig", "exit_sig"]
if not latest.empty:
    st.dataframe(latest[show_cols].sort_values("Ticker"), use_container_width=True)
else:
    st.info("No rows match today; check your date range/interval.")

# ----------------------------- Email alerts -----------------------------

# De-dup prevent multiple sends per day within this session
if "_alert_state" not in st.session_state:
    st.session_state._alert_state = {"date": None, "entries": set(), "exits": set()}

if alert_email:
    alert_state = st.session_state._alert_state
    today_str = str(pd.to_datetime(latest_date).date()) if not pd.isna(latest_date) else None
    if alert_state["date"] != today_str:
        alert_state["date"] = today_str
        alert_state["entries"] = set()
        alert_state["exits"] = set()

    # ENTRY alerts
    if enable_email_entries and today_str:
        entries = latest[latest["entry_sig"]]
        if not entries.empty:
            new_ticks = [t for t in entries["Ticker"].tolist() if t not in alert_state["entries"]]
            if new_ticks:
                subject = f"Swing Dashboard: {len(new_ticks)} new ENTRY signal(s) on {today_str}"
                lines = ["Entry signals:
"]
                for _, r in entries[entries["Ticker"].isin(new_ticks)].iterrows():
                    lines.append(
                        f"- {r['Ticker']}: Close={r['Close']:.2f}, SMA20={r['SMA20']:.2f}, SMA50={r['SMA50']:.2f}, RSI14={r['RSI14']:.1f}"
                    )
                body = "
".join(lines)
                res = send_email_alert(subject, body, alert_email)
                st.toast(res)
                alert_state["entries"].update(new_ticks)

    # EXIT alerts
    if enable_email_exits and today_str:
        exits = latest[latest["exit_sig"]]
        if not exits.empty:
            new_ticks = [t for t in exits["Ticker"].tolist() if t not in alert_state["exits"]]
            if new_ticks:
                subject = f"Swing Dashboard: {len(new_ticks)} new EXIT signal(s) on {today_str}"
                lines = ["Exit signals:
"]
                for _, r in exits[exits["Ticker"].isin(new_ticks)].iterrows():
                    lines.append(
                        f"- {r['Ticker']}: Close={r['Close']:.2f} triggered exit rule (SMA20/RSI/MACD)."
                    )
                body = "
".join(lines)
                res = send_email_alert(subject, body, alert_email)
                st.toast(res)
                alert_state["exits"].update(new_ticks)

# ----------------------------- Risk / Position Sizing panel -----------------------------


st.subheader("🎯 Risk & Size Suggestions (based on latest bar)")

if acct_size and risk_pct and atr_mult:
    latest_by_ticker = (
        df.sort_values(["Ticker", "Date"]).groupby("Ticker").tail(1)
    )
    latest_by_ticker = latest_by_ticker.set_index("Ticker")

    rows = []
    for tkr in tickers:
        if tkr not in latest_by_ticker.index:
            continue
        row = latest_by_ticker.loc[tkr]
        close = float(row["Close"]) if not pd.isna(row["Close"]) else np.nan
        atr14 = float(row["ATR14"]) if not pd.isna(row["ATR14"]) else np.nan
        if np.isnan(close) or np.isnan(atr14) or atr14 <= 0:
            continue
        stop = round(close - atr_mult * atr14, 2)
        risk_per_share = max(close - stop, 0.01)
        dollar_risk = acct_size * (risk_pct / 100.0)
        qty = math.floor(dollar_risk / risk_per_share) if risk_per_share > 0 else 0
        rows.append({
            "Ticker": tkr,
            "Close": round(close, 2),
            "ATR14": round(atr14, 2),
            "Stop(@ATR×{:.1f})".format(atr_mult): stop,
            "$Risk": round(dollar_risk, 2),
            "Qty": int(qty),
            "$Position": round(qty * close, 2),
        })

    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("Ticker"), use_container_width=True)
    else:
        st.info("Need valid ATR/Close values to compute risk sizing.")

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
            st.line_chart(g.set_index("Date")[ ["BBU", "BBM", "BBL"] ].dropna(), height=200)

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

# ----------------------------- Position Log -----------------------------

st.subheader("🧾 Position Log")

with st.expander("Add position"):
    c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1,1,1])
    open_date = c1.date_input("Open Date", value=dt.date.today())
    tkr = c2.selectbox("Ticker", options=st.session_state.tickers or [""], index=0 if st.session_state.tickers else 0)
    side = c3.selectbox("Side", ["LONG", "SHORT"], index=0)
    qty = c4.number_input("Qty", min_value=0.0, step=1.0)
    entry = c5.number_input("Entry", min_value=0.0, step=0.01, format="%.2f")
    stop = c6.number_input("Stop", min_value=0.0, step=0.01, format="%.2f")
    c7, c8 = st.columns([1,3])
    target = c7.number_input("Target", min_value=0.0, step=0.01, format="%.2f")
    notes = c8.text_input("Notes", value="")
    add_pos = st.button("Add Position")

    if add_pos and tkr and qty and entry:
        new_row = {
            "OpenDate": pd.to_datetime(open_date),
            "Ticker": tkr,
            "Side": side,
            "Qty": float(qty),
            "Entry": float(entry),
            "Stop": float(stop) if stop else np.nan,
            "Target": float(target) if target else np.nan,
            "Notes": notes,
            "Closed": False,
            "CloseDate": pd.NaT,
            "ExitPrice": np.nan,
        }
        st.session_state.positions = pd.concat([st.session_state.positions, pd.DataFrame([new_row])], ignore_index=True)
        st.success("Position added")

# Compute live P/L using latest Close
if not st.session_state.positions.empty:
    last_close = (
        df.sort_values(["Ticker", "Date"]).groupby("Ticker").tail(1)[["Ticker", "Close"]]
    )
    last_close = dict(zip(last_close["Ticker"], last_close["Close"]))

    pos = st.session_state.positions.copy()
    pos["LastClose"] = pos["Ticker"].map(last_close)
    # Unrealized P/L for open trades; realized for closed rows
    def _pl(row):
        if pd.isna(row["ExitPrice"]) and not row["Closed"]:
            px = row["LastClose"]
        else:
            px = row["ExitPrice"]
        if pd.isna(px) or pd.isna(row["Entry"]) or pd.isna(row["Qty"]):
            return np.nan
        mult = 1 if row["Side"] == "LONG" else -1
        return mult * (px - row["Entry"]) * row["Qty"]

    pos["PnL"] = pos.apply(_pl, axis=1)

    st.data_editor(
        pos,
        disabled=["OpenDate", "Ticker", "Side", "Qty", "Entry", "Stop", "Target"],
        use_container_width=True,
        key="positions_editor",
    )

    # Save back any edits (like marking Closed, setting ExitPrice/CloseDate)
    st.session_state.positions = pos

    cdl, cul, cdl2 = st.columns([1,1,2])
    # Download CSV
    csv = pos.to_csv(index=False).encode("utf-8")
    cdl.download_button("Download CSV", csv, file_name="positions.csv", mime="text/csv")
    # Upload CSV
    up = cul.file_uploader("Upload CSV", type="csv")
    if up is not None:
        try:
            nd = pd.read_csv(up, parse_dates=["OpenDate", "CloseDate"], keep_default_na=True)
            st.session_state.positions = nd
            st.success("Positions loaded from CSV")
        except Exception as e:
            st.error(f"CSV load failed: {e}")

st.caption("Not financial advice. Educational purposes only.")
