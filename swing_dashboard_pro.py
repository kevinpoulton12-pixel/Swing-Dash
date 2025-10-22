# swing_dashboard_pro.py
# Streamlit Swing Trading Dashboard (Watchlist + Positions + Trade Log)
# Small-account friendly, supports fractional shares

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

try:
    import yfinance as yf
except Exception:
    yf = None

# ---------- Indicators ----------
def sma(s, n): 
    return s.rolling(n).mean()

def rsi_wilder(close, n=14):
    d = close.diff()
    gain = (d.where(d > 0, 0.0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def atr_wilder(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()

def rule_engine(row):
    trend = "Up" if row["SMA20"] > row["SMA50"] else "Down"
    note, action = "", ""
    if trend == "Up":
        near = abs(row["Close"] - row["SMA20"]) <= row["ATR14"]
        neutral = 35 <= row["RSI14"] <= 55
        if (row["Close"] - row["SMA20"]) > 2 * row["ATR14"]:
            action, note = "Trim/Take Partial", "Extended > 2×ATR above 20SMA"
        elif near and neutral:
            action, note = "Look to Buy", "Near 20SMA + neutral RSI"
        else:
            action = "Hold"
    else:
        action = "Avoid/Wait"
        if row["Close"] > row["SMA50"]:
            note = "Counter-trend"
    return pd.Series([trend, note, action], index=["Trend","SetupNote","SuggestedAction"])

# ---------- Position sizing ----------
def position_size(account_size, risk_pct, entry, stop, allow_fractional=False):
    if all(x is not None for x in [account_size, risk_pct, entry, stop]) and entry > stop:
        rps = entry - stop
        dollars = account_size * (risk_pct/100.0)
        if rps <= 0:
            return (0 if not allow_fractional else 0.0), 0.0, 0.0
        if allow_fractional:
            shares = dollars / rps
        else:
            shares = int(np.floor(dollars / rps))
        return shares, rps, shares * entry
    return (0 if not allow_fractional else 0.0), 0.0, 0.0

# ---------- UI: Sidebar ----------
st.set_page_config(page_title="Swing Trading Dashboard", layout="wide")
st.title("📈 Swing Trading Dashboard — Watchlist • Positions • Trade Log")

with st.sidebar:
    st.header("Settings")
    account_size = st.number_input("Account Size ($)", min_value=0.0, value=1000.0, step=50.0, format="%.2f")
    risk_pct     = st.number_input("Risk per Trade (%)", min_value=0.0, value=1.00, step=0.25, format="%.2f")
    stop_mult    = st.number_input("ATR Multiplier (Initial Stop)", min_value=0.1, value=1.5, step=0.1, format="%.1f")
    trail_mult   = st.number_input("ATR Multiplier (Trail Stop)", min_value=0.1, value=2.0, step=0.1, format="%.1f")
    allow_fractional = st.checkbox("Allow fractional shares", value=True)
    st.caption("Uptrend = SMA20 > SMA50. Buy zone ≈ within 1×ATR of 20SMA with RSI14 between 35–55.")

    st.divider()
    st.subheader("Data Source")
    source = st.radio("Choose source", ["Upload CSV", "Yahoo Finance"], index=0)
    if source == "Yahoo Finance":
        tickers = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, NVDA")
        end   = dt.date.today()
        start = end - dt.timedelta(days=365)
        start_date = st.date_input("Start", value=start)
        end_date   = st.date_input("End",   value=end)
    else:
        uploaded = st.file_uploader("Upload OHLCV CSV (Date,Open,High,Low,Close,Volume[,Ticker])", type=["csv"])

# ---------- Load Data ----------
df = pd.DataFrame()

if source == "Yahoo Finance":
    if yf is None:
        st.error("yfinance not installed. Run: pip install yfinance")
    else:
        syms = [s.strip().upper() for s in tickers.split(",") if s.strip()]
        if len(syms):
            data = yf.download(syms, start=start_date, end=end_date, auto_adjust=False, threads=True, group_by='ticker')
            frames = []
            for sym in syms:
                try:
                    d = data[sym].reset_index().rename(columns=str.title)
                    d["Ticker"] = sym
                    frames.append(d[["Date","Open","High","Low","Close","Volume","Ticker"]])
                except Exception:
                    pass
            if len(frames):
                df = pd.concat(frames).dropna().reset_index(drop=True)
else:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "Ticker" not in df.columns:
            df["Ticker"] = "TICK"
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Ticker","Date"]).reset_index(drop=True)

if df.empty:
    st.info("Load data to begin — use Yahoo Finance or upload your CSV.")
    st.stop()

# ---------- Indicators & Signals ----------
def compute_indicators(g):
    g = g.sort_values("Date").copy()
    g["SMA20"] = sma(g["Close"], 20)
    g["SMA50"] = sma(g["Close"], 50)
    g["RSI14"] = rsi_wilder(g["Close"], 14)
    g["ATR14"] = atr_wilder(g["High"], g["Low"], g["Close"], 14)
    g[["Trend","SetupNote","SuggestedAction"]] = g.apply(rule_engine, axis=1)
    return g

df = df.groupby("Ticker", group_keys=False).apply(compute_indicators)
latest = df.sort_values("Date").groupby("Ticker").tail(1).copy()

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🔎 Watchlist", "📦 Positions", "📘 Trade Log"])

# --- Watchlist ---
with tab1:
    st.subheader("Signals (latest close)")
    cols = ["Ticker","Date","Close","SMA20","SMA50","RSI14","ATR14","Trend","SetupNote","SuggestedAction"]
    st.dataframe(latest[cols].set_index("Ticker"), use_container_width=True)

    st.markdown("**Position Sizing**")
    c1, c2, c3, c4, c5 = st.columns([1.2,1.2,1.2,1.2,2])
    with c1:
        sel = st.selectbox("Select Ticker", options=sorted(latest["Ticker"].unique()))
    row = latest[latest["Ticker"] == sel].iloc[0]
    with c2:
        planned_entry = st.number_input("Planned Entry", value=float(round(row["Close"],2)))
    with c3:
        initial_stop = planned_entry - (row["ATR14"] * stop_mult)
        st.metric("Initial Stop", f"{initial_stop:.2f}")
    with c4:
        st.metric("ATR14", f"{row['ATR14']:.2f}")
        st.metric("20/50 SMA", f"{row['SMA20']:.2f} / {row['SMA50']:.2f}")
    with c5:
        shares, rps, alloc = position_size(account_size, risk_pct, planned_entry, initial_stop, allow_fractional)
        shares_display = f"{shares:.4f}" if allow_fractional else str(int(shares))
        st.write(f"**Risk/Share:** {rps:.2f}")
        st.write(f"**Position Size (sh):** {shares_display}")
        st.write(f"**Alloc ($):** {alloc:,.2f}")

    if (not allow_fractional) and isinstance(shares, int) and shares == 0:
        st.caption("Shares = 0 because your per-share risk exceeds your $ risk limit. Try a cheaper ticker, tighter stop, or enable fractional shares.")

    st.caption("Sizing = floor((Account × Risk%) / (Entry − Stop)) if fractional OFF; else (Account × Risk%)/(Entry − Stop).")

    # Charts (if Plotly available)
    try:
        import plotly.graph_objects as go
        import plotly.express as px

        st.subheader("Charts")
        view = df[df["Ticker"] == sel].copy().tail(200)
        candles = go.Candlestick(
            x=view["Date"], open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"], name="Price"
        )
        sma20_line = go.Scatter(x=view["Date"], y=view["SMA20"], name="SMA20")
        sma50_line = go.Scatter(x=view["Date"], y=view["SMA50"], name="SMA50")
        fig = go.Figure(data=[candles, sma20_line, sma50_line])
        fig.update_layout(height=480, xaxis_rangeslider_visible=False, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig, use_container_width=True)

        rsi_fig = px.line(view, x="Date", y="RSI14", title="RSI14")
        rsi_fig.add_hline(y=30, line_dash="dot")
        rsi_fig.add_hline(y=70, line_dash="dot")
        st.plotly_chart(rsi_fig, use_container_width=True)
    except Exception as e:
        st.info("Install Plotly for charts: pip install plotly")
        st.caption(f"(Chart import error: {e})")

# --- Positions ---
def default_positions_df():
    cols = [
        "Ticker","EntryDate","EntryPrice","Shares","InitialStop",
        "ATR14_Entry","TrailMult","TrailStop","Last","Target1R","Target2R",
        "UnrealizedPL","PLpct","Rmultiple","Status","Notes"
    ]
    return pd.DataFrame(columns=cols)

with tab2:
    st.subheader("Open Positions")
    if "positions" not in st.session_state:
        st.session_state.positions = default_positions_df()

    def recompute_positions(df_pos, market_latest, trail_mult_default):
        df_pos = df_pos.copy()
        last_map = market_latest.set_index("Ticker")["Close"].to_dict()
        atr_map  = market_latest.set_index("Ticker")["ATR14"].to_dict()

        for i, r in df_pos.iterrows():
            t = r.get("Ticker")
            if not isinstance(t, str) or t == "":
                continue
            last = last_map.get(t)
            atr  = atr_map.get(t)

            if pd.notna(last):
                df_pos.at[i, "Last"] = last

            # Targets (1R/2R)
            if pd.notna(r.get("EntryPrice")) and pd.notna(r.get("InitialStop")):
                risk = r["EntryPrice"] - r["InitialStop"]
                df_pos.at[i, "Target1R"] = r["EntryPrice"] + risk
                df_pos.at[i, "Target2R"] = r["EntryPrice"] + 2*risk

            # Trail stop
            use_mult = r.get("TrailMult", trail_mult_default)
            if pd.notna(last) and pd.notna(atr) and pd.notna(use_mult) and pd.notna(r.get("InitialStop")):
                trail_stop = max(r["InitialStop"], last - atr * float(use_mult))
                df_pos.at[i, "TrailStop"] = trail_stop

            # P/L and R
            if pd.notna(last) and pd.notna(r.get("EntryPrice")) and pd.notna(r.get("Shares")):
                pl = (last - r["EntryPrice"]) * r["Shares"]
                df_pos.at[i, "UnrealizedPL"] = pl
                df_pos.at[i, "PLpct"] = (last / r["EntryPrice"]) - 1
                if pd.notna(r.get("InitialStop")) and (r["EntryPrice"] - r["InitialStop"]) != 0:
                    r_mult = (last - r["EntryPrice"]) / (r["EntryPrice"] - r["InitialStop"])
                    df_pos.at[i, "Rmultiple"] = r_mult
        return df_pos

    edited = st.data_editor(
        st.session_state.positions,
        num_rows="dynamic",
        column_config={
            "EntryDate": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Status": st.column_config.SelectboxColumn(options=["Open","Reduce","Close Soon","On Watch"]),
        },
        use_container_width=True
    )
    st.session_state.positions = recompute_positions(edited, latest, trail_mult)

    cpos1, cpos2, cpos3 = st.columns(3)
    with cpos1:
        st.download_button(
            "⬇️ Download Positions CSV",
            st.session_state.positions.to_csv(index=False).encode(),
            "positions.csv",
            "text/csv",
        )
    with cpos2:
        pos_csv = st.file_uploader("Upload Positions CSV", type=["csv"], key="pos_up")
        if pos_csv is not None:
            st.session_state.positions = pd.read_csv(pos_csv)
            st.success("Positions loaded.")
    with cpos3:
        if st.button("Recompute Now"):
            st.session_state.positions = recompute_positions(st.session_state.positions, latest, trail_mult)

# --- Trade Log ---
def default_log_df():
    cols = ["Ticker","Setup","EntryDate","ExitDate","EntryPrice","ExitPrice","Shares","GrossPL","R","Notes"]
    return pd.DataFrame(columns=cols)

with tab3:
    st.subheader("Closed Trades / Journal")
    if "tradelog" not in st.session_state:
        st.session_state.tradelog = default_log_df()

    def recompute_log(df_log, pos_df):
        df_log = df_log.copy()
        stop_map = {}
