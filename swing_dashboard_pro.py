

# swing_dashboard_pro.py
# Streamlit Swing Trading Dashboard Pro

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

try:
    import yfinance as yf
except Exception:
    yf = None

def sma(s, n): 
    return s.rolling(n).mean()

def rsi_wilder(close, n=14):
    d = close.diff()
    gain = (d.where(d>0,0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    loss = (-d.where(d<0,0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    rs = gain / loss.replace(0,np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def atr_wilder(h,l,c,n=14):
    pc = c.shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()

def rule_engine(row):
    trend = "Up" if row['SMA20']>row['SMA50'] else "Down"
    note, action = "", ""
    if trend=="Up":
        near = abs(row['Close']-row['SMA20'])<=row['ATR14']
        neutral = 35<=row['RSI14']<=55
        if (row['Close']-row['SMA20'])>2*row['ATR14']:
            action, note = "Trim/Take Partial", ">2×ATR above 20SMA"
        elif near and neutral:
            action, note = "Look to Buy", "Near 20SMA + neutral RSI"
        else:
            action = "Hold"
    else:
        action = "Avoid/Wait"
        if row['Close']>row['SMA50']: note="Counter-trend"
    return pd.Series([trend,note,action], index=['Trend','SetupNote','SuggestedAction'])

def position_size(acct, risk_pct, entry, stop):
    if acct and risk_pct and entry and stop and entry>stop:
        rps = entry-stop
        dollars = acct*(risk_pct/100)
        shares = int(np.floor(dollars/rps)) if rps>0 else 0
        return shares, rps, shares*entry
    return 0,0.0,0.0

st.set_page_config(page_title="Swing Trading Dashboard", layout="wide")
st.title("📈 Swing Trading Dashboard")

with st.sidebar:
    st.header("Settings")
    account_size = st.number_input("Account Size ($)", min_value=0.0, value=1000.0, step=100.0)
    risk = st.number_input("Risk per Trade (%)", 1.0, step=0.25, format="%.2f")
    stop_mult = st.number_input("ATR Multiplier (Initial Stop)", 1.5, step=0.1, format="%.1f")
    st.divider()
    source = st.radio("Data Source", ["Upload CSV","Yahoo Finance"])

if source=="Yahoo Finance":
    syms = st.text_input("Tickers (comma-separated)", "AVGO, NVDA, MSFT")
    end = dt.date.today(); start = end - dt.timedelta(days=365)
    sdt = st.date_input("Start", start); edt = st.date_input("End", end)
else:
    uploaded = st.file_uploader("Upload CSV (Date,Open,High,Low,Close,Volume[,Ticker])", type=["csv"])

df = pd.DataFrame()
if source=="Yahoo Finance":
    if yf is None:
        st.error("Install yfinance: pip install yfinance")
    else:
        symlist = [x.strip().upper() for x in syms.split(",") if x.strip()]
        data = yf.download(symlist, start=sdt, end=edt, auto_adjust=False, threads=True, group_by='ticker')
        frames=[]
        for sym in symlist:
            d = data[sym].reset_index().rename(columns=str.title)
            d["Ticker"]=sym
            frames.append(d[["Date","Open","High","Low","Close","Volume","Ticker"]])
        df = pd.concat(frames).dropna()
else:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "Ticker" not in df.columns: df["Ticker"]="TICK"
        df["Date"]=pd.to_datetime(df["Date"])
        df=df.sort_values(["Ticker","Date"])

if df.empty:
    st.info("Load data to begin.")
    st.stop()

def compute(g):
    g=g.sort_values("Date").copy()
    g["SMA20"]=sma(g["Close"],20); g["SMA50"]=sma(g["Close"],50)
    g["RSI14"]=rsi_wilder(g["Close"],14); g["ATR14"]=atr_wilder(g["High"],g["Low"],g["Close"],14)
    g[["Trend","SetupNote","SuggestedAction"]]=g.apply(rule_engine,axis=1)
    return g

df = df.groupby("Ticker", group_keys=False).apply(compute)
latest = df.sort_values("Date").groupby("Ticker").tail(1).copy()

st.subheader("Watchlist Signals")
cols = ["Ticker","Date","Close","SMA20","SMA50","RSI14","ATR14","Trend","SetupNote","SuggestedAction"]
st.dataframe(latest[cols].set_index("Ticker"), use_container_width=True)

c1,c2,c3,c4,c5=st.columns([1.2,1.2,1.2,1.2,2])
with c1: sel = st.selectbox("Select Ticker", sorted(latest["Ticker"].unique()))
row = latest[latest["Ticker"]==sel].iloc[0]
with c2: entry = st.number_input("Planned Entry", float(round(row["Close"],2)))
with c3:
    istop = entry - row["ATR14"]*stop_mult
    st.write(f"**Initial Stop:** {istop:.2f}")
with c4: st.write(f"ATR14: {row['ATR14']:.2f}")
with c5:
    shares, rps, alloc = position_size(acct, risk, entry, istop)
    st.write(f"Risk/Share: {rps:.2f} | Shares: {shares} | Alloc: ${alloc:,.2f}")

st.caption("Sizing = floor((Account×Risk%)/(Entry−Stop)).")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    st.subheader("Charts")
    view = df[df["Ticker"]==sel].copy().tail(200)
    candles = go.Candlestick(x=view["Date"],open=view["Open"],high=view["High"],low=view["Low"],close=view["Close"],name="Price")
    sma20 = go.Scatter(x=view["Date"],y=view["SMA20"],name="SMA20")
    sma50 = go.Scatter(x=view["Date"],y=view["SMA50"],name="SMA50")
    fig = go.Figure(data=[candles,sma20,sma50])
    fig.update_layout(height=500,xaxis_rangeslider_visible=False,margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig, use_container_width=True)
    rsi_fig = px.line(view,x="Date",y="RSI14",title="RSI14")
    rsi_fig.add_hline(y=30,line_dash="dot")
    rsi_fig.add_hline(y=70,line_dash="dot")
    st.plotly_chart(rsi_fig, use_container_width=True)
except Exception as e:
    st.info("Install Plotly for charts: pip install plotly")
    st.caption(f"(Chart import error: {e})")

st.success("Ready. Use Yahoo mode (needs internet) or CSV upload for offline use.")

