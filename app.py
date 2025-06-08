
# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
import ta

# ========== AUTO-REFRESH ==========
st_autorefresh(interval=60000, key="auto_refresh")

# ========== STYLING ==========
st.markdown("""
<style>
.ticker-wrapper {
    overflow: hidden;
    white-space: nowrap;
    box-sizing: border-box;
}
.ticker-text {
    display: inline-block;
    padding-left: 100%;
    animation: ticker 20s linear infinite;
    font-weight: bold;
    font-family: monospace;
    font-size: 18px;
}
@keyframes ticker {
    0% { transform: translateX(0%); }
    100% { transform: translateX(-100%); }
}
</style>
""", unsafe_allow_html=True)

# ========== INDEX TICKER ==========


def get_index_data(tickers):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                close = hist['Close'][-1]
                prev_close = hist['Close'][-2] if len(hist) > 1 else close
                change = close - prev_close
                pct_change = (change / prev_close) * \
                    100 if prev_close != 0 else 0
                data[ticker] = (close, change, pct_change)
        except Exception:
            data[ticker] = (None, None, None)
    return data


def format_ticker_text(name, close, change, pct_change):
    if change is None:
        return f"{name}: N/A"
    arrow = "▲" if change > 0 else "▼" if change < 0 else "▶"
    color = "green" if change > 0 else "red" if change < 0 else "gray"
    return f"<span style='color:{color}'>{name}: {close:.2f} {arrow} {pct_change:+.2f}%</span>"


def display_index_ticker():
    indexes = {
        "^DJI": "Dow Jones",
        "^IXIC": "Nasdaq",
        "^GSPC": "S&P 500",
        "^RUT": "Russell 2000",
    }
    index_data = get_index_data(indexes.keys())
    ticker_strings = [format_ticker_text(
        name, *index_data[ticker]) for ticker, name in indexes.items()]
    ticker_line = "  ⚫  ".join(ticker_strings)
    st.markdown(
        f'<div class="ticker-wrapper"><div class="ticker-text">{ticker_line}</div></div>', unsafe_allow_html=True)

# ========== SIDEBAR GUIDE ==========


def display_sidebar_guide():
    with st.sidebar.expander("📘 Indicator Guide"):
        st.markdown("""
        ### 📊 Technical Indicators
        **SMA20** – 20-day average.  
        **RSI** – Momentum scale (0-100).  
        🔻 RSI < 30 = Oversold.  
        🔺 RSI > 70 = Overbought.  
        **Support** – Recent price floor.  
        **Resistance** – Recent price ceiling.  
        **Buy** = RSI < 30 and Price > SMA  
        **Sell** = RSI > 70 and Price < SMA  
        **Earnings** – May cause price volatility.
        """)

# ========== TECHNICAL INDICATORS ==========


def compute_technical_indicators(df):
    df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    bb = ta.volatility.BollingerBands(df['Close'], window=20)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['Buy_Signal'] = ((df['RSI'] < 30) & (
        df['Close'] > df['SMA20'])).astype(int)
    df['Sell_Signal'] = ((df['RSI'] > 70) & (
        df['Close'] < df['SMA20'])).astype(int)
    return df

# ========== SUPPORT & RESISTANCE ==========


def get_support_resistance(df):
    recent = df.tail(30)
    return recent['Low'].min(), recent['High'].max()

# ========== EARNINGS ==========


def get_earnings_events(stock):
    events = []
    try:
        cal = stock.calendar
        if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            dates = cal.loc["Earnings Date"].values
            events = [(pd.to_datetime(d), "Earnings")
                      for d in dates if pd.notna(d)]
    except Exception:
        pass
    return events

# ========== CHARTING ==========


def plot_candlestick(df, support, resistance, earnings):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA20'], mode='lines', name="SMA20", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df[df['Buy_Signal'] == 1].index, y=df[df['Buy_Signal'] == 1]['Close'],
                             mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name="Buy"))
    fig.add_trace(go.Scatter(x=df[df['Sell_Signal'] == 1].index, y=df[df['Sell_Signal'] == 1]['Close'],
                             mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name="Sell"))
    fig.add_hline(y=support, line=dict(color='orange',
                  dash='dot'), annotation_text="Support")
    fig.add_hline(y=resistance, line=dict(color='purple',
                  dash='dot'), annotation_text="Resistance")
    for date, label in earnings:
        fig.add_vline(x=date, line=dict(color='black', dash='dot'))
        fig.add_trace(go.Scatter(x=[date], y=[df['High'].max()], mode='markers+text',
                                 text=[label], marker=dict(color='black', symbol='x', size=12), textposition="top right",
                                 hovertext=[f"{label}: {date.strftime('%Y-%m-%d')}"], showlegend=False))
    st.plotly_chart(fig, use_container_width=True)

# ========== RSI CHART ==========


def plot_rsi(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'], name="RSI", line=dict(color='blue')))
    fig.add_hline(y=70, line=dict(color='red', dash='dash'))
    fig.add_hline(y=30, line=dict(color='green', dash='dash'))
    st.subheader("RSI (Relative Strength Index)")
    st.plotly_chart(fig, use_container_width=True)

# ========== PRICE PREDICTION ==========


def predict_price(df):
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA20', 'RSI']
    X = df[features].fillna(0)
    y = df['Target'].fillna(0)
    model = XGBRegressor(n_estimators=100, max_depth=3, verbosity=0)
    model.fit(X[:-1], y[:-1])
    return model.predict(X.iloc[[-1]])[0]

# ========== MOVEMENT PREDICTION ==========


def predict_movement(df):
    df['Lag_Close_1'] = df['Close'].shift(1)
    df['Lag_RSI_1'] = df['RSI'].shift(1)
    df['Lag_MACD_1'] = df['MACD'].shift(1)
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Price_Change'] = df['Close'].diff().shift(-1)
    df['UpDown'] = (df['Price_Change'] > 0).astype(int)
    df = df.dropna()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA20', 'RSI',
                'MACD', 'MACD_Signal', 'BB_High', 'BB_Low',
                'Lag_Close_1', 'Lag_RSI_1', 'Lag_MACD_1', 'DayOfWeek', 'Month']
    X = df[features].fillna(0)
    y = df['UpDown']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X.iloc[[-1]])[0]
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    return prediction, accuracy

# ========== MAIN APP ==========


def main():
    display_index_ticker()
    display_sidebar_guide()

    st.title("📈 Stock Tracker & Analyzer")
    ticker = st.text_input("Enter a stock symbol:", "AAPL").upper()
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")

    if df.empty:
        st.warning("No data found.")
        return

    df = compute_technical_indicators(df)
    support, resistance = get_support_resistance(df)
    earnings = get_earnings_events(stock)

    latest = df.iloc[-1]
    st.subheader(f"📌 Daily Metrics: {ticker}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Close", f"${latest['Close']:.2f}")
        st.metric("Low", f"${latest['Low']:.2f}")
    with col2:
        st.metric("Open", f"${latest['Open']:.2f}")
        st.metric("High", f"${latest['High']:.2f}")
    with col3:
        st.metric("Volume", f"{int(latest['Volume']):,}")
    with col4:
        daily_change = latest['Close'] - latest['Open']
        pct = (daily_change / latest['Open']) * 100
        st.metric("Daily Change", f"${daily_change:.2f}", f"{pct:+.2f}%")

    plot_candlestick(df, support, resistance, earnings)
    plot_rsi(df)

    predicted_price = predict_price(df)
    st.subheader("📉 Price Prediction")
    st.write(f"Predicted next close: **${predicted_price:.2f}**")

    movement, accuracy = predict_movement(df)
    movement_text = "UP 📈" if movement else "DOWN 📉"
    st.subheader("🔮 Movement Prediction")
    st.write(f"Predicted movement: **{movement_text}**")
    st.write(f"Model accuracy: **{accuracy:.2%}**")


if __name__ == "__main__":
    main()
