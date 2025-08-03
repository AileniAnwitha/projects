import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Streamlit page settings
st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Predictor")

# --- Stock options ---
options = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Infosys (INFY.NS)": "INFY.NS",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "Custom (Type below)": "CUSTOM"
}

# --- Select stock ---
stock_choice = st.selectbox("🔍 Choose a stock:", list(options.keys()))
custom_ticker = ""
if options[stock_choice] == "CUSTOM":
    custom_ticker = st.text_input("✏️ Enter custom stock ticker (e.g., GOOGL, TCS.NS):").strip().upper()

# --- Form for prediction days ---
with st.form("prediction_form"):
    future_days = st.selectbox("📆 Days to predict ahead:", list(range(1, 11)))
    submitted = st.form_submit_button("📊 Predict")

# --- After submit ---
if submitted:
    ticker = custom_ticker if options[stock_choice] == "CUSTOM" else options[stock_choice]
    if options[stock_choice] == "CUSTOM" and not ticker:
        st.error("⚠️ Please enter a valid custom stock ticker.")
        st.stop()

    # Date range: last 5 years to today
    today = date.today()
    start_date = date(today.year - 5, today.month, today.day)
    end_date = today

    st.info(f"Fetching data for **{ticker}** from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("⚠️ No data found for this ticker.")
        st.stop()

    # --- Feature Engineering ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df.dropna(inplace=True)

    X = df[['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50']]
    y = df['Close']

    # --- Train Model ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # --- Predict future ---
    future_input = df[['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50']].iloc[-future_days:].values
    future_pred = model.predict(future_input)

    # --- Show predictions ---
    st.subheader("📊 Prediction Results")
    if future_days == 1:
        st.success(f"Predicted closing price after 1 day: **${future_pred[0]:.2f}**")
    else:
        st.success(f"Predicted closing prices for next {future_days} days:")
        st.write([f"${float(p):.2f}" for p in future_pred])

    # --- Trend Analysis ---
    latest_close = float(df['Close'].iloc[-1])
    final_predicted = float(future_pred[-1])
    change = final_predicted - latest_close
    percent_change = (change / latest_close) * 100

    if change > 0:
        st.info(f"📈 Predicted to increase by **{percent_change:.2f}%** compared to last close (${latest_close:.2f})")
    elif change < 0:
        st.warning(f"📉 Predicted to decrease by **{abs(percent_change):.2f}%** compared to last close (${latest_close:.2f})")
    else:
        st.info(f"⚖️ Predicted to remain the same as last close (${latest_close:.2f})")

    # --- Plot: Actual vs Predicted
    st.subheader("📈 Actual vs Predicted (Sample)")
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots()
    ax.plot(y_test.values[:100], label="Actual")
    ax.plot(y_pred[:100], label="Predicted")
    ax.set_xlabel("Sample Days")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)
