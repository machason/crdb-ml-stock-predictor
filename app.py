import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import pickle
# -----------------------
# Paths
# -----------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "crdb_data.csv")

# -----------------------
# Load Model + Scaler
# -----------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

# -----------------------
# Load Dataset
# -----------------------
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
        return df
    return None

data = load_data()

# -----------------------
# Sidebar Navigation
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["📊 Dashboard", "🔮 Prediction", "📂 Dataset"])

# -----------------------
# Feature Creation
# -----------------------
def create_features(open_p, high_p, low_p, close_p, volume):
    high_low_pct = ((high_p - low_p) / low_p * 100) if low_p != 0 else 0
    open_close_pct = ((close_p - open_p) / open_p * 100) if open_p != 0 else 0
    daily_return = open_close_pct
    vol_10 = abs(daily_return)
    vol_20 = abs(daily_return)
    ret_1 = daily_return

    return np.array([[high_low_pct, open_close_pct, volume, ret_1, vol_10, vol_20]])

# =====================================================
# 📊 DASHBOARD PAGE
# =====================================================
if page == "📊 Dashboard":

    st.title("CRDB Stock Dashboard")

    if data is not None:
        st.subheader("Latest Market Data")
        st.dataframe(data.tail())

        if "Close" in data.columns:
            st.subheader("Historical Close Price")
            data_chart = data.copy()
            if "Date" in data_chart.columns:
                data_chart.set_index("Date", inplace=True)
            st.line_chart(data_chart["Close"])
    else:
        st.warning("No dataset found.")

# =====================================================
# 🔮 PREDICTION PAGE
# =====================================================
elif page == "🔮 Prediction":

    st.title("5-Day Return Prediction")

    st.subheader("Manual Input")

    col1, col2 = st.columns(2)

    with col1:
        open_price = st.number_input("Open", 0.01, value=500.0)
        high_price = st.number_input("High", 0.01, value=510.0)

    with col2:
        low_price = st.number_input("Low", 0.01, value=495.0)
        close_price = st.number_input("Close", 0.01, value=505.0)

    volume = st.number_input("Volume", 0.0, value=0.0)

    if st.button("Predict from Manual Input"):

        features = create_features(open_price, high_price, low_price, close_price, volume)
        features_scaled = scaler.transform(features)
        predicted_return = model.predict(features_scaled)[0]

        st.success(f"Predicted 5-Day Return: {predicted_return*100:.2f}%")

        # Growth Projection
        days = np.arange(0, 6)
        projected_prices = close_price * ((1 + predicted_return) ** (days / 5))

        growth_df = pd.DataFrame({
            "Day": days,
            "Projected Price": projected_prices
        })

        st.line_chart(growth_df.set_index("Day"))

    # -----------------------
    # Auto Predict from Dataset
    # -----------------------
    st.subheader("Auto Predict from Latest Dataset Row")

    if data is not None and st.button("Predict from Latest Data"):

        last_row = data.iloc[-1]

        open_p = last_row["Open"]
        high_p = last_row["High"]
        low_p = last_row["Low"]
        close_p = last_row["Close"]
        volume_p = last_row["Volume"]

        features = create_features(open_p, high_p, low_p, close_p, volume_p)
        features_scaled = scaler.transform(features)
        predicted_return = model.predict(features_scaled)[0]

        st.success(f"Predicted 5-Day Return: {predicted_return*100:.2f}%")

        days = np.arange(0, 6)
        projected_prices = close_p * ((1 + predicted_return) ** (days / 5))

        growth_df = pd.DataFrame({
            "Day": days,
            "Projected Price": projected_prices
        })

        st.line_chart(growth_df.set_index("Day"))

# =====================================================
# 📂 DATASET PAGE
# =====================================================
elif page == "📂 Dataset":

    st.title("Dataset Viewer")

    if data is not None:
        st.dataframe(data)

        st.write("Dataset Shape:", data.shape)

        if "Close" in data.columns:
            st.subheader("Close Price Chart")
            chart_data = data.copy()
            if "Date" in chart_data.columns:
                chart_data.set_index("Date", inplace=True)
            st.line_chart(chart_data["Close"])
    else:
        st.warning("Dataset not found. Make sure 'crdb_data.csv' is in the same folder.")
