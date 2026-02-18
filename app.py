import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="CRDB 5-Day Return Predictor", layout="centered")

st.title("📈 CRDB 5-Day Stock Return Prediction")
st.markdown(
    "Enter today's market data to estimate the expected 5-day future return "
    "of CRDB Bank stock using the trained Machine Learning model."
)

# ────────────────────────────────────────────────
# Load saved model and scaler
# ────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.markdown("---")

# ────────────────────────────────────────────────
# User Inputs (Human-Friendly)
# ────────────────────────────────────────────────
st.subheader("📊 Enter Today's Market Data")

open_price  = st.number_input("Open Price (TZS)", min_value=0.0, value=500.0)
high_price  = st.number_input("High Price (TZS)", min_value=0.0, value=510.0)
low_price   = st.number_input("Low Price (TZS)", min_value=0.0, value=495.0)
close_price = st.number_input("Close Price (TZS)", min_value=0.0, value=505.0)
volume      = st.number_input("Trading Volume", min_value=0.0, value=1000000.0)

previous_close = st.number_input(
    "Previous Close Price (TZS)",
    min_value=0.0,
    value=500.0
)

predict_button = st.button("🔮 Predict 5-Day Return")

st.markdown("---")

# ────────────────────────────────────────────────
# Prediction Logic
# ────────────────────────────────────────────────
if predict_button:

    # Feature engineering (same logic as notebook)
    try:
        hl_pct = (high_price - low_price) / low_price
        oc_pct = (close_price - open_price) / open_price
        ret_1  = (close_price - previous_close) / previous_close

        # Since we don’t have rolling volatility in live input,
        # we approximate using absolute daily return
        vol_10 = abs(ret_1)
        vol_20 = abs(ret_1)

        input_data = np.array([[hl_pct, oc_pct, volume, ret_1, vol_10, vol_20]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        st.success(f"📈 Predicted 5-Day Return: {prediction * 100:.2f}%")

        # Interpretation
        if prediction > 0:
            st.info("📊 Model expects a positive price movement over the next 5 days.")
        elif prediction < 0:
            st.warning("📉 Model expects a negative price movement over the next 5 days.")
        else:
            st.write("Model predicts neutral movement.")

    except ZeroDivisionError:
        st.error("Low price or open price cannot be zero.")
