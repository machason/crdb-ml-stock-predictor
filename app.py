import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import pickle
# ----------------------
# Paths to model files
# ----------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ----------------------
# Load models
# ----------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        st.stop()

model, scaler = load_model()

# ----------------------
# App UI
# ----------------------
st.title("CRDB Stock 5-Day Return Predictor")

st.markdown(
    "Enter today's market data to predict the expected **5-day forward return** "
    "for CRDB Bank stock."
)

st.subheader("Enter Today's Stock Data")

col1, col2 = st.columns(2)

with col1:
    open_price = st.number_input("Open Price (TZS)", min_value=0.01, value=500.0, step=0.01)
    high_price = st.number_input("High Price (TZS)", min_value=0.01, value=510.0, step=0.01)

with col2:
    low_price = st.number_input("Low Price (TZS)", min_value=0.01, value=495.0, step=0.01)
    close_price = st.number_input("Close Price (TZS)", min_value=0.01, value=505.0, step=0.01)

# Volume included because model was trained with it
volume = st.number_input("Volume", min_value=0.0, value=0.0, step=1.0)

predict_button = st.button("🔮 Predict 5-Day Return", type="primary")

# ----------------------
# Feature Engineering
# ----------------------
def create_features(open_p, high_p, low_p, close_p, volume):
    open_p = float(open_p)
    high_p = float(high_p)
    low_p = float(low_p)
    close_p = float(close_p)
    volume = float(volume)

    high_low_pct = ((high_p - low_p) / low_p * 100) if low_p != 0 else 0.0
    open_close_pct = ((close_p - open_p) / open_p * 100) if open_p != 0 else 0.0
    daily_return = open_close_pct
    vol_10 = abs(daily_return)
    vol_20 = abs(daily_return)
    ret_1 = daily_return

    # Must match training order exactly:
    # ['HL_PCT', 'OC_PCT', 'Volume', 'RET_1', 'VOL_10', 'VOL_20']
    return np.array([[high_low_pct, open_close_pct, volume, ret_1, vol_10, vol_20]])

# ----------------------
# Prediction + Chart
# ----------------------
if predict_button:
    try:
        features = create_features(open_price, high_price, low_price, close_price, volume)
        features_scaled = scaler.transform(features)
        predicted_return = model.predict(features_scaled)[0]

        st.subheader(f"Predicted 5-Day Return: **{predicted_return * 100:.2f}%**")

        # Interpretation
        if predicted_return > 0.005:
            st.success("📈 The model expects a positive price movement.")
        elif predicted_return < -0.005:
            st.warning("📉 The model expects a negative price movement.")
        else:
            st.info("The model predicts neutral movement.")

        # ----------------------
        # 5-Day Growth Projection Chart
        # ----------------------
        st.subheader("📊 5-Day Projected Price Growth")

        days = np.arange(0, 6)

        # Compounded projection (better than linear)
        projected_prices = close_price * ((1 + predicted_return) ** (days / 5))

        growth_df = pd.DataFrame({
            "Day": days,
            "Projected Price (TZS)": projected_prices
        })

        st.line_chart(growth_df.set_index("Day"))

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.caption(
    "Volatility features (VOL_10, VOL_20) are approximated using today's movement only. "
    "This projection is model-based and not financial advice."
)
