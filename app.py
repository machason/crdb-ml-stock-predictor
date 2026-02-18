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
# Load models (cached for performance)
# ----------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Make sure model.pkl and scaler.pkl are in the repository root.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, scaler = load_model()

# ----------------------
# App title and description
# ----------------------
st.title("CRDB Stock 5-Day Return Predictor")
st.markdown(
    "Enter today's market data to predict the expected **5-day forward return** for CRDB Bank stock."
)

# ----------------------
# User inputs (with safe minimum values)
# ----------------------
st.subheader("Enter Today's Stock Data")

col1, col2 = st.columns(2)

with col1:
    open_price = st.number_input("Open Price (TZS)", min_value=0.01, value=500.0, step=0.01, format="%.2f")
    high_price = st.number_input("High Price (TZS)", min_value=0.01, value=510.0, step=0.01, format="%.2f")

with col2:
    low_price = st.number_input("Low Price (TZS)", min_value=0.01, value=495.0, step=0.01, format="%.2f")
    close_price = st.number_input("Close Price (TZS)", min_value=0.01, value=505.0, step=0.01, format="%.2f")

# Optional: allow user to input today's volume or keep default 0
volume = st.number_input("Volume (optional, default 0)", min_value=0.0, value=0.0, step=1.0, format="%.0f")

predict_button = st.button("🔮 Predict 5-Day Return", type="primary")

# ----------------------
# Safe feature engineering function
# ----------------------
def create_features(open_p, high_p, low_p, close_p, volume=0.0):
    """
    Computes features safely – prevents division by zero and ensures proper shape for model.
    Includes Volume placeholder for scaler compatibility.
    """
    # Ensure inputs are floats
    open_p = float(open_p)
    high_p = float(high_p)
    low_p  = float(low_p)
    close_p = float(close_p)
    volume = float(volume)

    # Safe calculations
    high_low_pct = ((high_p - low_p) / low_p * 100) if low_p != 0 else 0.0
    open_close_pct = ((close_p - open_p) / open_p * 100) if open_p != 0 else 0.0
    daily_return = open_close_pct          # same as OC_PCT in training
    vol_10 = abs(daily_return)             # fallback for rolling volatility
    vol_20 = abs(daily_return)             # fallback for rolling volatility
    ret_1 = daily_return                   # single-day return proxy

    # Return features in the same order as training
    return np.array([[high_low_pct, open_close_pct, volume, ret_1, vol_10, vol_20]])

# ----------------------
# Prediction logic
# ----------------------
if predict_button:
    try:
        features = create_features(open_price, high_price, low_price, close_price, volume)
        features_scaled = scaler.transform(features)
        predicted_return = model.predict(features_scaled)[0]

        # Display result
        st.subheader(f"Predicted 5-Day Return: **{predicted_return * 100:.2f}%**")

        # Interpretation
        if predicted_return > 0.005:
            st.success("📈 The model expects a **positive** price movement over the next 5 days.")
        elif predicted_return < -0.005:
            st.warning("📉 The model expects a **negative** price movement over the next 5 days.")
        else:
            st.info("The model predicts **neutral** movement (close to 0%).")

    except ValueError as ve:
        st.error(str(ve))
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}\n\nPlease check your input values.")

# Footer note
st.markdown("---")
st.caption("Note: Volatility features (VOL_10, VOL_20) are approximated from today's data only, "
           "as historical sequence is not available in live prediction.")
