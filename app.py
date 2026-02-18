import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# --- Load model and scaler ---
@st.cache_data(show_spinner=True)
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# --- App layout ---
st.title("CRDB Bank 5-Day Stock Return Predictor")
st.write("Enter stock data to predict the 5-day forward return.")

# --- Input form ---
with st.form("input_form"):
    open_price = st.number_input("Open Price", min_value=0.0, step=0.1)
    high_price = st.number_input("High Price", min_value=0.0, step=0.1)
    low_price = st.number_input("Low Price", min_value=0.0, step=0.1)
    close_price = st.number_input("Close Price", min_value=0.0, step=0.1)
    submit = st.form_submit_button("Predict")

if submit:
    # --- Feature engineering ---
    high_low_pct = (high_price - low_price) / low_price
    open_close_pct = (close_price - open_price) / open_price
    # Example: Add placeholder volatility (since we only have one row)
    vol_10 = 0.02  
    vol_20 = 0.03  

    # --- Prepare feature array ---
    features = np.array([[high_low_pct, open_close_pct, vol_10, vol_20]])
    features_scaled = scaler.transform(features)

    # --- Predict ---
    pred = model.predict(features_scaled)[0]
    st.success(f"Predicted 5-day forward return: {pred:.4f}")
