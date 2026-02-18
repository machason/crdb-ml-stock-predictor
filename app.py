
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
@st.cache_data  
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Make sure model.pkl and scaler.pkl are in the repo.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, scaler = load_model()

# ----------------------
# App title
# ----------------------
st.title("CRDB Stock 5-Day Return Predictor")

# ----------------------
# User inputs
# ----------------------
st.subheader("Enter today's stock data:")

open_price = st.number_input("Open Price", min_value=0.0, step=0.01)
high_price = st.number_input("High Price", min_value=0.0, step=0.01)
low_price = st.number_input("Low Price", min_value=0.0, step=0.01)
close_price = st.number_input("Close Price", min_value=0.0, step=0.01)

# ----------------------
# Feature Engineering
# ----------------------
def create_features(open_p, high_p, low_p, close_p):
    high_low_pct = (high_p - low_p) / low_p * 100
    open_close_pct = (close_p - open_p) / open_p * 100
    daily_return = (close_p - open_p) / open_p * 100

    vol_10 = 0
    vol_20 = 0
    return np.array([[high_low_pct, open_close_pct, daily_return, vol_10, vol_20]])

features = create_features(open_price, high_price, low_price, close_price)
features_scaled = scaler.transform(features)

# ----------------------
# Prediction
# ----------------------
predicted_return = model.predict(features_scaled)[0]
st.subheader(f"Predicted 5-day return: {predicted_return:.2f}%")
