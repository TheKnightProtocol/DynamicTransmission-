import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -----------------------------
# Locate Project Directory
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# -----------------------------
# Load Model & Scaler
# -----------------------------
@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        st.error(f"❌ Model file not found:\n{MODEL_PATH}")
        st.info("Run train_model.py first to generate random_forest.pkl")
        st.stop()

    if not SCALER_PATH.exists():
        st.error(f"❌ Scaler file not found:\n{SCALER_PATH}")
        st.info("Run train_model.py first to generate scaler.pkl")
        st.stop()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler

# Load once and cache
model, scaler = load_model()
