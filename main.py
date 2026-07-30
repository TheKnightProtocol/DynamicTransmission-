import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------
# Locate Dataset
# ----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Change this if your CSV is somewhere else
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"

# ----------------------------------------------------
# Train Model Automatically
# ----------------------------------------------------

@st.cache_resource
def train_model():

    if not DATA_PATH.exists():
        st.error(f"Dataset not found:\n{DATA_PATH}")
        st.info("Place ai4i2020.csv inside the data folder.")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    # Remove unnecessary columns
    for col in ["UDI", "Product ID", "Type"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model

# Load Model
model = train_model()
