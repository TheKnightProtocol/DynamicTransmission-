import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Diagnostic", layout="wide")

st.title("Dynamic Transmission AI")

BASE_DIR = Path(__file__).resolve().parent

st.write("Current directory:", BASE_DIR)

DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"

st.write("Dataset path:", DATA_PATH)

st.write("Dataset exists:", DATA_PATH.exists())

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)

    st.success("Dataset Loaded Successfully!")

    st.write(df.head())

    st.write(df.columns.tolist())
else:
    st.error("Dataset NOT Found!")
