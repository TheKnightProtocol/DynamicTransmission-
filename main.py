import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/random_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Predictive Maintenance")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Single Prediction",
        "Batch Prediction",
        "About"
    ]
)

# -----------------------------
# Utility Functions
# -----------------------------
def health_score(prob):
    return round((1 - prob) * 100, 2)

def recommendation(prob):

    if prob >= 0.85:
        return "🔴 Immediate Maintenance Required"

    elif prob >= 0.60:
        return "🟠 Schedule Maintenance"

    elif prob >= 0.30:
        return "🟡 Increase Monitoring"

    return "🟢 Machine Healthy"

# -----------------------------
# Dashboard
# -----------------------------
if page == "Dashboard":

    st.title("🏭 Predictive Maintenance of Industrial Gearboxes")

    st.write(
        """
        This dashboard predicts machine failure using
        Machine Learning.

        **Internship Inspired Project**
        """
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Model", "Random Forest")
    c2.metric("Version", "1.0")
    c3.metric("Status", "Running")

    st.image(
        "https://images.unsplash.com/photo-1565043589221-1a6fd9ae45c7?w=1200",
        use_container_width=True
    )

# -----------------------------
# Single Prediction
# -----------------------------
elif page == "Single Prediction":

    st.header("Machine Sensor Inputs")

    col1, col2 = st.columns(2)

    with col1:

        air = st.number_input(
            "Air Temperature (K)",
            value=298.5
        )

        process = st.number_input(
            "Process Temperature (K)",
            value=309.7
        )

        rpm = st.number_input(
            "Rotational Speed (RPM)",
            value=1500
        )

    with col2:

        torque = st.number_input(
            "Torque (Nm)",
            value=42.0
        )

        wear = st.number_input(
            "Tool Wear (min)",
            value=100
        )

    if st.button("Predict Failure"):

        df = pd.DataFrame([[
            air,
            process,
            rpm,
            torque,
            wear
        ]], columns=[
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ])

        X = scaler.transform(df)

        probability = model.predict_proba(X)[0][1]
        prediction = model.predict(X)[0]

        score = health_score(probability)

        st.success("Prediction Completed")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Failure Probability",
                f"{probability:.2%}"
            )

            st.metric(
                "Health Score",
                f"{score}/100"
            )

            st.metric(
                "Prediction",
                "Failure" if prediction else "Healthy"
            )

        with c2:

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Machine Health"},
                gauge={
                    'axis': {'range': [0,100]},
                    'bar': {'color': "green"},
                    'steps':[
                        {'range':[0,40],'color':'red'},
                        {'range':[40,70],'color':'yellow'},
                        {'range':[70,100],'color':'lightgreen'}
                    ]
                }
            ))

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.info(recommendation(probability))

# -----------------------------
# Batch Prediction
# -----------------------------
elif page == "Batch Prediction":

    st.header("Upload CSV")

    uploaded = st.file_uploader(
        "Choose CSV",
        type=["csv"]
    )

    if uploaded:

        df = pd.read_csv(uploaded)

        st.write(df.head())

        X = scaler.transform(df)

        prob = model.predict_proba(X)[:,1]

        pred = model.predict(X)

        df["Prediction"] = pred

        df["Failure Probability"] = prob

        df["Health Score"] = (1-prob)*100

        st.success("Batch Prediction Completed")

        st.dataframe(df)

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            file_name="predictions.csv"
        )

# -----------------------------
# About
# -----------------------------
else:

    st.title("About Project")

    st.write(
        """
        ## Predictive Maintenance of Industrial Gearboxes

        **Technology Stack**

        - Python
        - Scikit-Learn
        - Random Forest
        - Streamlit
        - Plotly
        - Pandas

        **Dataset**

        AI4I 2020 Predictive Maintenance Dataset

        **Objective**

        Predict industrial gearbox failures before they occur,
        enabling proactive maintenance and reducing downtime.
        """
    )

st.sidebar.markdown("---")
st.sidebar.write("Developed by Sankalp Sharma")
st.sidebar.write(datetime.now().strftime("%d %B %Y"))
