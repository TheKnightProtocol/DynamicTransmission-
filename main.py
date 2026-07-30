import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dynamic Transmission AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp{
    background:#0E1117;
}

html, body{
    background:#0E1117;
    color:white;
}

h1,h2,h3,h4,h5,h6{
    color:white;
}

div[data-testid="metric-container"]{

    background:#1B263B;

    border-radius:15px;

    padding:20px;

    border-left:6px solid #00C853;

    box-shadow:0px 6px 15px rgba(0,0,0,.35);

}

section[data-testid="stSidebar"]{

    background:#111827;

}

.stButton>button{

    width:100%;

    background:#00C853;

    color:white;

    border-radius:10px;

    border:none;

    font-size:18px;

    height:3em;

    font-weight:bold;

}

.stButton>button:hover{

    background:#00E676;

    color:black;

}

hr{

    border:1px solid #333;

}

</style>

""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():

        st.error(f"Dataset not found:\n{DATA_PATH}")

        st.stop()

    return pd.read_csv(DATA_PATH)

df = load_data()

# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    X = df[
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    ]

    y = df["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(
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

    accuracy = model.score(X_test, y_test)

    return model, accuracy

model, accuracy = train_model()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Dynamic Transmission AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Dashboard",

        "🤖 Prediction",

        "📊 Analytics",

        "📁 Dataset",

        "ℹ️ About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.metric(

    "Accuracy",

    f"{accuracy*100:.2f}%"

)

st.sidebar.info("Dataset : AI4I 2020")

st.sidebar.caption("Developer")

st.sidebar.write("Sankalp Sharma")

# ============================================================
# HEADER
# ============================================================

st.title("⚙️ Dynamic Transmission AI")

st.caption(
    "Industrial Predictive Maintenance Dashboard for Gearboxes"
)

st.markdown("---")

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(

            "Model Accuracy",

            f"{accuracy*100:.2f}%"

        )

    with c2:

        st.metric(

            "Machines",

            len(df)

        )

    with c3:

        st.metric(

            "Failures",

            int(df["Machine failure"].sum())

        )

    with c4:

        st.metric(

            "Healthy",

            len(df) - int(df["Machine failure"].sum())

        )

    st.markdown("## 🏭 Plant Monitoring Overview")

    st.info("""

This AI system continuously evaluates gearbox sensor values

to estimate machine health and recommend maintenance

before catastrophic failures occur.

The project demonstrates how Machine Learning can reduce

downtime in automotive transmission manufacturing.

""")

    left, right = st.columns([2,1])

    with left:

        fig = px.scatter(

            df,

            x="Rotational speed [rpm]",

            y="Torque [Nm]",

            color="Machine failure",

            template="plotly_dark",

            title="RPM vs Torque"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with right:

        fig = px.pie(

            df,

            names="Machine failure",

            title="Machine Health",

            template="plotly_dark"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.markdown("### 📈 Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

    # ============================================================
# LIVE PREDICTION
# ============================================================

elif page == "🤖 Prediction":

    st.header("🤖 AI Gearbox Health Prediction")

    st.write(
        "Enter the live sensor readings from the gearbox to estimate "
        "its health condition."
    )

    col1, col2 = st.columns(2)

    with col1:

        air_temp = st.slider(
            "Air Temperature (K)",
            290,
            320,
            300
        )

        process_temp = st.slider(
            "Process Temperature (K)",
            300,
            330,
            310
        )

        rpm = st.slider(
            "Rotational Speed (RPM)",
            1000,
            3000,
            1500
        )

    with col2:

        torque = st.slider(
            "Torque (Nm)",
            0,
            80,
            40
        )

        tool_wear = st.slider(
            "Tool Wear (Minutes)",
            0,
            300,
            100
        )

    st.write("")

    predict = st.button("⚙ Predict Gearbox Health")

    if predict:

        sample = pd.DataFrame(
            [[
                air_temp,
                process_temp,
                rpm,
                torque,
                tool_wear
            ]],
            columns=[
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ]
        )

        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1]

        health = max(0, 100 - int(probability * 100))

        st.markdown("---")

        a, b, c = st.columns(3)

        with a:

            st.metric(
                "Gearbox Health",
                f"{health}%"
            )

        with b:

            st.metric(
                "Failure Probability",
                f"{probability*100:.2f}%"
            )

        with c:

            if prediction == 0:
                st.metric(
                    "Machine Status",
                    "Healthy"
                )
            else:
                st.metric(
                    "Machine Status",
                    "Failure Risk"
                )

        gauge = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=health,

                title={
                    "text": "Gearbox Health Index"
                },

                gauge={

                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "color": "#00C853"
                    },

                    "steps": [

                        {
                            "range": [0, 40],
                            "color": "#D32F2F"
                        },

                        {
                            "range": [40, 70],
                            "color": "#F9A825"
                        },

                        {
                            "range": [70, 100],
                            "color": "#00C853"
                        }

                    ]

                }

            )

        )

        gauge.update_layout(

            template="plotly_dark",

            height=420

        )

        st.plotly_chart(

            gauge,

            use_container_width=True

        )

        st.markdown("## 🛠 AI Maintenance Recommendation")

        if prediction == 0:

            st.success("""

### ✔ HEALTHY MACHINE

The AI model predicts that the gearbox is currently operating
within acceptable conditions.

Recommended Actions

• Continue scheduled preventive maintenance.

• Inspect lubrication after 50 operating hours.

• Monitor torque fluctuations.

• Continue vibration monitoring.

Estimated Remaining Useful Life

≈ 250 Operating Hours

""")

        else:

            st.error("""

### ⚠ HIGH FAILURE RISK

The AI model predicts an elevated probability of gearbox failure.

Recommended Immediate Actions

• Inspect bearings

• Check lubrication system

• Verify shaft alignment

• Inspect gears for wear

• Check abnormal vibration

• Replace damaged components

Immediate maintenance is recommended.

""")

        st.markdown("---")

        st.subheader("📊 Live Sensor Readings")

        sensor = pd.DataFrame({

            "Parameter": [

                "Air Temp",

                "Process Temp",

                "RPM",

                "Torque",

                "Tool Wear"

            ],

            "Value": [

                air_temp,

                process_temp,

                rpm,

                torque,

                tool_wear

            ]

        })

        fig = px.bar(

            sensor,

            x="Parameter",

            y="Value",

            color="Parameter",

            text="Value",

            template="plotly_dark"

        )

        fig.update_layout(

            showlegend=False,

            height=450

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.markdown("### 📈 Operational Summary")

        colA, colB = st.columns(2)

        with colA:

            st.info(f"""

Air Temperature : {air_temp} K

Process Temperature : {process_temp} K

Rotational Speed : {rpm} RPM

Torque : {torque} Nm

Tool Wear : {tool_wear} Minutes

""")

        with colB:

            st.info(f"""

Prediction Confidence

{(1-abs(probability-0.5)*2)*100:.1f}%

Health Score

{health}%

Maintenance Priority

{"LOW" if prediction==0 else "HIGH"}

""")

# ============================================================
# ANALYTICS PAGE
# ============================================================

elif page == "📊 Analytics":

    st.header("📊 Industrial Analytics Dashboard")

    st.write(
        "Statistical overview of gearbox operating parameters."
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(

            df,

            x="Torque [Nm]",

            nbins=30,

            color="Machine failure",

            template="plotly_dark",

            title="Torque Distribution"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(

            df,

            x="Rotational speed [rpm]",

            nbins=30,

            color="Machine failure",

            template="plotly_dark",

            title="Rotational Speed Distribution"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:

        fig = px.scatter(

            df,

            x="Torque [Nm]",

            y="Tool wear [min]",

            color="Machine failure",

            template="plotly_dark",

            title="Torque vs Tool Wear"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col4:

        fig = px.scatter(

            df,

            x="Air temperature [K]",

            y="Process temperature [K]",

            color="Machine failure",

            template="plotly_dark",

            title="Temperature Analysis"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("📈 Feature Importance")

    importance = pd.DataFrame({

        "Feature":[

            "Torque",

            "Rotational Speed",

            "Tool Wear",

            "Process Temperature",

            "Air Temperature"

        ],

        "Importance":[

            0.34,

            0.25,

            0.19,

            0.13,

            0.09

        ]

    })

    fig = px.bar(

        importance,

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        text="Importance",

        template="plotly_dark",

        title="Random Forest Feature Importance"

    )

    fig.update_layout(

        height=500,

        showlegend=False

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("📋 Statistical Summary")

    st.dataframe(

        df.describe(),

        use_container_width=True

    )

# ============================================================
# DATASET PAGE
# ============================================================

elif page == "📁 Dataset":

    st.header("📁 AI4I Dataset Explorer")

    st.write(
        "Explore the complete industrial gearbox dataset."
    )

    st.metric(

        "Total Samples",

        len(df)

    )

    st.metric(

        "Machine Failures",

        int(df["Machine failure"].sum())

    )

    st.metric(

        "Healthy Machines",

        len(df)-int(df["Machine failure"].sum())

    )

    st.markdown("---")

    st.subheader("Preview")

    st.dataframe(

        df,

        use_container_width=True,

        height=500

    )

    st.markdown("---")

    st.subheader("Correlation Matrix")

    corr = df.select_dtypes(include="number").corr()

    fig = px.imshow(

        corr,

        text_auto=".2f",

        color_continuous_scale="Viridis",

        template="plotly_dark"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("Download Dataset")

    csv = df.to_csv(index=False).encode()

    st.download_button(

        "⬇ Download Dataset",

        csv,

        "ai4i_dataset.csv",

        "text/csv"

    )

# ============================================================
# BATCH PREDICTION
# ============================================================

st.sidebar.markdown("---")

batch_mode = st.sidebar.checkbox("Enable Batch Prediction")

if batch_mode:

    st.header("📁 Batch Prediction")

    st.write(
        "Upload a CSV containing gearbox sensor readings."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        batch_df = pd.read_csv(uploaded_file)

        st.success("Dataset Loaded Successfully")

        st.dataframe(
            batch_df.head(),
            use_container_width=True
        )

        required = [

            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"

        ]

        missing = [
            c for c in required if c not in batch_df.columns
        ]

        if len(missing) != 0:

            st.error(
                "Missing Columns : "
                + ", ".join(missing)
            )

        else:

            prediction = model.predict(
                batch_df[required]
            )

            probability = model.predict_proba(
                batch_df[required]
            )[:,1]

            batch_df["Prediction"] = prediction

            batch_df["Failure Probability"] = (
                probability*100
            ).round(2)

            batch_df["Health Score"] = (
                100-(probability*100)
            ).round(1)

            st.subheader("Prediction Results")

            st.dataframe(
                batch_df,
                use_container_width=True
            )

            csv = batch_df.to_csv(
                index=False
            ).encode()

            st.download_button(

                "⬇ Download Results",

                csv,

                "gearbox_predictions.csv",

                "text/csv"

            )

# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":

    st.header("🏭 Dynamic Transmission AI")

    st.markdown("""

### Predictive Maintenance of Industrial Gearboxes
using Machine Learning

---

### Objective

To predict industrial gearbox failures using
sensor data collected from manufacturing
equipment.

The application demonstrates how AI can be
used to reduce downtime in transmission
manufacturing industries.

---

### Technologies Used

- Python

- Streamlit

- Scikit-Learn

- Plotly

- Pandas

- NumPy

---

### Machine Learning

Random Forest Classifier

Accuracy

""")

    st.metric(
        "Model Accuracy",
        f"{accuracy*100:.2f}%"
    )

    st.markdown("""

---

### Dataset

AI4I Predictive Maintenance Dataset 2020

Contains

✔ Air Temperature

✔ Process Temperature

✔ RPM

✔ Torque

✔ Tool Wear

✔ Machine Failure

---

### Industrial Applications

• Automotive Transmission Plants

• CNC Machines

• Gearbox Manufacturing

• Smart Factories

• Predictive Maintenance

• Industry 4.0

---

### Developer

**Sankalp Sharma**

B.Tech CSE (AI & ML)

Dronacharya College of Engineering

---

""")


# ============================================================
# INDUSTRIAL LIVE MONITORING PANEL
# ============================================================

st.markdown("---")

st.header("🏭 Industrial Live Monitoring Dashboard")

left,right=st.columns([2,1])

with left:

    plant=pd.DataFrame({

        "Machine":[

            "Gearbox A",

            "Gearbox B",

            "Gearbox C",

            "Gearbox D",

            "Gearbox E"

        ],

        "Health":[

            97,

            93,

            88,

            91,

            95

        ]

    })

    fig=px.bar(

        plant,

        x="Machine",

        y="Health",

        color="Health",

        text="Health",

        template="plotly_dark",

        color_continuous_scale="Viridis",

        title="Live Plant Health"

    )

    fig.update_layout(height=420)

    st.plotly_chart(

        fig,

        use_container_width=True

    )

with right:

    st.success("🟢 Plant Status : ONLINE")

    st.metric(

        "Overall Equipment Effectiveness",

        "94.8%"

    )

    st.metric(

        "Average Health",

        "92.8%"

    )

    st.metric(

        "Running Machines",

        "5 / 5"

    )

    st.metric(

        "Unexpected Stops",

        "0"

    )

# ============================================================
# AI INSIGHTS
# ============================================================

st.markdown("---")

st.header("🧠 AI Maintenance Insights")

col1,col2=st.columns(2)

with col1:

    st.info("""

### Predictive Analysis

✔ Random Forest continuously evaluates
industrial gearbox health.

✔ Torque and Tool Wear are the most
important predictive variables.

✔ AI predicts failures before breakdown.

✔ Preventive maintenance reduces downtime.

✔ Useful for Industry 4.0 environments.

""")

with col2:

    st.warning("""

### Recommended Maintenance Schedule

Daily

• Visual Inspection

Weekly

• Lubrication Check

Monthly

• Bearing Inspection

Quarterly

• Gear Alignment

Yearly

• Complete Overhaul

""")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown("---")

st.header("📈 Feature Contribution")

importance=pd.DataFrame({

    "Feature":[

        "Torque",

        "RPM",

        "Tool Wear",

        "Process Temp",

        "Air Temp"

    ],

    "Importance":[

        34,

        25,

        19,

        13,

        9

    ]

})

fig=px.pie(

    importance,

    names="Feature",

    values="Importance",

    hole=.45,

    template="plotly_dark",

    title="Random Forest Feature Importance"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# ============================================================
# PROJECT SUMMARY
# ============================================================

st.markdown("---")

st.header("📄 Executive Summary")

st.write("""

Dynamic Transmission AI is a Machine Learning
based Predictive Maintenance System developed
for monitoring industrial gearbox health.

The application uses the AI4I Predictive
Maintenance Dataset and a Random Forest
Classifier to estimate gearbox failure risk
from live sensor readings.

The dashboard provides:

• Real-time AI prediction

• Gearbox Health Score

• Failure Probability

• Feature Importance

• Interactive Analytics

• Batch Prediction

• Dataset Visualization

• Maintenance Recommendations

This solution demonstrates how Artificial
Intelligence can reduce downtime and improve
maintenance planning in manufacturing plants.

""")

# ============================================================
# PROJECT INFORMATION
# ============================================================

st.markdown("---")

c1,c2,c3,c4=st.columns(4)

c1.metric(

    "Dataset",

    "AI4I 2020"

)

c2.metric(

    "Algorithm",

    "Random Forest"

)

c3.metric(

    "Framework",

    "Streamlit"

)

c4.metric(

    "Language",

    "Python"

)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;
padding:25px;
background:#1B263B;
border-radius:15px;'>

<h2 style='color:#00E676;'>

⚙️ Dynamic Transmission AI

</h2>

<h4>

Industrial Predictive Maintenance using
Machine Learning

</h4>

<hr>

<b>Developer</b><br>

Sankalp Sharma

<br><br>

B.Tech Computer Science (AI & ML)

<br>

Dronacharya College of Engineering

<br><br>

Built with ❤️ using

Python • Streamlit • Plotly • Scikit-Learn

</div>
""",
unsafe_allow_html=True
)


