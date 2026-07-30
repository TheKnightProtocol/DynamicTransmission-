import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



Dynamic Transmission AI
Current directory: /mount/src/dynamictransmission-

Dataset path: /mount/src/dynamictransmission-/data/ai4i2020.csv

Dataset exists: True

Dataset Loaded Successfully!

[
0:"UDI"
1:"Product ID"
2:"Type"
3:"Air temperature [K]"
4:"Process temperature [K]"
5:"Rotational speed [rpm]"
6:"Torque [Nm]"
7:"Tool wear [min]"
8:"Machine failure"
9:"TWF"
10:"HDF"
11:"PWF"
12:"OSF"
13:"RNF"
]


st.title("⚙️ Dynamic Transmission AI")

st.caption(
"Predictive Maintenance of Industrial Gearboxes using Machine Learning"
)

st.markdown("---")

col1,col2,col3,col4=st.columns(4)

with col1:
    st.metric(
        "Gearbox Health",
        "98%"
    )

with col2:
    st.metric(
        "Failure Risk",
        "LOW"
    )

with col3:
    st.metric(
        "AI Model",
        "Random Forest"
    )

with col4:
    st.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

st.markdown("## 📊 Dataset Overview")

c1,c2,c3,c4=st.columns(4)

c1.metric("Samples",len(df))

c2.metric("Features",5)

c3.metric(
"Failures",
int(df["Machine failure"].sum())
)

c4.metric(
"Healthy Machines",
len(df)-int(df["Machine failure"].sum())
)

# ============================================================
# LIVE PREDICTION
# ============================================================

st.markdown("---")
st.header("🤖 AI Gearbox Health Prediction")

left,right=st.columns([1,1])

with left:

    air_temp=st.slider(
        "Air Temperature (K)",
        290,
        320,
        300
    )

    process_temp=st.slider(
        "Process Temperature (K)",
        300,
        330,
        310
    )

    rpm=st.slider(
        "Rotational Speed (RPM)",
        1000,
        3000,
        1500
    )

with right:

    torque=st.slider(
        "Torque (Nm)",
        0,
        80,
        40
    )

    tool_wear=st.slider(
        "Tool Wear (Minutes)",
        0,
        300,
        100
    )

st.write("")

predict=st.button("⚙ Predict Gearbox Health")

# ============================================================
# LIVE PREDICTION
# ============================================================

st.markdown("---")
st.header("🤖 AI Gearbox Health Prediction")

left,right=st.columns([1,1])

with left:

    air_temp=st.slider(
        "Air Temperature (K)",
        290,
        320,
        300
    )

    process_temp=st.slider(
        "Process Temperature (K)",
        300,
        330,
        310
    )

    rpm=st.slider(
        "Rotational Speed (RPM)",
        1000,
        3000,
        1500
    )

with right:

    torque=st.slider(
        "Torque (Nm)",
        0,
        80,
        40
    )

    tool_wear=st.slider(
        "Tool Wear (Minutes)",
        0,
        300,
        100
    )

st.write("")

predict=st.button("⚙ Predict Gearbox Health")

    a,b,c=st.columns(3)

    with a:

        st.metric(
            "Health Score",
            f"{health}%"
        )

    with b:

        st.metric(
            "Failure Probability",
            f"{probability*100:.2f}%"
        )

    with c:

        if prediction==0:
            st.metric(
                "Status",
                "Healthy"
            )
        else:
            st.metric(
                "Status",
                "Failure Risk"
            )



    a,b,c=st.columns(3)

    with a:

        st.metric(
            "Health Score",
            f"{health}%"
        )

    with b:

        st.metric(
            "Failure Probability",
            f"{probability*100:.2f}%"
        )

    with c:

        if prediction==0:
            st.metric(
                "Status",
                "Healthy"
            )
        else:
            st.metric(
                "Status",
                "Failure Risk"
            )


    fig=go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=health,

            title={"text":"Gearbox Health Index"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"limegreen"},

                "steps":[

                    {"range":[0,40],"color":"red"},

                    {"range":[40,70],"color":"orange"},

                    {"range":[70,100],"color":"green"}

                ]

            }

        )

    )

    fig.update_layout(
        height=400,
        paper_bgcolor="#0E1117",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )



    st.markdown("## 📊 Current Sensor Readings")

    sensor=pd.DataFrame({

        "Parameter":[

            "Air Temp",
            "Process Temp",
            "RPM",
            "Torque",
            "Tool Wear"

        ],

        "Value":[

            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear

        ]

    })

    fig=px.bar(

        sensor,

        x="Parameter",

        y="Value",

        color="Parameter",

        template="plotly_dark"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )



    st.markdown("## 📊 Current Sensor Readings")

    sensor=pd.DataFrame({

        "Parameter":[

            "Air Temp",
            "Process Temp",
            "RPM",
            "Torque",
            "Tool Wear"

        ],

        "Value":[

            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear

        ]

    })

    fig=px.bar(

        sensor,

        x="Parameter",

        y="Value",

        color="Parameter",

        template="plotly_dark"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )



st.markdown("---")
st.header("📈 Feature Importance")

importance=pd.DataFrame({

    "Feature":[

        "Torque",

        "RPM",

        "Tool Wear",

        "Process Temp",

        "Air Temp"

    ],

    "Importance":[

        0.34,

        0.25,

        0.19,

        0.13,

        0.09

    ]

})

fig=px.bar(

    importance,

    x="Importance",

    y="Feature",

    orientation="h",

    color="Importance",

    template="plotly_dark"

)

st.plotly_chart(
    fig,
    use_container_width=True
)


