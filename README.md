# ⚙️ Predictive Maintenance of Industrial Gearboxes using Machine Learning

<div align="center">

### Intelligent Predictive Maintenance Platform for Industry 4.0 Manufacturing

*Leveraging Machine Learning, Industrial IoT, and Data Analytics to Predict Gearbox Failures Before They Occur.*

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-teal?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

---

### Designed for Industrial Manufacturing • AI Engineering • Predictive Analytics • Industry 4.0

</div>

---

# 📌 Overview

Industrial gearboxes are among the most critical components in manufacturing systems. Unexpected gearbox failures can result in unplanned production downtime, costly repairs, equipment damage, and significant financial losses.

This project presents an end-to-end Artificial Intelligence solution that continuously analyzes industrial sensor data to estimate gearbox health, predict potential failures, and recommend preventive maintenance before catastrophic breakdowns occur.

The solution combines Machine Learning, Industrial IoT concepts, Explainable AI, and interactive analytics into a production-oriented predictive maintenance platform.

---

# 🎯 Business Problem

Traditional maintenance strategies are often:

- Reactive (repair after failure)
- Time-based (scheduled maintenance regardless of equipment condition)

These approaches can lead to:

- Expensive emergency repairs
- Unplanned production interruptions
- Reduced equipment lifespan
- Increased operational costs

This project introduces a data-driven predictive maintenance workflow that enables maintenance teams to act before failures occur.

---

# 🚀 Objectives

- Develop an intelligent gearbox health monitoring system.
- Predict gearbox failures using machine learning.
- Estimate machine health based on sensor telemetry.
- Detect abnormal operating conditions.
- Recommend preventive maintenance actions.
- Visualize equipment performance through interactive dashboards.
- Demonstrate a production-style AI workflow for Industry 4.0 environments.

---

# 🏭 Industrial Applications

This solution is applicable across industries that rely on rotating machinery, including:

- Automotive Manufacturing
- Transmission & Gearbox Manufacturing
- Steel Plants
- Cement Plants
- Mining Operations
- Food Processing
- Pharmaceutical Manufacturing
- Packaging Industry
- Material Handling Systems
- Power Generation
- Oil & Gas
- Heavy Engineering

---

# ⚡ Key Features

## Machine Learning

- Predictive Failure Classification
- Remaining Useful Life (RUL) Estimation
- Ensemble Learning
- Automated Model Selection
- Hyperparameter Optimization
- Cross Validation
- Explainable AI (SHAP)

---

## Industrial Analytics

- Gearbox Health Monitoring
- Bearing Condition Analysis
- Vibration Trend Analysis
- Temperature Monitoring
- Lubrication Health Assessment
- Power Consumption Analysis
- Runtime Monitoring

---

## Dashboard

Interactive Streamlit dashboard including:

- Live Predictions
- Machine Health Score
- Sensor Analytics
- Historical Trends
- Failure Probability
- Feature Importance
- Maintenance Recommendations

---

## REST API

FastAPI endpoints for

- Prediction
- Health Monitoring
- Batch Inference
- Model Metadata

---

# 🧠 Machine Learning Pipeline

```
Industrial Sensors
        │
        ▼
Data Acquisition
        │
        ▼
Data Validation
        │
        ▼
Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Prediction Engine
        │
        ▼
Maintenance Recommendation
        │
        ▼
Dashboard & API
```

---

# 📊 Sensor Parameters

The predictive model utilizes multiple industrial sensor measurements including:

| Sensor | Description |
|---------|-------------|
| Temperature | Gearbox Operating Temperature |
| Oil Temperature | Lubrication Condition |
| Oil Pressure | Hydraulic Performance |
| RPM | Rotational Speed |
| Torque | Mechanical Load |
| Motor Current | Electrical Consumption |
| Motor Voltage | Electrical Supply |
| Vibration X | Horizontal Vibration |
| Vibration Y | Vertical Vibration |
| Vibration Z | Axial Vibration |
| Runtime Hours | Operating Duration |
| Bearing Health | Bearing Condition |
| Lubricant Level | Oil Availability |
| Lubricant Viscosity | Lubrication Quality |
| Gear Wear Score | Mechanical Wear |
| Humidity | Environmental Conditions |
| Load Percentage | Operational Load |
| Noise Level | Acoustic Monitoring |
| Power Consumption | Energy Usage |
| Efficiency | Machine Efficiency |

---

# 🧪 Machine Learning Models

The project compares multiple algorithms before selecting the optimal model.

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Extra Trees
- Support Vector Machine
- XGBoost
- LightGBM
- CatBoost
- Voting Ensemble

---

# 📈 Evaluation Metrics

The following metrics are computed for every model.

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Cross Validation Score
- Feature Importance
- SHAP Explainability

---

# 📂 Repository Structure

```
Predictive-Gearbox-Maintenance/

│
├── app/
│   ├── app.py
│   └── prediction_api.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_dataset.csv
│
├── models/
│   ├── gearbox_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   ├── metrics.json
│   └── final_report.pdf
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── dashboard.py
│   ├── logger.py
│   ├── config.py
│   └── utils.py
│
├── tests/
│
├── requirements.txt
├── README.md
├── LICENSE
└── main.py
```

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Predictive-Gearbox-Maintenance.git
```

Move into the project

```bash
cd Predictive-Gearbox-Maintenance
```

Create virtual environment

```bash
python -m venv .venv
```

Activate environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Training

```bash
python src/train.py
```

---

# 📊 Launch Dashboard

```bash
streamlit run app/app.py
```

---

# 🚀 Start REST API

```bash
uvicorn app.prediction_api:app --reload
```

---

# 📷 Dashboard Preview

```
+-------------------------------------------------------------+
|                    Machine Health Dashboard                 |
+-------------------------------------------------------------+
| Health Score                96 %                            |
| Failure Probability         4 %                             |
| Remaining Useful Life       420 Hours                       |
| Maintenance Status          Normal                          |
+-------------------------------------------------------------+

Trend Charts

✔ Temperature

✔ Oil Pressure

✔ RPM

✔ Vibration

✔ Torque

✔ Feature Importance

✔ Failure Probability
```

---

# 🔧 Maintenance Recommendation Engine

Example output

```
Machine ID

GBX-204

Failure Probability

91%

Risk Level

Critical

Remaining Useful Life

27 Hours

Recommendation

• Replace bearing immediately

• Inspect lubrication system

• Check shaft alignment

• Schedule maintenance within 24 hours
```

---

# 📊 Expected Performance

| Metric | Score |
|---------|-------|
| Accuracy | > 93% |
| Precision | > 92% |
| Recall | > 91% |
| ROC-AUC | > 95% |

*Performance may vary depending on dataset characteristics and model configuration.*

---

# 🔮 Future Enhancements

- Real-time IoT Sensor Integration
- MQTT Streaming Pipeline
- Edge AI Deployment
- Digital Twin Simulation
- Deep Learning (LSTM)
- Predictive Remaining Useful Life Models
- Kubernetes Deployment
- CI/CD Pipeline
- Cloud Deployment on AWS/Azure
- Grafana Monitoring
- MLflow Experiment Tracking

---

# 🛠 Technology Stack

### Programming

- Python

### Machine Learning

- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost

### Data Processing

- NumPy
- Pandas

### Visualization

- Matplotlib
- Plotly

### Dashboard

- Streamlit

### API

- FastAPI

### Explainability

- SHAP

### Deployment

- Docker

---

# 🤝 Contributing

Contributions are welcome.

Please fork the repository, create a feature branch, and submit a pull request following standard GitHub contribution guidelines.

---

# 📄 License

This project is released under the MIT License.

---

# 👨‍💻 Author

**Sankalp Sharma**

Artificial Intelligence & Machine Learning Engineer

Interested in:

- Machine Learning
- Industrial AI
- Predictive Analytics
- Industry 4.0
- Computer Vision
- MLOps
- Intelligent Manufacturing

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star.

*"Turning Industrial Data into Intelligent Decisions."*

</div>
