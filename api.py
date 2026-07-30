from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from io import BytesIO

# -----------------------------
# Load Model & Scaler
# -----------------------------
MODEL_PATH = "models/random_forest.pkl"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

scaler = None
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Predictive Maintenance API",
    description="Industrial Gearbox Failure Prediction using Machine Learning",
    version="1.0.0",
)

# -----------------------------
# Enable CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Input Schema
# -----------------------------
class MachineInput(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


# -----------------------------
# Utility Functions
# -----------------------------
def calculate_health_score(probability: float) -> float:
    score = (1 - probability) * 100
    return round(score, 2)


def maintenance_recommendation(probability: float) -> str:
    if probability >= 0.85:
        return "Immediate Maintenance Required"
    elif probability >= 0.60:
        return "Schedule Maintenance Soon"
    elif probability >= 0.30:
        return "Increase Monitoring Frequency"
    else:
        return "Machine Operating Normally"


def prepare_dataframe(data):
    df = pd.DataFrame([{
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear
    }])

    return df


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {
        "project": "Predictive Maintenance of Industrial Gearboxes",
        "company": "Dynamic Transmission Limited",
        "status": "API Running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: MachineInput):

    df = prepare_dataframe(data)

    X = df.values

    if scaler:
        X = scaler.transform(X)

    probability = float(model.predict_proba(X)[0][1])
    prediction = int(model.predict(X)[0])

    return {
        "prediction": prediction,
        "failure_probability": round(probability, 4),
        "health_score": calculate_health_score(probability),
        "maintenance": maintenance_recommendation(probability)
    }


@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Upload a CSV file."
        )

    content = await file.read()

    df = pd.read_csv(BytesIO(content))

    expected_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    for col in expected_columns:
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing column: {col}"
            )

    X = df[expected_columns]

    if scaler:
        X = scaler.transform(X)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    df["Prediction"] = predictions
    df["Failure Probability"] = probabilities
    df["Health Score"] = [
        calculate_health_score(p)
        for p in probabilities
    ]

    df["Maintenance"] = [
        maintenance_recommendation(p)
        for p in probabilities
    ]

    return df.to_dict(orient="records")
