import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# Load Dataset
# ----------------------------

DATA_PATH = "data/ai4i2020.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

# ----------------------------
# Remove Unnecessary Columns
# ----------------------------

drop_cols = []

for col in ["UDI", "Product ID", "Type"]:

    if col in df.columns:
        drop_cols.append(col)

df = df.drop(columns=drop_cols)

# ----------------------------
# Target Column
# ----------------------------

TARGET = "Machine failure"

X = df.drop(columns=[TARGET])

y = df[TARGET]

# ----------------------------
# Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------
# Scale
# ----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Model
# ----------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)

print()

print("Accuracy:", accuracy_score(y_test, pred))

print()

print(classification_report(y_test, pred))

# ----------------------------
# Save
# ----------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/random_forest.pkl")

joblib.dump(scaler, "models/scaler.pkl")

print()

print("Models Saved Successfully!")
