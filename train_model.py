import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("ai4i2020.csv")

print(df.head())

# Remove unnecessary columns
drop_cols = ["UDI", "Product ID", "Type"]

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred))

# Save
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/random_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel Saved Successfully")
