import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def train_model():

    df = pd.read_csv("ai4i2020.csv")

    df = df.drop(columns=["UDI","Product ID","Type"])

    X = df.drop("Machine failure", axis=1)
    y = df["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, scaler

model, scaler = train_model()
