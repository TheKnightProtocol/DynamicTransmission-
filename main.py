from pathlib import Path
import joblib

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()

    if not SCALER_PATH.exists():
        st.error(f"Scaler not found: {SCALER_PATH}")
        st.stop()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler
