import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

MODEL_FILE = "LR.pkl"
FEATURES_FILE = "features.pkl"

st.set_page_config(page_title="Customer Spending Predictor", layout="wide")

@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    features = getattr(model, "feature_names_in_", None)
    if features is None and os.path.exists(FEATURES_FILE):
        with open(FEATURES_FILE, "rb") as f:
            features = pickle.load(f)
    return model, list(features) if features is not None else None

model, feature_names = load_model()

st.title("Customer Yearly Spending Predictor 💰")
st.write("Predict a customer's **Yearly Amount Spent** using behavioral features.")

# --- Single prediction
st.header("🔹 Single Customer Prediction")
if feature_names:
    cols = st.columns(2)
    user_input = {}
    for i, feat in enumerate(feature_names):
        with cols[i % 2]:
            user_input[feat] = st.number_input(feat, 0.0, 100.0, 10.0)
    if st.button("Predict"):
        df = pd.DataFrame([user_input])
        pred = model.predict(df)[0]
        st.success(f"Predicted Yearly Amount Spent: ${pred:,.2f}")
else:
    st.info("Feature names unavailable — please use batch CSV mode below.")

# --- Batch prediction
st.header("🔹 Batch Prediction from CSV")
uploaded = st.file_uploader("Upload CSV with same feature columns", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    cols_ok = feature_names if feature_names else df.select_dtypes(include=[np.number]).columns
    missing = [c for c in cols_ok if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        preds = model.predict(df[cols_ok])
        df["Predicted_Yearly_Spend"] = preds
        st.dataframe(df.head())
        st.download_button("Download predictions", df.to_csv(index=False), "predictions.csv")

st.caption("Model file: " + MODEL_FILE)
