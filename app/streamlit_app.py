import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =========================
# Load Artifacts
# =========================
MODELS_DIR = Path("models")
model = joblib.load(MODELS_DIR / "churn_model.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
feature_list = joblib.load(MODELS_DIR / "model_features.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction System")

st.write("Fill customer details to predict if they may churn.")

# =========================
# Form Layout
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.number_input("Age", 18, 100, 30)
    Senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    Referred = st.selectbox("Referred a Friend", ["Yes", "No"])
    Referrals = st.number_input("Number of Referrals", 0, 20, 0)

with col2:
    Tenure = st.number_input("Tenure in Months", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])
    InternetService = st.selectbox("Internet Service", ["Yes", "No"])
    AvgGB = st.number_input("Avg Monthly GB Download", 0, 500, 50)
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])

with col3:
    DeviceProtect = st.selectbox("Device Protection Plan", ["Yes", "No"])
    TechSupport = st.selectbox("Premium Tech Support", ["Yes", "No"])
    StreamTV = st.selectbox("Streaming TV", ["Yes", "No"])
    StreamMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
    StreamMusic = st.selectbox("Streaming Music", ["Yes", "No"])
    UnlimitedData = st.selectbox("Unlimited Data", ["Yes", "No"])
    Paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

MonthlyCharge = st.number_input("Monthly Charge", 0.0, 500.0, 50.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
TotalRefunds = st.number_input("Total Refunds", 0.0, 10000.0, 0.0)
ExtraData = st.number_input("Total Extra Data Charges", 0, 1000, 0)
LongDist = st.number_input("Total Long Distance Charges", 0.0, 2000.0, 0.0)
Satisfaction = st.selectbox("Satisfaction Score (1-5)", [1,2,3,4,5])
CLTV = st.number_input("CLTV", 0, 10000, 5000)

Contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
Payment = st.selectbox("Payment Method", ["Credit Card", "Mailed Check", "Bank Withdrawal"])

predict_btn = st.button("Predict Customer Churn ✅")

# =========================
# Predict
# =========================
if predict_btn:

    def yesno(val): return 1 if val == "Yes" else 0

    raw = {
        "Gender": 1 if Gender=="Male" else 0,
        "Age": Age,
        "Senior Citizen": yesno(Senior),
        "Married": yesno(Married),
        "Dependents": yesno(Dependents),
        "Referred a Friend": yesno(Referred),
        "Number of Referrals": Referrals,
        "Tenure in Months": Tenure,
        "Phone Service": yesno(PhoneService),
        "Avg Monthly Long Distance Charges": LongDist,
        "Multiple Lines": yesno(MultipleLines),
        "Internet Service": yesno(InternetService),
        "Avg Monthly GB Download": AvgGB,
        "Online Security": yesno(OnlineSecurity),
        "Online Backup": yesno(OnlineBackup),
        "Device Protection Plan": yesno(DeviceProtect),
        "Premium Tech Support": yesno(TechSupport),
        "Streaming TV": yesno(StreamTV),
        "Streaming Movies": yesno(StreamMovies),
        "Streaming Music": yesno(StreamMusic),
        "Unlimited Data": yesno(UnlimitedData),
        "Paperless Billing": yesno(Paperless),
        "Monthly Charge": MonthlyCharge,
        "Total Charges": TotalCharges,
        "Total Refunds": TotalRefunds,
        "Total Extra Data Charges": ExtraData,
        "Total Long Distance Charges": LongDist,
        "Satisfaction Score": Satisfaction,
        "CLTV": CLTV,
        f"Contract_One Year": 1 if Contract=="One Year" else 0,
        f"Contract_Two Year": 1 if Contract=="Two Year" else 0,
        f"Payment Method_Credit Card": 1 if Payment=="Credit Card" else 0,
        f"Payment Method_Mailed Check": 1 if Payment=="Mailed Check" else 0
    }

    # Fill missing dummy columns
    for col in feature_list:
        raw.setdefault(col, 0)

    df = pd.DataFrame([raw])[feature_list]

    X_scaled = scaler.transform(df)
    prob = model.predict_proba(X_scaled)[0][1]

    st.subheader(f"🎯 Churn Probability: **{prob*100:.2f}%**")

    if prob >= 0.70:
        st.error("⚠️ High churn risk. Customer likely to leave.\nOffer discount / loyalty points immediately.")
    elif prob >= 0.45:
        st.warning("🟡 Medium churn risk. Monitor and follow up.")
    else:
        st.success("🟢 Low churn risk. Customer is safe!")

    st.write("---")
    st.caption("Prediction logged successfully.")

    # save
    pd.DataFrame([raw]).assign(pred_prob=prob).to_csv("predictions_log.csv", mode="a", header=False)

