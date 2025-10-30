import joblib
import numpy as np
import pandas as pd

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/model_features.pkl")

def preprocess_input(data: dict):
    """
    Convert input dict to dataframe, align columns with training features, and scale.
    """
    df = pd.DataFrame([data])
    
    # Ensure all expected columns exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_names]  # reorder columns
    
    scaled = scaler.transform(df)
    return scaled

def predict_churn(data: dict):
    """
    Predict churn probability from user input data.
    """
    processed = preprocess_input(data)
    prob = model.predict_proba(processed)[0][1]
    label = model.predict(processed)[0]
    return {
        "prediction": int(label),
        "churn_probability": float(prob)
    }

# For testing
if __name__ == "__main__":
    sample = {
        "Tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 850,
        "SeniorCitizen": 0,
        "Gender": 1,
        "InternetService_Fiber optic": 1,
        "InternetService_No": 0,
        "Contract_Month-to-month": 1,
        "Contract_One year": 0,
        "Contract_Two year": 0

    }
    
    result = predict_churn(sample)
    print(result)
