import streamlit as st
import pandas as pd
import joblib

# Load Logistic v3 model
model_path = "src/models/logistic_v3.pkl"
logistic_v3 = joblib.load(model_path)

# Training features (same as used during training)
features = ['Dependents', 'tenure', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
            'TenureGroup', 'TotalPerMonth']

st.title("Customer Churn Prediction - Logistic v3")

# User inputs (only 4 main)
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0, step=1.0)
dependents = st.selectbox("Dependents", ["No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Map inputs to float
dependents_map = {"No": 0.0, "Yes": 1.0}
internet_service_map = {"DSL": 0.0, "Fiber optic": 1.0, "No": 2.0}

input_dict = {
    "Dependents": dependents_map[dependents],
    "tenure": float(tenure),
    "InternetService": internet_service_map[internet_service],
    # fill other features with default numeric values
    "OnlineSecurity": 0.0,
    "OnlineBackup": 0.0,
    "DeviceProtection": 0.0,
    "TechSupport": 0.0,
    "StreamingTV": 0.0,
    "StreamingMovies": 0.0,
    "Contract": 0.0,
    "PaperlessBilling": 0.0,
    "PaymentMethod": 0.0,
    "MonthlyCharges": float(monthly_charges),
    "TenureGroup": 0.0,
    "TotalPerMonth": float(monthly_charges)  # or some default calculation
}

input_df = pd.DataFrame([input_dict], columns=features)

# Predict
pred_prob = logistic_v3.predict_proba(input_df)[:, 1][0]
pred_class = logistic_v3.predict(input_df)[0]

st.write(f"Prediction: {'Churn' if pred_class==1 else 'No Churn'}")
st.write(f"Probability of Churn: {pred_prob:.2f}")