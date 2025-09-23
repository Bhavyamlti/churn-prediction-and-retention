# app_pca.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# ===============================
# Load original training dataset
# ===============================
data_path = "C:/Users/T8569/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)

# Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna(0, inplace=True)

# Only a few features for PCA
features = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService']
X = df[features]

# Encode categorical features
categorical_cols = ['Contract','InternetService']
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_cat = encoder.fit_transform(X[categorical_cols])
X_num = X.drop(columns=categorical_cols).values
X_processed = np.hstack([X_num, X_cat])

y = df['Churn'].map({'Yes':1,'No':0}).values

# Scale and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Train model
model = LogisticRegression()
model.fit(X_pca, y)

# ===============================
# Streamlit UI
# ===============================
st.title("Churn Prediction with PCA (4 Inputs)")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
contract = st.selectbox("Contract Type", options=['Month-to-month','One year','Two year'])
internet_service = st.selectbox("Internet Service", options=['DSL','Fiber optic','No'])

# Convert user input to dataframe
user_input = pd.DataFrame({
    'tenure':[tenure],
    'MonthlyCharges':[monthly_charges],
    'Contract':[contract],
    'InternetService':[internet_service]
})

# Preprocess user input
user_cat = encoder.transform(user_input[categorical_cols])
user_num = user_input.drop(columns=categorical_cols).values
user_processed = np.hstack([user_num, user_cat])
user_scaled = scaler.transform(user_processed)
user_pca = pca.transform(user_scaled)

# Predict
prob = model.predict_proba(user_pca)[0][1]
pred = "Yes" if prob >= 0.5 else "No"

st.write(f"**Churn Probability:** {prob:.2f}")
st.write(f"**Predicted Churn:** {pred}")