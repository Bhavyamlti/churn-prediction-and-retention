import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn.cluster import KMeans
import altair as alt
import shap

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("Customer Churn Prediction & Analysis")

# ===============================
# Load dataset
# ===============================
data_path = "C:/Users/T8569/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv"
data = pd.read_csv(data_path)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.fillna(0, inplace=True)
y = (data['Churn'] == 'Yes').astype(int)

# ===============================
# Features
# ===============================
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['Contract', 'InternetService', 'Dependents', 'PhoneService', 'MultipleLines',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV','StreamingMovies', 'PaymentMethod']

# ===============================
# Load PCA + Logistic pipeline
# ===============================
pipeline = joblib.load("src/models/logistic_v3.pkl")

# ===============================
# KMeans clustering
# ===============================
# For simplicity, scale + PCA first
X_cat = pipeline['encoder'].transform(data[['Contract','InternetService']])
X_num = data[['tenure','MonthlyCharges']].values
X_processed = np.hstack([X_num, X_cat])
X_scaled = pipeline['scaler'].transform(X_processed)
X_pca = pipeline['pca'].transform(X_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)
data['Cluster'] = clusters
cluster_map = {0:"Low-value / Price-sensitive",
               1:"Medium-value / At-risk",
               2:"High-value / Premium"}

# ===============================
# Kaplan-Meier Retention
# ===============================
kmf = KaplanMeierFitter()
kmf.fit(data['tenure'], event_observed=y)

km_fig, km_ax = plt.subplots(figsize=(6,4))
kmf.plot_survival_function(ax=km_ax)
km_ax.set_xlabel("Tenure (months)")
km_ax.set_ylabel("Retention Probability")
km_ax.set_title("Customer Retention / Survival Curve")
st.subheader("Overall Retention Curve")
st.pyplot(km_fig)

# ===============================
# User Input
# ===============================
st.subheader("Predict Churn for a New Customer")
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", 0, 100, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 1000.0, 70.0)
    contract = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

# ===============================
# Prepare user input for prediction
# ===============================
user_input = pd.DataFrame({
    'tenure':[tenure],
    'MonthlyCharges':[monthly_charges],
    'Contract':[contract],
    'InternetService':[internet_service]
})

# PCA + Logistic preprocessing
user_cat = pipeline['encoder'].transform(user_input[['Contract','InternetService']])
user_num = user_input[['tenure','MonthlyCharges']].values
user_processed = np.hstack([user_num, user_cat])
user_scaled = pipeline['scaler'].transform(user_processed)
user_pca = pipeline['pca'].transform(user_scaled)

# Predict
churn_prob = pipeline['logistic'].predict_proba(user_pca)[0][1]
km_prob = kmf.survival_function_at_times(tenure).values[0]
user_cluster = kmeans.predict(user_pca)[0]

# ===============================
# Display Metrics
# ===============================
st.subheader("Predicted Metrics")
m1, m2, m3 = st.columns(3)

if churn_prob < 0.3:
    m1.success(f"{churn_prob:.2%} (Low Risk)")
elif churn_prob < 0.7:
    m1.warning(f"{churn_prob:.2%} (Medium Risk)")
else:
    m1.error(f"{churn_prob:.2%} (High Risk)")

m2.metric("Estimated Retention", f"{km_prob:.2%}")
m3.metric("Customer Segment", cluster_map[user_cluster])

# ===============================
# SHAP explanation
# ===============================
st.subheader("Possible Reasons for Churn")
explainer = shap.LinearExplainer(pipeline['logistic'], X_pca)
shap_values = explainer.shap_values(user_pca)[0]

feature_names = np.array(['tenure','MonthlyCharges'] + list(pipeline['encoder'].get_feature_names_out()))
importance = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)

for f, val in importance[:5]:
    if val > 0:
        st.write(f"- {f} **increases churn risk**")
    else:
        st.write(f"- {f} **reduces churn risk**")

# ===============================
# User vs Cluster Comparison
# ===============================
st.subheader("User vs Segment Comparison")
cluster_data = data[data['Cluster'] == user_cluster]

if not cluster_data.empty:
    cluster_avg = cluster_data[['tenure','MonthlyCharges']].mean().reset_index()
    cluster_avg.columns = ['Feature','Average']

    user_values = pd.DataFrame({
        'Feature':['tenure','MonthlyCharges'],
        'Average':[tenure, monthly_charges]
    })

    chart = alt.Chart(cluster_avg).mark_bar(color='lightblue').encode(
        x='Feature', y='Average'
    ) + alt.Chart(user_values).mark_bar(color='orange').encode(
        x='Feature', y='Average'
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No cluster data available for comparison.")