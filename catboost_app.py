import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import altair as alt
import shap
import joblib

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("Customer Churn Prediction & Analysis")

# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv("C:/Users/T8569/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID
data = data.drop(columns=['customerID'])

# Define features
features = data.columns.drop('Churn')
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Churn')
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Fill missing numeric values
data[numeric_cols] = data[numeric_cols].fillna(0)

# -------------------------------
# Load trained CatBoost model
# -------------------------------
cat_model = CatBoostClassifier()
cat_model.load_model("models/catboost_v3.cbm")

# -------------------------------
# KMeans clustering
# -------------------------------
X_scaled = StandardScaler().fit_transform(data[features])
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters
cluster_map = {
    0: "Low-value / Price-sensitive",
    1: "Medium-value / At-risk",
    2: "High-value / Premium"
}

# -------------------------------
# Kaplan-Meier survival curve
# -------------------------------
kmf = KaplanMeierFitter()
T = data['tenure']
E = (data['Churn'] == 'Yes').astype(int)
kmf.fit(T, event_observed=E)

km_fig, km_ax = plt.subplots(figsize=(6,4))
kmf.plot_survival_function(ax=km_ax)
km_ax.set_xlabel("Tenure (months)")
km_ax.set_ylabel("Retention Probability")
km_ax.set_title("Customer Retention / Survival Curve")
st.subheader("Overall Retention Curve")
st.pyplot(km_fig)

# -------------------------------
# User Input
# -------------------------------
st.subheader("Predict Churn for a New Customer")
input_col1, input_col2 = st.columns(2)

with input_col1:
    tenure = st.slider("Tenure (months)", 0, 100, 12)
    monthly_charges = st.slider("Monthly Charges", 0.0, 1000.0, 70.0)
    total_charges = st.slider("Total Charges", 0.0, 10000.0, tenure*monthly_charges)
with input_col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["0","1"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Build user dataframe
user_df = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [int(senior)],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# -------------------------------
# Prediction using CatBoost Pool
# -------------------------------
user_pool = Pool(user_df, cat_features=categorical_cols)
churn_prob = cat_model.predict_proba(user_pool)[0][1]

# Retention probability from KM
km_prob = kmf.survival_function_at_times(tenure).values[0]

# Cluster assignment
user_cluster = kmeans.predict(StandardScaler().fit_transform(user_df[features]))[0]

# -------------------------------
# Display metrics
# -------------------------------
st.subheader("Predicted Metrics")
metric_col1, metric_col2, metric_col3 = st.columns(3)

# Churn risk color
if churn_prob < 0.3:
    metric_col1.success(f"{churn_prob:.2%} (Low Risk)")
elif churn_prob < 0.7:
    metric_col1.warning(f"{churn_prob:.2%} (Medium Risk)")
else:
    metric_col1.error(f"{churn_prob:.2%} (High Risk)")

metric_col2.metric("Estimated Retention", f"{km_prob:.2%}")
metric_col3.metric("Customer Segment", cluster_map[user_cluster])

# -------------------------------
# SHAP explanation
# -------------------------------
st.subheader("Possible Reasons for Churn")
explainer = shap.TreeExplainer(cat_model)
shap_values = explainer.shap_values(user_pool)

shap.summary_plot(shap_values, user_df, feature_names=features, plot_type="bar", show=False)
st.pyplot(plt.gcf())

# -------------------------------
# Compare user vs cluster averages
# -------------------------------
st.subheader("User vs Segment Comparison")
cluster_data = data[data['Cluster']==user_cluster]
if not cluster_data.empty:
    cluster_avg = cluster_data[features].mean().reset_index()
    cluster_avg.columns = ['Feature', 'Average']
    user_values = pd.DataFrame({'Feature': features, 'Average': user_df.iloc[0][features].values})
    comparison_chart = alt.Chart(cluster_avg).mark_bar(color='lightblue').encode(x='Feature', y='Average') + \
                       alt.Chart(user_values).mark_bar(color='orange').encode(x='Feature', y='Average')
    st.altair_chart(comparison_chart, use_container_width=True)
else:
    st.info("No cluster data available for comparison.")