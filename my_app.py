import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import shap
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import altair as alt

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("Customer Churn Prediction & Analysis")

# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv("C:/Users/T8569/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# -------------------------------
# Features and target
# -------------------------------
features = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService']
X = data[features].copy()
y = (data['Churn'] == 'Yes').astype(int)

# Fill missing numeric values
X['tenure'] = pd.to_numeric(X['tenure'], errors='coerce')
X['MonthlyCharges'] = pd.to_numeric(X['MonthlyCharges'], errors='coerce')
X = X.fillna(0)

# -------------------------------
# Define categorical and numeric features
# -------------------------------
numeric_features = ['tenure', 'MonthlyCharges']
categorical_features = ['Contract', 'InternetService']

# -------------------------------
# Preprocessing + Logistic Regression Pipeline
# -------------------------------
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=2)),
    ('logistic', LogisticRegression(random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Fit pipeline
pipeline.fit(X_train, y_train)

# -------------------------------
# KMeans Clustering
# -------------------------------
X_scaled = preprocessor.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters

cluster_map = {
    0: "Low-value / Price-sensitive",
    1: "Medium-value / At-risk",
    2: "High-value / Premium"
}

# -------------------------------
# Kaplan-Meier Survival / Retention
# -------------------------------
kmf = KaplanMeierFitter()
T = data['tenure']
E = y
kmf.fit(T, event_observed=E)

# Plot KM curve
km_fig, km_ax = plt.subplots(figsize=(6,4))
kmf.plot_survival_function(ax=km_ax)
km_ax.set_xlabel("Tenure (months)")
km_ax.set_ylabel("Retention Probability")
km_ax.set_title("Customer Retention / Survival Curve")
st.subheader("Overall Retention Curve")
st.pyplot(km_fig)

# -------------------------------
# User Input for Prediction
# -------------------------------
st.subheader("Predict Churn for a New Customer")

input_col1, input_col2 = st.columns(2)

with input_col1:
    tenure = st.slider("Tenure (months)", 0, 100, 12)  # increased range
    monthly_charges = st.slider("Monthly Charges", 0, 1000, 70)  # increased range
with input_col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

user_df = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'Contract': [contract],
    'InternetService': [internet_service]
})

# Preprocess + PCA
user_transformed = pipeline.named_steps['pca'].transform(
    pipeline.named_steps['preprocessor'].transform(user_df)
)

# Churn probability
churn_prob = pipeline.named_steps['logistic'].predict_proba(user_transformed)[0][1]

# Retention probability from KM
km_prob = kmf.survival_function_at_times(tenure).values[0]

# Cluster assignment
user_cluster = kmeans.predict(preprocessor.transform(user_df))[0]

# -------------------------------
# Display Metrics with Color
# -------------------------------
st.subheader("Predicted Metrics")
metric_col1, metric_col2, metric_col3 = st.columns(3)

# Churn risk color coding
if churn_prob < 0.3:
    metric_col1.success(f"{churn_prob:.2%} (Low Risk)")
elif churn_prob < 0.7:
    metric_col1.warning(f"{churn_prob:.2%} (Medium Risk)")
else:
    metric_col1.error(f"{churn_prob:.2%} (High Risk)")

metric_col2.metric("Estimated Retention", f"{km_prob:.2%}")
metric_col3.metric("Customer Segment", cluster_map[user_cluster])

# -------------------------------
# SHAP Explanation as Text (using model without PCA)
# -------------------------------
st.subheader("Possible Reasons for Churn")

# Build a pipeline without PCA for SHAP explanations
logistic_no_pca = Pipeline([
    ('preprocessor', preprocessor),
    ('logistic', LogisticRegression(random_state=42))
])
logistic_no_pca.fit(X_train, y_train)

# Get feature names after preprocessing
ohe = preprocessor.named_transformers_['cat']
ohe_features = ohe.get_feature_names_out(categorical_features)
all_features = numeric_features + list(ohe_features)

# Transform input for SHAP
user_preprocessed = preprocessor.transform(user_df)
X_train_preprocessed = preprocessor.transform(X_train)

# SHAP explainer on non-PCA logistic regression
explainer = shap.LinearExplainer(logistic_no_pca.named_steps['logistic'], X_train_preprocessed)
shap_values = explainer.shap_values(user_preprocessed)[0]

# Rank features by importance
feature_importance = sorted(
    zip(all_features, shap_values),
    key=lambda x: abs(x[1]),
    reverse=True
)

# Display top churn reasons
st.write("**Top Factors Influencing This Prediction:**")
for feature, value in feature_importance[:5]:
    clean_feature = feature.replace("Contract_", "").replace("InternetService_", "")
    if value > 0:
        st.write(f"- {clean_feature} **increases churn risk**")
    else:
        st.write(f"- {clean_feature} **reduces churn risk**")


# -------------------------------
# Compare User vs Cluster Averages (safe check)
# -------------------------------
st.subheader("User vs Segment Comparison")

cluster_data = data[data['Cluster'] == user_cluster]

if not cluster_data.empty:
    cluster_avg = cluster_data[['tenure', 'MonthlyCharges']].mean().reset_index()
    cluster_avg.columns = ['Feature', 'Average']

    user_values = pd.DataFrame({
        'Feature': ['tenure', 'MonthlyCharges'],
        'Average': [tenure, monthly_charges]
    })

    comparison_chart = alt.Chart(cluster_avg).mark_bar(color='lightblue').encode(
        x='Feature', y='Average'
    ) + alt.Chart(user_values).mark_bar(color='orange').encode(
        x='Feature', y='Average'
    )

    st.altair_chart(comparison_chart, use_container_width=True)
else:
    st.info("No cluster data available for comparison.")