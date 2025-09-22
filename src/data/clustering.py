# src/data/clustering.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import numpy as np
import os

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
processed_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'X_train_fe.csv')
output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(output_dir, exist_ok=True)

# Load dataset
try:
    X = pd.read_csv(processed_path)
    print(f"[INFO] Dataset loaded with shape {X.shape}")
except Exception as e:
    print(f"[ERROR] Could not load dataset: {e}")
    exit()

# Numeric encoding for Contract if exists
if 'Contract' in X.columns:
    X['Contract_Num'] = X['Contract'].map({'Month-to-month':0, 'One year':1, 'Two year':2})

# Active services count
service_cols = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
existing_service_cols = [col for col in service_cols if col in X.columns]
for col in existing_service_cols:
    X[col] = X[col].replace(['No', 'No internet service'], 0)
    X[col] = X[col].replace(['DSL', 'Fiber optic', 'Yes'], 1)
if existing_service_cols:
    X['Active_Services'] = X[existing_service_cols].sum(axis=1)

# Main numeric features for clustering
cluster_features = [col for col in ['tenure','MonthlyCharges'] if col in X.columns]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_cluster = pd.DataFrame(imputer.fit_transform(X[cluster_features]), columns=cluster_features)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
X['Cluster'] = kmeans.fit_predict(X_cluster)

# Define plots
plots = [
    ('Tenure vs MonthlyCharges', 'tenure', 'MonthlyCharges'),
    ('Tenure vs Contract', 'tenure', 'Contract_Num'),
    ('MonthlyCharges vs Active Services', 'MonthlyCharges', 'Active_Services')
]

# Make plots
for title, x_col, y_col in plots:
    if x_col in X.columns and y_col in X.columns:
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=X, x=x_col, y=y_col, hue='Cluster', palette='Set2', s=60)
        plt.title(f'Customer Clusters: {title}')
        plt.legend(title='Cluster')
        save_path = os.path.join(output_dir, f'clustering_{x_col}_{y_col}.png')
        plt.savefig(save_path)
        print(f"[INFO] Saved plot: {save_path}")
        plt.close()
    else:
        print(f"[WARN] Skipping plot '{title}' because columns are missing.")

print("[INFO] Clustering completed and plots saved.")