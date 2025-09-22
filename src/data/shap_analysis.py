# src/data/shap_analysis.py
import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

# ------------------------------
# Paths
# ------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
models_dir = os.path.join(PROJECT_ROOT, 'models')

X_train_path = os.path.join(processed_dir, 'X_train_fe.csv')
X_test_path = os.path.join(processed_dir, 'X_test_fe.csv')

# Output directory for SHAP plots
shap_output_dir = os.path.join(processed_dir, 'shap_plots')
os.makedirs(shap_output_dir, exist_ok=True)

# ------------------------------
# Load datasets
# ------------------------------
try:
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    print(f"[INFO] Loaded datasets: X_train {X_train.shape}, X_test {X_test.shape}")
except Exception as e:
    print(f"[ERROR] Could not load datasets: {e}")
    exit()

# ------------------------------
# Load CatBoost model (v1 recall-optimized)
# ------------------------------
catboost_model_path = os.path.join(models_dir, 'catboost_v1.pkl')
try:
    model = joblib.load(catboost_model_path)
    print(f"[INFO] Loaded model from {catboost_model_path}")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    exit()

# ------------------------------
# Compute SHAP values
# ------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ------------------------------
# Summary plot
# ------------------------------
summary_plot_path = os.path.join(shap_output_dir, 'shap_summary.png')
plt.figure(figsize=(12,8))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(summary_plot_path)
plt.close()
print(f"[INFO] SHAP summary plot saved: {summary_plot_path}")

# ------------------------------
# Feature importance bar plot
# ------------------------------
bar_plot_path = os.path.join(shap_output_dir, 'shap_bar.png')
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(bar_plot_path)
plt.close()
print(f"[INFO] SHAP feature importance bar plot saved: {bar_plot_path}")

# ------------------------------
# Dependence plots for top 3 features
# ------------------------------
shap_values_df = pd.DataFrame(shap_values, columns=X_test.columns)
top_features = shap_values_df.abs().mean().sort_values(ascending=False).head(3).index.tolist()

for feature in top_features:
    dep_plot_path = os.path.join(shap_output_dir, f'shap_dependence_{feature}.png')
    plt.figure(figsize=(8,6))
    shap.dependence_plot(feature, shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(dep_plot_path)
    plt.close()
    print(f"[INFO] SHAP dependence plot saved: {dep_plot_path}")

print("[INFO] SHAP analysis completed.")