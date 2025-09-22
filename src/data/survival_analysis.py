# src/data/survival_analysis.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# ------------------------
# Configuration / Paths
# ------------------------
DATA_PATH = r"C:/Users/T8569/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(PROJECT_ROOT, "processed", "survival_plot.png")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "processed", "survival_summary.csv")

# ------------------------
# Load dataset
# ------------------------
try:
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Dataset loaded with shape: {df.shape}")
except Exception as e:
    print(f"[ERROR] Could not load dataset: {e}")
    exit(1)

# ------------------------
# Preprocessing
# ------------------------
# Encode Churn as 0/1
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
df['tenure'] = df['tenure'].astype(float)

# ------------------------
# Survival Analysis
# ------------------------
kmf = KaplanMeierFitter()

plt.figure(figsize=(10, 6))

for churn_value, label in zip([0, 1], ["No Churn", "Churn"]):
    mask = df['Churn'] == churn_value
    kmf.fit(durations=df.loc[mask, 'tenure'], event_observed=df.loc[mask, 'Churn'], label=label)
    kmf.plot_survival_function()

plt.title("Customer Survival Curve (Kaplan-Meier)")
plt.xlabel("Tenure (months)")
plt.ylabel("Survival Probability")
plt.grid(True)

# ------------------------
# Save outputs
# ------------------------
os.makedirs(os.path.join(PROJECT_ROOT, "processed"), exist_ok=True)
plt.savefig(OUTPUT_PNG)
print(f"[INFO] Survival plot saved to: {OUTPUT_PNG}")

# Save summary table
summary_df = kmf.survival_function_.reset_index()
summary_df.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] Survival summary saved to: {OUTPUT_CSV}")

plt.show()