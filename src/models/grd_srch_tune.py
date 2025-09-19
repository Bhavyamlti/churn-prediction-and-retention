# src/models/tune.py
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Paths
MODEL_DIR = "src/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X_train = pd.read_csv("data/processed/X_train_fe.csv")
X_test = pd.read_csv("data/processed/X_test_fe.csv")
y_train = pd.read_csv("data/processed/y_train_fe.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test_fe.csv").squeeze()

print(f"✅ Loaded feature-engineered dataset: {X_train.shape}, {X_test.shape}")

results = {}

# -------------------------------
# Logistic Regression
# -------------------------------
param_grid_lr = {
    "penalty": ["l1", "l2"],
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "saga"],
    "max_iter": [500, 1000]
}

grid_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, scoring="f1", n_jobs=-1, verbose=1)
grid_lr.fit(X_train, y_train)

print("\n🔹 Logistic Regression Best Params:", grid_lr.best_params_)
y_pred = grid_lr.predict(X_test)
print(classification_report(y_test, y_pred))
results["Logistic_v3"] = grid_lr.best_params_

joblib.dump(grid_lr.best_estimator_, os.path.join(MODEL_DIR, "Logistic_v3.pkl"))

# -------------------------------
# Random Forest
# -------------------------------
param_dist_rf = {
    "n_estimators": [100, 200, 500],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

rand_rf = RandomizedSearchCV(RandomForestClassifier(), param_dist_rf, cv=3, n_iter=20, scoring="f1", n_jobs=-1, verbose=1)
rand_rf.fit(X_train, y_train)

print("\n🔹 Random Forest Best Params:", rand_rf.best_params_)
y_pred = rand_rf.predict(X_test)
print(classification_report(y_test, y_pred))
results["RF_v3"] = rand_rf.best_params_

joblib.dump(rand_rf.best_estimator_, os.path.join(MODEL_DIR, "RF_v3.pkl"))

# -------------------------------
# XGBoost
# -------------------------------
param_grid_xgb = {
"n_estimators": [100, 200, 300],
"max_depth": [3, 5, 7],
"learning_rate": [0.01, 0.05, 0.1],
"subsample": [0.8, 1.0],
"colsample_bytree": [0.8, 1.0]
}

grid_xgb = GridSearchCV(
XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
param_grid_xgb,
cv=3,
scoring="f1",
n_jobs=-1,
verbose=1
)
grid_xgb.fit(X_train, y_train)

print("\n🔹 XGBoost Best Params:", grid_xgb.best_params_)
y_pred = grid_xgb.predict(X_test)
print(classification_report(y_test, y_pred))
results["XGB_v3"] = grid_xgb.best_params_

joblib.dump(grid_xgb.best_estimator_, os.path.join(MODEL_DIR, "XGB_v3.pkl"))

# -------------------------------
# Final Summary
# -------------------------------
print("\n✅ Final Tuned Model Parameters (v3):")
for model, params in results.items():
    print(model, ":", params)


# Save Logistic Regression v3 model
joblib.dump(grid_lr.best_estimator_, "src/models/logistic_v3.pkl")
print("✅ Saved: src/models/logistic_v3.pkl")