# src/models/tune.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score
from catboost import CatBoostClassifier


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

# ===============================
# CatBoost - Version 1 (pre-tuned for recall)
# ===============================
print("Training CatBoost v1 (recall-optimized)...")

# these params are from the notebook you found with ~91% recall
catboost_v1 = CatBoostClassifier(
    verbose=False,
    random_state=0,
    scale_pos_weight=5 # handles imbalance
)

categorical_features_indices = np.where(X_train.dtypes != float)[0]

catboost_v1.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_test, y_test)
)

# ===============================
# Evaluate
# ===============================
y_pred = catboost_v1.predict(X_test)

accuracy = round(accuracy_score(y_test, y_pred), 4)
recall = round(recall_score(y_test, y_pred), 4)
roc_auc = round(roc_auc_score(y_test, y_pred), 4)
precision = round(precision_score(y_test, y_pred), 4)

print(f"CatBoost v1 Results -> Accuracy: {accuracy}, Recall: {recall}, ROC_AUC: {roc_auc}, Precision: {precision}")

# ===============================
# Save Model
# ===============================
model_path = "models/catboost_v1.pkl"
joblib.dump(catboost_v1, model_path)
print(f"✅ CatBoost v1 saved at {model_path}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv("C:/Users/T8569/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Target
y = (data['Churn'] == 'Yes').astype(int)

# Features (all except customerID and Churn)
X = data.drop(columns=['customerID', 'Churn'])

# -------------------------------
# Encode categorical columns
# -------------------------------
categorical_cols = X.select_dtypes(include='object').columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Train CatBoost
# -------------------------------
cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    random_state=42,
    verbose=100,
    eval_metric='Recall',
    auto_class_weights='Balanced'
)

cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)

# -------------------------------
# Evaluate Performance
# -------------------------------
y_pred = cat_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

cat_model.save_model('models/catboost_v3.cbm')
