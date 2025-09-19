import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_fe.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_fe.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train_fe.csv")).squeeze()
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test_fe.csv")).squeeze()

print(f"✅ Loaded feature-engineered dataset: {X_train.shape}, {X_test.shape}")

def evaluate_model(name, model, X_test, y_test):
    """Return metrics for a trained model"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n📊 {name} Performance:")
    print(classification_report(y_test, y_pred))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# -----------------------------
# Logistic Regression (v2)
# -----------------------------
log_reg = LogisticRegression(max_iter=500, solver="liblinear")
log_reg_params = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"]
}

grid_lr = GridSearchCV(log_reg, log_reg_params, cv=5, scoring="f1", n_jobs=-1, verbose=1)
grid_lr.fit(X_train, y_train)

best_lr = grid_lr.best_estimator_
metrics_lr = evaluate_model("LogisticRegression_v2", best_lr, X_test, y_test)

joblib.dump(best_lr, os.path.join(MODEL_DIR, "logistic_v2.pkl"))
print("✅ Saved logistic_v2.pkl")

# -----------------------------
# Random Forest (v2)
# -----------------------------
rf = RandomForestClassifier(random_state=42)
rf_params = {
    "n_estimators": [100, 200, 500],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10]
}

grid_rf = RandomizedSearchCV(rf, rf_params, cv=5, scoring="f1", n_jobs=-1, n_iter=5, verbose=1, random_state=42)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
metrics_rf = evaluate_model("RandomForest_v2", best_rf, X_test, y_test)

joblib.dump(best_rf, os.path.join(MODEL_DIR, "rf_v2.pkl"))
print("✅ Saved rf_v2.pkl")

# -----------------------------
# XGBoost (v2)
# -----------------------------
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0]
}

grid_xgb = RandomizedSearchCV(xgb, xgb_params, cv=5, scoring="f1", n_jobs=-1, n_iter=5, verbose=1, random_state=42)
grid_xgb.fit(X_train, y_train)

best_xgb = grid_xgb.best_estimator_
metrics_xgb = evaluate_model("XGBoost_v2", best_xgb, X_test, y_test)

joblib.dump(best_xgb, os.path.join(MODEL_DIR, "xgb_v2.pkl"))
print("✅ Saved xgb_v2.pkl")

# -----------------------------
# Compare and Save Best Model
# -----------------------------
all_results = {
    "Logistic_v2": metrics_lr,
    "RandomForest_v2": metrics_rf,
    "XGBoost_v2": metrics_xgb,
}

print("\n✅ Final Comparison (v2 tuned models):")
for model, metrics in all_results.items():
    print(f"{model}: {metrics}")

# Pick best by F1
best_model_name = max(all_results, key=lambda m: all_results[m]["f1"])
best_model_path = os.path.join(MODEL_DIR, f"{best_model_name.lower()}.pkl")

joblib.dump(joblib.load(best_model_path), os.path.join(MODEL_DIR, "best_model.pkl"))
print(f"\n🏆 Best model updated: {best_model_name} → best_model.pkl")