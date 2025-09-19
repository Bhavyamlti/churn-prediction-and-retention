# src/models/train.py
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def train_and_save_models(X_train, y_train, X_test, y_test):
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
        "svm": SVC(probability=True),
        "xgb": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    }

    best_acc = 0
    best_model = None
    best_name = None
    results = {}

    for name, model in models.items():
        print(f"\n🚀 Training {name.upper()}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(classification_report(y_test, y_pred))

        # Save as v1
        model_path = os.path.join(MODEL_DIR, f"{name}_v1.pkl")
        joblib.dump(model, model_path)
        print(f"💾 Saved {name}_v1.pkl")

        # Track best
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    # Save best model
    best_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model, best_path)
    print(f"\n🏆 Best Model: {best_name.upper()} with acc={best_acc:.4f}")
    print(f"💾 Saved best_model.pkl")

    return results

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    results = train_and_save_models(X_train, y_train, X_test, y_test)
    print("\n✅ Training complete. Results:", results)