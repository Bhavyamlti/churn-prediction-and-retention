# src/registry.py
import os

# Base path for saved models
BASE_MODEL_PATH = os.path.join("models")

# Registry of all available models
MODEL_REGISTRY = {
    "logistic_v1": os.path.join(BASE_MODEL_PATH, "logistic_v1.pkl"),
    "logistic_v2": os.path.join(BASE_MODEL_PATH, "logistic_v2.pkl"),
    "logistic_v3": os.path.join(BASE_MODEL_PATH, "logistic_v3.pkl"),
    "rf_v1": os.path.join(BASE_MODEL_PATH, "rf_v1.pkl"),
    "rf_v2": os.path.join(BASE_MODEL_PATH, "rf_v2.pkl"),
    "rf_v3": os.path.join(BASE_MODEL_PATH, "RF_v3.pkl"),
    "svm_v1": os.path.join(BASE_MODEL_PATH, "svm_v1.pkl"),
    "xgb_v1": os.path.join(BASE_MODEL_PATH, "xgb_v1.pkl"),
    "xgb_v2": os.path.join(BASE_MODEL_PATH, "xgb_v2.pkl"),
    "xgb_v3": os.path.join(BASE_MODEL_PATH, "XGB_v3.pkl"),
    "catboost_v1": os.path.join(BASE_MODEL_PATH, "catboost_v1.pkl"),  # recall-optimized
    "best_model": os.path.join(BASE_MODEL_PATH, "best_model.pkl")
}