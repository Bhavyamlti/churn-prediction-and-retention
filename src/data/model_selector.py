# src/model_selector.py
import sys
import os
import joblib
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from .registry import MODEL_REGISTRY

def load_model(model_name: str):
    """
    Load a trained model from the MODEL_REGISTRY.
    Example: load_model("catboost_v1")
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry.")
    
    model_path = MODEL_REGISTRY[model_name]
    return joblib.load(model_path)

# Check all registered models
print("Registered models in MODEL_REGISTRY:")
for model_name in MODEL_REGISTRY.keys():
    print("-", model_name)

# Load CatBoost v1 and check recall
from src.data.registry import MODEL_REGISTRY
from sklearn.metrics import recall_score

# Load test data (make sure you have X_test, y_test ready)

X_test = pd.read_csv("data/processed/X_test_fe.csv")
y_test = pd.read_csv("data/processed/y_test_fe.csv")

# Load CatBoost v1
catboost_v1 = MODEL_REGISTRY["catboost_v1"]  # key matches registry
catboost_v1 = joblib.load(catboost_v1)

y_pred = catboost_v1.predict(X_test)
recall = recall_score(y_test, y_pred)
print(f"\nCatBoost v1 Recall on test set: {recall:.4f}")