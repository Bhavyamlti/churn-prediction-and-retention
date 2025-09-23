# train_models_v1.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# ===============================
# Load original dataset
# ===============================
data_path = "C:/Users/T8569/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv"
data = pd.read_csv(data_path)

# Target
target = 'Churn'
y = data[target].map({'Yes':1, 'No':0})

# Features to use (subset, like in PCA code)
features = ['Dependents', 'tenure', 'InternetService', 'OnlineSecurity', 
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 
            'StreamingTV', 'StreamingMovies', 'Contract', 
            'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges']

X = data[features]

# ===============================
# Preprocessing
# ===============================
# Identify categorical and numerical features
categorical_features = X.select_dtypes(include='object').columns.tolist()
numerical_features = X.select_dtypes(include=['int64','float64']).columns.tolist()

# Column transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# ===============================
# Train/test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# ===============================
# Apply preprocessing and PCA
# ===============================
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('pca', PCA(n_components=min(13, len(numerical_features) + len(categorical_features)-1)))  # keep 13 as in previous PCA
])

X_train_pca = pipeline.fit_transform(X_train)
X_test_pca = pipeline.transform(X_test)

# ===============================
# Models dictionary
# ===============================
models = {
    'logistic_v1': LogisticRegression(random_state=42, max_iter=1000),
    'decision_tree_v1': DecisionTreeClassifier(random_state=42),
    'random_forest_v1': RandomForestClassifier(random_state=42, n_estimators=100),
    'xgboost_v1': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'catboost_v1': CatBoostClassifier(verbose=False, random_state=42)
}

# ===============================
# Train, evaluate and save models
# ===============================
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_pca, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_pca)
    acc = round(accuracy_score(y_test, y_pred),4)
    prec = round(precision_score(y_test, y_pred),4)
    rec = round(recall_score(y_test, y_pred),4)
    roc = round(roc_auc_score(y_test, y_pred),4)
    
    print(f"{name} -> Accuracy: {acc}, Precision: {prec}, Recall: {rec}, ROC_AUC: {roc}")
    
    # Save model
    model_path = f"models/{name}.pkl"
    joblib.dump({'model': model, 'pipeline': pipeline}, model_path)
    print(f"✅ Saved {name} with pipeline at {model_path}\n")

print("All v1 models trained and saved successfully.")