import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

# ===============================
# Load original training dataset
# ===============================
data_path = "C:/Users/T8569/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)

# Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna(0, inplace=True)

# Features
features = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService']
X = df[features]

# Encode categorical features
categorical_cols = ['Contract','InternetService']
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_cat = encoder.fit_transform(X[categorical_cols])
X_num = X.drop(columns=categorical_cols).values
X_processed = np.hstack([X_num, X_cat])

# Target
y = df['Churn'].map({'Yes':1,'No':0}).values

# Scale and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_pca, y)

# ===============================
# Save full pipeline as logistic_v3
# ===============================
pipeline = {
    'encoder': encoder,
    'scaler': scaler,
    'pca': pca,
    'logistic': model
}

joblib.dump(pipeline, "src/models/logistic_v3.pkl")
print("Saved PCA + Logistic pipeline as logistic_v3.pkl")