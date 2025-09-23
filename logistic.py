import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load original dataset
data = pd.read_csv("C:/Users/T8569/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Features for UI
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Dependents']
X = data[features].copy()
y = (data['Churn'] == 'Yes').astype(int)

# Convert TotalCharges to numeric (fix empty strings)
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

# Fill missing values
X = X.fillna(0)

# Define categorical and numeric features
categorical_features = ['Dependents']
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Full pipeline: preprocessing + PCA + LogisticRegression
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=2)),
    ('logistic', LogisticRegression(random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")