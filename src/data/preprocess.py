import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

PROCESSED_DIR = (r"data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
        return df

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        return df

    def scale_features(self, df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        return df

    def transform_and_save(self, df: pd.DataFrame, target_col: str, test_size=0.2):
        df = self.handle_missing(df)
        df = self.encode_categoricals(df)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        X = self.scale_features(X, numerical_cols)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Save processed data
        X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
        y_train.to_frame("target").to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
        y_test.to_frame("target").to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

        print("✅ Preprocessing done. Files saved in data/processed/")
        print("📊 Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)


if __name__ == "__main__":
    df = pd.read_csv("C:/Users/T8569/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    prep = Preprocessor()
    prep.transform_and_save(df, target_col="Churn")