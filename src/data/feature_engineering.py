import os
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE

PROCESSED_DIR = "data/processed"

def load_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))["target"]
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))["target"]
    return X_train, X_test, y_train, y_test

def add_features(df):
    # Tenure groups
    if "tenure" in df.columns:
        df["TenureGroup"] = pd.cut(
            df["tenure"], bins=[0, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
        ).astype(str)

    # Total charges per month
    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        df["TotalPerMonth"] = df["MonthlyCharges"] * df["tenure"]

    # Encode any object columns (so SMOTE can run)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    return df

def balance_data(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print("Before SMOTE:", y.value_counts().to_dict())
    print("After SMOTE:", y_res.value_counts().to_dict())
    return X_res, y_res

def select_features(X, y, k=15):
    selector = SelectKBest(mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected = X.columns[selector.get_support()].tolist()
    print("Selected Features:", selected)
    return pd.DataFrame(X_new, columns=selected)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    # Add new features
    X_train = add_features(X_train)
    X_test = add_features(X_test)

    # Balance classes
    X_train, y_train = balance_data(X_train, y_train)

    # Feature selection
    X_train = select_features(X_train, y_train, k=15)
    X_test = X_test[X_train.columns]  # align features

    # Save engineered data
    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train_fe.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test_fe.csv"), index=False)
    y_train.to_frame("target").to_csv(os.path.join(PROCESSED_DIR, "y_train_fe.csv"), index=False)
    y_test.to_frame("target").to_csv(os.path.join(PROCESSED_DIR, "y_test_fe.csv"), index=False)

    print("✅ Feature engineering completed. Files saved in data/processed/")