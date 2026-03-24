# train.py — Credit Risk Prediction with XGBoost + SHAP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import shap
import mlflow
import mlflow.xgboost
import joblib
import os

# ■■ DAY 1: Load and Explore Data ■■■■■■■■■■■■■■■■■■■■■■■■
def load_and_explore(filepath="data/credit_risk_dataset.csv"):
    """Load dataset and print exploration summary."""
    print("="*60)
    print("CREDIT RISK DATASET — EXPLORATION")
    print("="*60)

    df = pd.read_csv(filepath)

    print(f"\nShape: {df.shape}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names and types:")
    print(df.dtypes)

    print("\nFirst 3 rows:")
    print(df.head(3))

    print("\nMissing values:")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0])

    print("\nTarget distribution (loan_status):")
    print(df["loan_status"].value_counts())
    print(df["loan_status"].value_counts(normalize=True).round(3))

    print("\nNumerical summary:")
    print(df.describe().round(2))

    return df

# ■■ DAY 2: Preprocessing ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def preprocess(df):
    """Clean and prepare data for training."""
    print("\n" + "="*60)
    print("PREPROCESSING")
    print("="*60)

    df = df.copy()

    # 1. Handle missing values
    df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)
    df["loan_int_rate"].fillna(df["loan_int_rate"].median(), inplace=True)
    print(f"Nulls after filling: {df.isnull().sum().sum()}")

    # 2. Encode categorical columns
    categorical_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file"
    ]

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"Encoded {col}: {list(le.classes_)}")

    # 3. Define features and target
    feature_cols = [
        "person_age", "person_income", "person_home_ownership",
        "person_emp_length", "loan_intent", "loan_grade",
        "loan_amnt", "loan_int_rate", "loan_percent_income",
        "cb_person_default_on_file", "cb_person_cred_hist_length"
    ]

    X = df[feature_cols]
    y = df["loan_status"]

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain size: {X_train.shape[0]:,}")
    print(f"Test size: {X_test.shape[0]:,}")

    return X_train, X_test, y_train, y_test, label_encoders, feature_cols

if __name__ == "__main__":
    df = load_and_explore()