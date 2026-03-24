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

if __name__ == "__main__":
    df = load_and_explore()