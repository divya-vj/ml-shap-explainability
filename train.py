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
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())
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
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # 4. Train-test split — 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain size: {X_train.shape[0]:,}")
    print(f"Test size: {X_test.shape[0]:,}")

    return X_train, X_test, y_train, y_test, label_encoders, feature_cols

# ■■ DAY 3: Train XGBoost with MLflow ■■■■■■■■■■■■■■■■■■■■
def train_model(X_train, X_test, y_train, y_test, feature_cols):
    """Train XGBoost model and track with MLflow."""

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3,
        "random_state": 42,
        "eval_metric": "auc",
        "use_label_encoder": False
    }

    mlflow.set_experiment("credit-risk-prediction")

    with mlflow.start_run(run_name="xgboost-baseline"):

        mlflow.log_params(params)

        print("\nTraining XGBoost model...")
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50
        )

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred)

        print(f"\nROC-AUC Score: {auc:.4f}")
        print(f"\nClassification Report:")
        print(report)

        # Log metrics to MLflow
        mlflow.log_metric("roc_auc", auc)

        # Log F1 scores
        lines = report.strip().split("\n")
        for line in lines:
            if "0" in line and len(line.split()) == 5:
                mlflow.log_metric("f1_class_0", float(line.split()[3]))
            if "1" in line and len(line.split()) == 5:
                mlflow.log_metric("f1_class_1", float(line.split()[3]))

        mlflow.xgboost.log_model(model, "model")

        print(f"\nMLflow run logged.")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

    return model

# ■■ DAY 4: SHAP Explainability ■■■■■■■■■■■■■■■■■■■■■■■■■■
def compute_shap(model, X_train, X_test, feature_cols):
    """Compute SHAP values for global and local explanations."""
    print("\n" + "="*60)
    print("SHAP EXPLAINABILITY")
    print("="*60)

    model.get_booster().feature_names = feature_cols
    explainer = shap.TreeExplainer(model.get_booster())


    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X_test)

    print(f"SHAP values shape: {shap_values.shape}")

    # Global Feature Importance
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_shap": mean_shap
    }).sort_values("mean_shap", ascending=False)

    print("\nGlobal Feature Importance (mean |SHAP|):")
    print(importance_df.to_string(index=False))

    # Local Explanation — Single Prediction
    idx = 0
    sample_shap = shap_values[idx]
    base_value = explainer.expected_value

    print(f"\nLocal explanation for sample {idx}:")
    print(f"Base value (average prediction): {base_value:.4f}")
    print(f"Final prediction probability: {base_value + sample_shap.sum():.4f}")

    print("\nFeature contributions:")
    for feat, val in sorted(zip(feature_cols, sample_shap),
                            key=lambda x: abs(x[1]), reverse=True):
        direction = "INCREASES risk" if val > 0 else "DECREASES risk"
        print(f"  {feat:35s}: {val:+.4f} ({direction})")

    return explainer, shap_values


def save_artifacts(model, explainer, encoders, feature_cols):
    """Save model and artifacts for serving."""
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_model.joblib")
    joblib.dump(explainer, "models/shap_explainer.joblib")
    joblib.dump(encoders, "models/label_encoders.joblib")
    joblib.dump(feature_cols, "models/feature_cols.joblib")
    print("\nArtifacts saved to models/ directory")

if __name__ == "__main__":
    df = load_and_explore()
    X_train, X_test, y_train, y_test, encoders, features = preprocess(df)
    model = train_model(X_train, X_test, y_train, y_test, features)
    explainer, shap_values = compute_shap(model, X_train, X_test, features)
    save_artifacts(model, explainer, encoders, features)