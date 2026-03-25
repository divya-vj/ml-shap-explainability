# app.py — Credit Risk Predictor with SHAP Explainability
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #0E1117; }
[data-testid="stSidebar"] { background-color: #0D2B5E; }
[data-testid="stSidebar"] * { color: #DCE8FA !important; }
h1 { color: #1A56A8 !important; border-bottom: 2px solid #1A56A8; padding-bottom:8px; }
h2, h3 { color: #2A6ED4 !important; }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    model = joblib.load("models/xgboost_model.joblib")
    explainer = joblib.load("models/shap_explainer.joblib")
    encoders = joblib.load("models/label_encoders.joblib")
    features = joblib.load("models/feature_cols.joblib")
    return model, explainer, encoders, features

model, explainer, encoders, feature_cols = load_models()

# Sidebar
st.sidebar.title("🏦 Credit Risk Predictor")
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("1. Enter applicant details")
st.sidebar.markdown("2. Model predicts default risk")
st.sidebar.markdown("3. SHAP explains why")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance:**")
st.sidebar.metric("ROC-AUC Score", "0.95")
st.sidebar.metric("Training Samples", "26,064")
st.sidebar.metric("Features", "11")

st.title("🏦 Credit Risk Predictor")
st.markdown("*Enter applicant information. XGBoost predicts default risk. SHAP explains every decision.*")
st.markdown("---")

# Input form
st.subheader("📋 Applicant Information")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 30)
    income = st.number_input("Annual Income ($)", 1000, 500000, 50000, step=1000)
    emp_length = st.slider("Employment Length (years)", 0, 40, 5)
    home = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])

with col2:
    loan_amnt = st.number_input("Loan Amount ($)", 500, 35000, 10000, step=500)
    loan_intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL",
                                                 "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

with col3:
    int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 11.0, step=0.1)
    cred_hist = st.slider("Credit History Length (years)", 0, 30, 5)
    default_hist = st.selectbox("Previous Default on File", ["N", "Y"])

# Computed feature
loan_percent_income = loan_amnt / income if income > 0 else 0
st.info(f"Loan-to-Income Ratio: {loan_percent_income:.2%} (loan amount / annual income)")

if st.button("🔍 Predict Credit Risk", use_container_width=True):

    # Encode categoricals
    home_enc = encoders["person_home_ownership"].transform([home])[0]
    intent_enc = encoders["loan_intent"].transform([loan_intent])[0]
    grade_enc = encoders["loan_grade"].transform([loan_grade])[0]
    default_enc = encoders["cb_person_default_on_file"].transform([default_hist])[0]

    # Build feature array
    features_input = pd.DataFrame([[
        age, income, home_enc, emp_length, intent_enc,
        grade_enc, loan_amnt, int_rate, loan_percent_income,
        default_enc, cred_hist
    ]], columns=feature_cols)

    # Predict
    prob = model.predict_proba(features_input)[0][1]
    pred = model.predict(features_input)[0]

    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Default Probability", f"{prob:.1%}")
    c2.metric("Decision", "HIGH RISK" if pred == 1 else "LOW RISK")
    c3.metric("Confidence", f"{max(prob, 1-prob):.1%}")

    if pred == 1:
        st.error("🚨 HIGH RISK — This applicant has elevated default probability")
    else:
        st.success("✅ LOW RISK — This applicant appears creditworthy")

    # SHAP explanation
    st.markdown("---")
    st.subheader("🔍 SHAP Explanation — Why This Decision?")
    st.markdown("*Each bar shows how much each feature contributed to this specific prediction.*")

    shap_vals = explainer.shap_values(features_input)

    # Waterfall chart
    shap_series = pd.Series(
        shap_vals[0],
        index=feature_cols
    ).sort_values(key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    colors = ["#E74C3C" if v > 0 else "#2ECC71" for v in shap_series.values]
    ax.barh(shap_series.index, shap_series.values, color=colors)

    ax.set_xlabel("SHAP Value (impact on prediction)", color="white")
    ax.set_title("Feature Contributions to This Prediction\nRed = increases default risk | Green = decreases default risk",
                 color="white", pad=15)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(x=0, color="white", linewidth=0.5)

    st.pyplot(fig)
    plt.close()

    # Feature values table
    st.subheader("📊 Feature Impact Summary")
    impact_df = pd.DataFrame({
        "Feature": feature_cols,
        "Your Value": features_input.values[0],
        "SHAP Impact": shap_vals[0],
        "Direction": ["↑ Increases Risk" if v > 0 else "↓ Decreases Risk" for v in shap_vals[0]]
    }).sort_values("SHAP Impact", key=abs, ascending=False)

    impact_df["SHAP Impact"] = impact_df["SHAP Impact"].round(4)
    st.dataframe(impact_df, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555; font-size:12px;">
Credit Risk Predictor · XGBoost · SHAP · MLflow · Built by Divya VJ
</div>
""", unsafe_allow_html=True)