# 🏦 Credit Risk Predictor with SHAP Explainability

> Predict loan default risk with XGBoost and explain every decision using SHAP values.  
> Built as part of a structured AI/ML portfolio for placement preparation — 2026.

**[🚀 Live Demo](https://ml-shap-explainability-dtyvnm9c9yngabcq7dh6e8.streamlit.app/)**

---

## 📌 What This Project Does

Banks and lenders reject loan applications every day — but regulations now require them to explain *why*.  
This project builds a complete ML system that:

1. **Predicts** whether a loan applicant will default based on their financial profile
2. **Explains** every single prediction using SHAP values — showing which features drove the decision and by how much
3. **Tracks** all experiments with MLflow so every training run is reproducible
4. **Serves** the model through an interactive Streamlit web app

---

## 🎯 Key Results

| Metric | Score |
|---|---|
| ROC-AUC | **0.95** |
| Accuracy | **93%** |
| F1 Score (Default class) | **0.82** |
| Training samples | 26,064 |
| Test samples | 6,517 |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **XGBoost** | Gradient boosted classifier for tabular data |
| **SHAP** | Local and global prediction explainability |
| **MLflow** | Experiment tracking and model versioning |
| **scikit-learn** | Preprocessing, encoding, evaluation |
| **Streamlit** | Interactive web interface |
| **joblib** | Model serialization |

---

## 📂 Project Structure
```
ml-shap-explainability/
├── data/
│   └── credit_risk_dataset.csv   # Download from Kaggle (link below)
├── models/
│   ├── xgboost_model.joblib      # Trained XGBoost model
│   ├── label_encoders.joblib     # Fitted encoders for categorical features
│   └── feature_cols.joblib       # Feature column names
├── notebooks/
│   └── 01_exploration.py         # EDA notebook
├── train.py                      # Full training pipeline
├── app.py                        # Streamlit web app
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Credit Risk Dataset** — [Download from Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

- 32,581 loan applications
- 12 features: age, income, employment length, loan amount, loan grade, etc.
- Target: `loan_status` — 0 = repaid, 1 = defaulted
- Class distribution: **78% repaid, 22% default**

---

## 💡 Why SHAP?

Standard ML models are black boxes — they give a prediction but no explanation.  
SHAP (SHapley Additive exPlanations) solves this by computing the exact contribution of each feature to each individual prediction.

**Example output for a rejected application:**
```
loan_grade          → +0.88  (INCREASES default risk)
person_income       → -0.85  (DECREASES default risk)
loan_percent_income → +0.81  (INCREASES default risk)
```

This is what regulators, managers, and customers actually need — not just a yes/no answer.

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/divya-vj/ml-shap-explainability.git
cd ml-shap-explainability
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**  
Get `credit_risk_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) and place it in the `data/` folder.

**4. Train the model**
```bash
python train.py
```

**5. Launch the app**
```bash
streamlit run app.py
```

**6. View MLflow experiments**
```bash
mlflow ui
# Opens at http://localhost:5000
```

