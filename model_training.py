import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# -------------------------------------------------------------
# 1. Dataset Choice: UCI Breast Cancer Diagnostic Dataset
# (30 Numerical Features, 569 Instances -> Exceeds 12 features & 500 instances)
# -------------------------------------------------------------
print("Loading UCI Breast Cancer Diagnostic Dataset...")
data = load_breast_cancer(as_frame=True)
df = data.frame  # DataFrame containing 30 features + target column

# Save copy of full dataset locally
df.to_csv("dataset.csv", index=False)

X = df.drop(columns=["target"])
y = df["target"]

print(f"Dataset successfully loaded: {X.shape[1]} features, {X.shape[0]} instances.")

# -------------------------------------------------------------
# 2. Train-Test Split (80% Train, 20% Test)
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Export test dataset for Streamlit app evaluation
test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv("test_data.csv", index=False)
print("Saved `test_data.csv` for Streamlit evaluation.")

# Standard Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "model/standard_scaler.pkl")

# -------------------------------------------------------------
# 3. Define the 6 ML Classification Models
# -------------------------------------------------------------
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
    "Decision Tree": (DecisionTreeClassifier(max_depth=5, random_state=42), False),
    "kNN": (KNeighborsClassifier(n_neighbors=5), True),
    "Naive Bayes": (GaussianNB(), True),
    "Random Forest (Ensemble)": (RandomForestClassifier(n_estimators=100, random_state=42), False),
    "XGBoost (Ensemble)": (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), False)
}

results = []

print("\nTraining models and evaluating metrics...")
# -------------------------------------------------------------
# 4. Model Training & Metric Calculation
# -------------------------------------------------------------
for name, (model, needs_scaling) in models.items():
    X_tr = X_train_scaled if needs_scaling else X_train
    X_te = X_test_scaled if needs_scaling else X_test
    
    # Train
    model.fit(X_tr, y_train)
    
    # Predict
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # Calculate all 6 requested metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })
    
    # Save trained model object
    file_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    joblib.dump(model, os.path.join("model", file_name))
    print(f"Saved: model/{file_name}")

# -------------------------------------------------------------
# 5. Display Evaluation Results Table
# -------------------------------------------------------------
metrics_df = pd.DataFrame(results)
print("\n========================= MODEL EVALUATION COMPARISON TABLE =========================")
print(metrics_df.to_string(index=False))
print("=====================================================================================")
