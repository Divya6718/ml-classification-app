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

# =====================================================================
# STEP 1: DATASET SELECTION (UCI Breast Cancer Wisconsin Diagnostic)
# =====================================================================
print("==================================================")
print("   STEP 1: LOADING UCI BREAST CANCER DATASET      ")
print("==================================================")

data = load_breast_cancer(as_frame=True)
df = data.frame

X = df.drop("target", axis=1)
y = df["target"]

print(f"Total Instances (Rows) : {df.shape[0]} (Requirement >= 500 met)")
print(f"Total Features (Cols)  : {X.shape[1]} (Requirement >= 12 met)")
print(f"Class Distribution     : {dict(y.value_counts())}\n")

# Create output folder for models
os.makedirs("model", exist_ok=True)

# =====================================================================
# STEP 2: ML CLASSIFICATION MODELS & EVALUATION METRICS
# =====================================================================
print("==================================================")
print("   STEP 2: TRAINING MODELS & COMPUTING METRICS    ")
print("==================================================")

# 1. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Scale Features (Crucial for Logistic Regression and kNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and test dataset
joblib.dump(scaler, os.path.join("model", "scaler.joblib"))
test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv("test_data.csv", index=False)
print("Saved `test_data.csv` and `scaler.joblib` successfully.\n")

# 3. Define the 5 Classification Models
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
    "kNN": (KNeighborsClassifier(n_neighbors=5), True),
    "Naive Bayes": (GaussianNB(), False),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), False)
}

results = []

# 4. Train, Evaluate, and Save Model Artifacts
for name, (model, requires_scaling) in models.items():
    X_tr = X_train_scaled if requires_scaling else X_train
    X_te = X_test_scaled if requires_scaling else X_test
    
    # Train
    model.fit(X_tr, y_train)
    
    # Save Artifact
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join("model", filename))
    
    # Evaluate
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_te)

    results.append({
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
    })

# Print Results
results_df = pd.DataFrame(results)
print("--- MODEL EVALUATION METRICS TABLE ---")
print(results_df.to_string(index=False))
print("\n")


# =====================================================================
# STEP 3: WRITE Streamlit `app.py` FILE AUTOMATICALLY
# =====================================================================
app_code = """import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

st.set_page_config(page_title="Breast Cancer Classification Dashboard", page_icon="", layout="wide")

MODEL_FILES = {
    "Logistic Regression": ("logistic_regression.joblib", True),
    "Decision Tree": ("decision_tree.joblib", False),
    "kNN": ("knn.joblib", True),
    "Naive Bayes": ("naive_bayes.joblib", False),
    "Random Forest": ("random_forest.joblib", False)
}

@st.cache_resource
def load_scaler():
    scaler_path = os.path.join("model", "scaler.joblib")
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

@st.cache_resource
def load_model(model_key):
    filename, _ = MODEL_FILES[model_key]
    model_path = os.path.join("model", filename)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

scaler = load_scaler()

st.sidebar.title("Navigation & Settings")
model_choice = st.sidebar.selectbox("Select Classification Model:", list(MODEL_FILES.keys()))
uploaded_file = st.sidebar.file_uploader("Upload CSV Test Dataset", type=["csv"])

st.title("Breast Cancer Prediction & Model Evaluation Dashboard")
st.write(f"Selected Model: **{model_choice}**")

tab1, tab2, tab3 = st.tabs(["Model Evaluation", "Batch Predictions", "Dataset Info"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
elif os.path.exists("test_data.csv"):
    data = pd.read_csv("test_data.csv")
else:
    data = None

with tab1:
    st.header(f"Performance Metrics: {model_choice}")
    if data is not None and "target" in data.columns:
        X_test = data.drop(columns=["target"])
        y_test = data["target"]
        model = load_model(model_choice)
        
        if model is not None:
            _, requires_scaling = MODEL_FILES[model_choice]
            X_input = scaler.transform(X_test) if (requires_scaling and scaler is not None) else X_test
            
            y_pred = model.predict(X_input)
            y_proba = model.predict_proba(X_input)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_input)
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            col2.metric("AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
            col3.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
            col4.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
            col5.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
            col6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")
            
            st.markdown("---")
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=["Actual Benign (0)", "Actual Malignant (1)"], columns=["Predicted Benign (0)", "Predicted Malignant (1)"])
            st.dataframe(cm_df, use_container_width=True)
    else:
        st.warning("Please upload a CSV file containing a `'target'` column.")

with tab2:
    st.header("Batch Prediction Results")
    if data is not None:
        X_input_raw = data.drop(columns=["target"]) if "target" in data.columns else data.copy()
        model = load_model(model_choice)
        if model is not None:
            _, requires_scaling = MODEL_FILES[model_choice]
            X_input_prep = scaler.transform(X_input_raw) if (requires_scaling and scaler is not None) else X_input_raw
            
            preds = model.predict(X_input_prep)
            probs = model.predict_proba(X_input_prep)[:, 1] if hasattr(model, "predict_proba") else [None]*len(preds)
            
            res = X_input_raw.copy()
            res["Predicted_Class"] = preds
            res["Predicted_Label"] = res["Predicted_Class"].map({1: "Malignant", 0: "Benign"})
            res["Malignant_Probability"] = probs
            st.dataframe(res, use_container_width=True)

with tab3:
    st.header("Dataset Summary")
    if data is not None:
        st.write(f"**Total Samples:** {data.shape[0]}")
        st.write(f"**Total Features:** {data.shape[1] - (1 if 'target' in data.columns else 0)}")
        st.dataframe(data.head(10), use_container_width=True)
"""

with open("app.py", "w") as f:
    f.write(app_code)

print("app.py` generated successfully!")
