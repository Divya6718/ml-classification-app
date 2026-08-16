import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Breast Cancer Diagnostics Classifier",
    page_icon="🔬",
    layout="wide"
)

# App Header
st.title("🔬 Breast Cancer Diagnostic — ML Classification Dashboard")
st.markdown("### Interactive evaluation and metrics demonstration trained on the UCI Dataset")
st.write("---")

# Model File Registry
MODEL_FILES = {
    "logistic regression": ("logistic_regression.pkl", True),
    "decision tree": ("decision_tree.pkl", False),
    "knn": ("knn.pkl", True),
    "naive bayes": ("naive_bayes.pkl", True),
    "random forest (Ensemble)": ("random_forest.pkl", False),
    "xgboost (Ensemble)": ("xgboost.pkl", False)
}

# -------------------------------------------------------------
# Sidebar Configuration & Data Input
# -------------------------------------------------------------
st.sidebar.header("🕹️ Controls & Settings")

selected_model_name = st.sidebar.selectbox(
    "Select ML Model:",
    list(MODEL_FILES.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV Data:", 
    type=["csv"],
    help="Upload test data CSV with matching feature columns."
)

# Load Uploaded CSV or Default Test Data
if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom Test Dataset Loaded!")
else:
    if os.path.exists("test_data.csv"):
        test_data = pd.read_csv("test_data.csv")
        st.sidebar.info("Loaded default `test_data.csv`")
    else:
        st.error("Default `test_data.csv` not found! Please run `model_training.py` first or upload a CSV file.")
        st.stop()

# Feature/Target Separation
if "target" in test_data.columns:
    X_test = test_data.drop(columns=["target"])
    y_test = test_data["target"]
else:
    X_test = test_data.copy()
    y_test = None

# Load Model & Scaler Binaries
model_filename, needs_scaling = MODEL_FILES[selected_model_name]
model_path = os.path.join("model", model_filename)
scaler_path = os.path.join("model", "standard_scaler.pkl")

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Scale test features if required by algorithm
    X_eval = scaler.transform(X_test) if needs_scaling else X_test
    
    # Generate Predictions
    y_pred = model.predict(X_eval)
    
    # -------------------------------------------------------------
    # 1. Metric Displays
    # -------------------------------------------------------------
    if y_test is not None:
        y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader(f"📈 Performance Metrics for **{selected_model_name}**")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")
        
        st.write("---")

        # -------------------------------------------------------------
        # 2. Confusion Matrix & Classification Report
        # -------------------------------------------------------------
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("📌 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", ax=ax,
                        xticklabels=["Malignant (0)", "Benign (1)"],
                        yticklabels=["Malignant (0)", "Benign (1)"])
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)
            
        with c2:
            st.subheader("📋 Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format("{:.4f}"))

    # -------------------------------------------------------------
    # 3. Model Output Table
    # -------------------------------------------------------------
    st.write("---")
    st.subheader("🔍 Model Predictions Sample")
    pred_df = X_test.copy()
    pred_df["Predicted Label"] = y_pred
    if y_test is not None:
        pred_df["Actual Label"] = y_test
    
    st.dataframe(pred_df.head(10))

else:
    st.error(f"Required model file `{model_path}` not found. Please run `model_training.py` first to generate models.")
