# 1. Install required dependencies
!pip install -q streamlit scikit-learn pandas numpy matplotlib seaborn joblib

import os
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# 2. Train Models and Generate Artifacts (.pkl & test_data.csv)
os.makedirs("model", exist_ok=True)

data = load_breast_cancer(as_frame=True)
df = data.frame.rename(columns={'target': 'Target'})

X = df.drop(columns=['Target'])
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit and save scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "model/scaler.pkl")

# Save test dataset
test_df = X_test.copy()
test_df["Target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)

# Train and save all 5 classification models
models = {
    "Logistic_Regression": (LogisticRegression(random_state=42, max_iter=1000), True),
    "Decision_Tree": (DecisionTreeClassifier(random_state=42), False),
    "KNN": (KNeighborsClassifier(n_neighbors=5), True),
    "Naive_Bayes": (GaussianNB(), False),
    "Random_Forest": (RandomForestClassifier(n_estimators=100, random_state=42), False)
}

for name, (model, requires_scaling) in models.items():
    X_tr = X_train_scaled if requires_scaling else X_train
    model.fit(X_tr, y_train)
    joblib.dump(model, f"model/{name}.pkl")

print("✅ Model training complete! Created model/ folder, .pkl artifacts, and test_data.csv")

# 3. Create app.py
app_code = """
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

st.set_page_config(
    page_title="Breast Cancer Diagnostics Classifier",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Breast Cancer Diagnostic — ML Classification Dashboard")
st.markdown("### Interactive evaluation and metrics demonstration trained on the UCI Dataset")
st.write("---")

MODEL_FILES = {
    "Logistic Regression": ("Logistic_Regression.pkl", True), 
    "Decision Tree": ("Decision_Tree.pkl", False),
    "K-Nearest Neighbors": ("KNN.pkl", True),
    "Naive Bayes": ("Naive_Bayes.pkl", False),
    "Random Forest": ("Random_Forest.pkl", False)
}

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

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom Test Dataset Loaded!")
else:
    if os.path.exists("test_data.csv"):
        test_data = pd.read_csv("test_data.csv")
        st.sidebar.info("Loaded default `test_data.csv`")
    else:
        st.error("Default `test_data.csv` not found! Please run training script first or upload a CSV file.")
        st.stop()

target_col = None
if "Target" in test_data.columns:
    target_col = "Target"
elif "target" in test_data.columns:
    target_col = "target"

if target_col:
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
else:
    X_test = test_data.copy()
    y_test = None

model_filename, needs_scaling = MODEL_FILES[selected_model_name]
model_path = os.path.join("model", model_filename)

scaler_path = os.path.join("model", "scaler.pkl")
if not os.path.exists(scaler_path):
    scaler_path = os.path.join("model", "standard_scaler.pkl")

if os.path.exists(model_path):
    model = joblib.load(model_path)
    
    if needs_scaling and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test.values if hasattr(X_test, "values") else X_test
    
    y_pred = model.predict(X_eval)
    
    if y_test is not None:
        y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
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

    st.write("---")
    st.subheader("🔍 Model Predictions Sample")
    pred_df = X_test.copy()
    pred_df["Predicted Label"] = y_pred
    if y_test is not None:
        pred_df["Actual Label"] = y_test.values
    
    st.dataframe(pred_df.head(10))

else:
    st.error(f"Required model file `{model_path}` not found.")
"""

with open("app.py", "w") as f:
    f.write(app_code)

print("✅ Successfully generated app.py file!")
