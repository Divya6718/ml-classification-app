import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. Import all requested classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Create output folder for model artifacts
os.makedirs("model", exist_ok=True)

# 2. Load dataset (Auto-detects CSV or generates synthetic binary data)
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

if csv_files:
    dataset_file = csv_files[0]
    print(f"Loading dataset from: '{dataset_file}'")
    df = pd.read_csv(dataset_file)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
else:
    print("No CSV file found. Using synthetic classification dataset...")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# 3. Train-test split & feature scaling (important for KNN, Logistic Regression)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for future inference
joblib.dump(scaler, "model/scaler.pkl")

# 4. Define model dictionary
models = {
    "logistic_regression": LogisticRegression(random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 5. Evaluate and save models
results = []

for name, model in models.items():
    # Use scaled data for KNN and Logistic Regression
    if name in ["logistic_regression", "knn"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Calculate AUC (handles binary and multi-class)
    if y_proba is not None and len(np.unique(y)) == 2:
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = np.nan

    # Save trained model to disk
    model_filepath = f"model/{name}.pkl"
    joblib.dump(model, model_filepath)

    results.append({
        "Model": name.replace('_', ' ').title(),
        "Accuracy": acc,
        "AUC Score": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC Score": mcc
    })

# 6. Display evaluation summary
metrics_df = pd.DataFrame(results)
print("\n=== Model Evaluation Metrics Summary ===")
print(metrics_df.to_string(index=False))

print("\nSaved artifacts in 'model/':")
print(os.listdir("model"))
