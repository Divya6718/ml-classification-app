import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Create output directory
os.makedirs("model", exist_ok=True)

# 1. Load Dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Fit & Save Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join("model", "scaler.joblib"))

# 4. Save Test Data CSV for Streamlit Testing
test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv("test_data.csv", index=False)

# 5. Define & Train Models
models = {
    "logistic_regression.joblib": (LogisticRegression(max_iter=1000, random_state=42), X_train_scaled),
    "decision_tree.joblib": (DecisionTreeClassifier(random_state=42), X_train),
    "knn.joblib": (KNeighborsClassifier(n_neighbors=5), X_train_scaled),
    "naive_bayes.joblib": (GaussianNB(), X_train),
    "random_forest.joblib": (RandomForestClassifier(n_estimators=100, random_state=42), X_train),
}

# 6. Fit and Save Artifacts
for filename, (model, X_tr) in models.items():
    model.fit(X_tr, y_train)
    filepath = os.path.join("model", filename)
    joblib.dump(model, filepath)
    print(f"Saved: {filepath}")

print("\n  All model files saved successfully in the `model/` directory!")
