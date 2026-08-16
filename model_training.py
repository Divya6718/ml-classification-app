import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def main():
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)

    # Step 1: Generate/Load Dataset (Minimum 12 Features, 500 Instances)
    # Using a 15-feature, 1000-instance synthetic dataset for demonstration.
    # (If using Kaggle/UCI data, load it via pd.read_csv here)
    X, y = make_classification(
        n_samples=1000, 
        n_features=15, 
        n_informative=10, 
        n_redundant=5, 
        random_state=42
    )
    
    feature_names = [f"feature_{i+1}" for i in range(15)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # Train-Test Split (80% Train, 20% Test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])
    
    # Save test dataset as test_data.csv for Streamlit upload and testing
    test_df.to_csv("test_data.csv", index=False)
    print("Saved test_data.csv (20% sample with target column)")

    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    # Step 2: Define Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []

    # Step 3: Train and Evaluate Models
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # Save trained model artifact
        file_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
        joblib.dump(model, os.path.join("model", file_name))
        
        # Predictions & Probabilities
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)

        # Compute Metrics
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

    # Display Metrics Table to paste into README.md
    results_df = pd.DataFrame(results)
    print("\n--- Model Evaluation Comparison Table ---")
    print(results_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
