a. Problem StatementThe goal of this project is to implement, evaluate, and deploy a machine learning classification pipeline for predicting breast cancer diagnosis (Malignant vs. Benign) based on features extracted from digitized fine needle aspirate (FNA) images of breast masses.Early and accurate diagnosis plays a critical role in clinical decisions. This pipeline compares multiple standard and ensemble machine learning classification models on key performance metrics and provides an interactive web interface for real-time model evaluation and predictions.
b. Dataset DescriptionDataset Name: UCI Breast Cancer Wisconsin (Diagnostic) DatasetSource: UCI Machine Learning Repository / sklearn.datasetsInstance Count: 569 instances (Exceeds assignment minimum of 500)Feature Count: 30 numerical features (Exceeds assignment minimum of 12)Target Class:0 = Malignant (212 instances)1 = Benign (357 instances)
Features BreakdownFeatures are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image:
Cell Features (10 core metrics): radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.
Computed Statistics: For each of the 10 core metrics, three values are provided:Mean (e.g., mean radius, mean texture)Standard Error (SE) (e.g., radius error, texture error)Worst / Largest Value (e.g., worst radius, worst texture)Total Features: $10 \text{ core features} \times 3 \text{ statistics} = 30 \text{ input features}$.
d. Models UsedSix classification models (including baseline algorithms and ensembles) were trained on an 80-20 train-test split. The computed evaluation metrics on the test dataset (test_data.csv) are detailed below
ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.9825	0.9974	0.973	1	0.9863	0.9628
Decision Tree	0.9386	0.9358	0.9459	0.9595	0.9526	0.8675
kNN	0.9649	0.9878	0.96	0.9865	0.973	0.9242
Naive Bayes	0.9386	0.9874	0.9342	0.973	0.953	0.8667
Random Forest (Ensemble)	0.9561	0.9924	0.9474	0.9865	0.9664	0.9054
XGBoost (Ensemble)	0.9649	0.9944	0.96	0.9865	0.973	0.9242
Model Performance Observations	
	
ML Model Name	Observation about model performance
Logistic Regression	Achieves the highest accuracy (98.25%) and F1-Score (0.9863) on this dataset. Standardization allows it to construct an optimal linear decision boundary with 100% recall (zero false negatives for benign cases).
Decision Tree	Captures non-linear decision splits effectively, but exhibits higher variance compared to tree ensembles, leading to lower overall generalization metrics (Accuracy: 93.86%, MCC: 0.8675).
kNN	Performs strongly after standard scaling (Accuracy: 96.49%, F1: 0.9730), leveraging distance proximity between feature clusters in high-dimensional space.
Naive Bayes	Provides a solid probabilistic baseline (AUC: 0.9874), though strong collinearity between physical cell features (e.g., radius, area, perimeter) slightly violates feature independence assumptions.
Random Forest (Ensemble)	Combines multiple decision trees via bootstrap aggregation (bagging) to effectively reduce tree variance, achieving an excellent ROC-AUC score of 0.9924.
XGBoost (Ensemble)	Sequentially minimizes residual errors via gradient boosting, producing high ROC-AUC (0.9944) and matching kNN in accuracy (96.49%) and MCC (0.9242).
Overall Winner for your dataset?	Logistic Regression — Achieved the overall top performance across Accuracy (0.9825), AUC (0.9974), Recall (1.0000), F1-Score (0.9863), and MCC (0.9628) due to clean linear separability in the standardized high-dimensional feature space.


