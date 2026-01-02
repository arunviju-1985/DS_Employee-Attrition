Employee attrition is a major challenge for organizations, leading to increased hiring costs, loss of talent, and reduced productivity.
This project aims to analyze employee data, identify key drivers of attrition, and build a high-performance predictive model to proactively identify employees at risk of leaving.
The solution integrates data preprocessing, exploratory data analysis (EDA), machine learning modeling, evaluation metrics, and business insights, with optional deployment via Streamlit.
Objectives
Analyze employee data to understand attrition patterns
Identify key factors influencing employee turnover
Build and evaluate predictive machine learning models
Improve recall and F1-score for attrition detection
Generate a ranked list of at-risk employees
Support HR teams with data-driven retention strategies
Data Preprocessing
Data loaded from MySQL database
Missing value checks
Label Encoding for categorical features
Feature-target separation
Stratified train-test split
Exploratory Data Analysis (EDA)
Attrition distribution analysis
Department-wise and role-wise attrition trends
Correlation analysis
Feature importance exploration
Feature Engineering
Encoding categorical variables
Feature selection experiments using RFE
Final feature set chosen based on performance
Handling Class Imbalance
Class imbalance addressed using:
class_weight='balanced'
Threshold tuning
SMOTE / SMOTENC evaluated (optional)
Model Development
Random Forest Classifier (final model)
Gradient Boosting & XGBoost evaluated for comparison
Hyperparameter tuning for optimal depth and estimators
Model Evaluation Metrics
The following metrics were used:
Accuracy
Precision
Recall
F1-Score
AUC-ROC
Confusion Matrix
Conclusion:
The project demonstrates the effectiveness of machine learning in predicting employee attrition
and supporting HR teams with data-driven decisions.
