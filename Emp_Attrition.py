import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# -------------------------------------------------------------------
# PAGE TITLE
# -------------------------------------------------------------------
st.title("Employee Attrition Prediction System")
st.write("HR Analytics • Machine Learning • MySQL • Streamlit")


# -------------------------------------------------------------------
# DATABASE CONFIGURATION
# -------------------------------------------------------------------
HOST = "localhost"
USER = "root"
PASSWORD = "aaron2013"
DATABASE = "employee_attrition"
TABLE = "emp_attrition"

df = None  # initialize


# -------------------------------------------------------------------
# FETCH DATA FROM MYSQL
# -------------------------------------------------------------------
if st.button("Fetch Data From MySQL"):
    try:
        conn = mysql.connector.connect(
            host=HOST, user=USER, password=PASSWORD, database=DATABASE
        )
        query = f"SELECT * FROM {TABLE}"
        df = pd.read_sql(query, conn)
        conn.close()

        st.success("Data Loaded Successfully!")
        st.dataframe(df.head())

        # ---------------------------------------------------------------
        # DATA PREPROCESSING
        # ---------------------------------------------------------------
        st.header("Data Preprocessing")

        if "Attrition" not in df.columns:
            st.error("Column 'Attrition' not found in table!")
        else:
            df = df.copy()

            # Label Encoding all categorical columns
            label_encoders = {}
            for col in df.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

            st.success("Categorical Columns Successfully Label Encoded")
            st.dataframe(df.head())

        # ---------------------------------------------------------------
        # EDA SECTION
        # ---------------------------------------------------------------
        st.header("Exploratory Data Analysis (EDA)")

        # 1. Attrition Count
        fig1 = px.histogram(df, x="Attrition", title="Attrition Distribution")
        st.plotly_chart(fig1)

        # 2. Monthly Income vs Attrition
        if "MonthlyIncome" in df.columns:
            fig2 = px.box(df, x="Attrition", y="MonthlyIncome",
                          title="Monthly Income by Attrition")
            st.plotly_chart(fig2)

        # 3. Department-wise Attrition
        if "Department" in df.columns:
            fig3 = px.histogram(df, x="Department", color="Attrition",
                                title="Attrition by Department")
            st.plotly_chart(fig3)

        # 4. Job Satisfaction vs Attrition
        if "JobSatisfaction" in df.columns:
            fig4 = px.histogram(df, x="JobSatisfaction", color="Attrition",
                                barmode="group",
                                title="Job Satisfaction vs Attrition")
            st.plotly_chart(fig4)

        # 5. Correlation Heatmap
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.corr(), cmap="coolwarm")
        st.pyplot(plt)

        # ---------------------------------------------------------------
        # TRAIN-TEST SPLIT
        # ---------------------------------------------------------------
        st.header("Train/Test Split")

        X = df.drop("Attrition", axis=1)
        y = df["Attrition"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        st.write(f"Training Data Size: {X_train.shape}")
        st.write(f"Testing Data Size: {X_test.shape}")

        # ---------------------------------------------------------------
        # MODEL TRAINING
        # ---------------------------------------------------------------
        st.header("Model Training")

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        st.success(f"Model Trained Successfully ✔")
        st.write(f"Accuracy: {acc:.3f}")
        st.write(f"Precision: {prec:.3f}")
        st.write(f"Recall: {rec:.3f}")
        st.write(f"F1 Score: {f1:.3f}")
        st.write(f"AUC-ROC: {auc:.3f}")

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # ---------------------------------------------------------------
        # FEATURE IMPORTANCE
        # ---------------------------------------------------------------
        st.header("Feature Importance (Key Attrition Drivers)")

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importance_df)

        fig_fi = px.bar(importance_df, x="Importance", y="Feature",
                        orientation="h",
                        title="Feature Importance")
        st.plotly_chart(fig_fi)

        # ---------------------------------------------------------------
        # AT-RISK EMPLOYEES LIST
        # ---------------------------------------------------------------
        st.header("Employees at High Attrition Risk")

        df["Attrition_Prob"] = model.predict_proba(X)[:, 1]
        high_risk_df = df.sort_values(by="Attrition_Prob", ascending=False)

        st.dataframe(high_risk_df.head(20))

        # ---------------------------------------------------------------
        # DASHBOARD (3-Tabs)
        # ---------------------------------------------------------------
        st.header("HR Analytics Dashboard")

        tab1, tab2, tab3 = st.tabs(["Income", "Satisfaction", "Tenure"])

        with tab1:
            st.subheader("Monthly Income Distribution")
            st.plotly_chart(px.box(df, x="Attrition", y="MonthlyIncome"))

        with tab2:
            st.subheader("Job Satisfaction Levels")
            st.plotly_chart(px.histogram(df, x="JobSatisfaction", color="Attrition"))

        with tab3:
            st.subheader("Years at Company vs Monthly Income")
            st.plotly_chart(px.scatter(df, x="YearsAtCompany",
                                       y="MonthlyIncome",
                                       color="Attrition"))


    except Exception as e:
        st.error(f"Error: {str(e)}")
