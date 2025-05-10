# Predicting Telco Customer Churn using IBM dataset

This project applies machine learning techniques to predict customer churn using a dataset containing customer behavior and subscription details. The aim is to identify customers likely to leave a service and gain insights through model interpretability using SHAP values.

## 📊 Project Overview

The notebook performs the following tasks:

- **Data Preprocessing**
  - Categorical encoding using LabelEncoder.
  - Feature scaling using StandardScaler.
  - Dropping irrelevant or low-impact features.
  
- **Exploratory Data Analysis (EDA)**
  - Correlation analysis.
  - KDE plots for feature distribution.
  - Heatmap for multivariate correlation.

- **Model Building**
  - **Random Forest Classifier**
  - **Logistic Regression**

- **Model Evaluation**
  - Classification Report
  - Confusion Matrix
  - Accuracy, Brier Score Loss, ROC AUC Score
  - SHAP analysis for model interpretability

## 🧰 Technologies & Libraries

- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn
- SHAP

> **Note:** The file data.csv is the dataset got from Kaggle [telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
