# 💳 Financial Transaction Fraud Detection Using Machine Learning

## 📌 Overview
This project focuses on detecting fraudulent financial transactions using machine learning classification algorithms. It involves data preprocessing, exploratory data analysis (EDA), handling class imbalance, model building, performance evaluation, and feature-importance analysis.

The dataset contains **6,362,620 transaction records and 11 columns**, with `isFraud` as the target variable.

## 🎯 Objectives
- Analyze transaction patterns and identify fraudulent activities.
- Perform data cleaning and preprocessing.
- Explore relationships between transaction features and fraud.
- Address class imbalance using SMOTE.
- Build and compare machine learning classification models.
- Evaluate models using classification metrics.
- Identify important features contributing to fraud detection.

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- LightGBM
- Jupyter Notebook

## 📊 Dataset Description

The dataset contains financial transaction records with the following important features:

- `step` — Transaction time step, where each step represents one hour.
- `type` — Type of transaction, such as CASH-IN, CASH-OUT, DEBIT, PAYMENT, and TRANSFER.
- `amount` — Transaction amount.
- `oldbalanceOrg` — Origin account balance before the transaction.
- `newbalanceOrig` — Origin account balance after the transaction.
- `oldbalanceDest` — Destination account balance before the transaction.
- `newbalanceDest` — Destination account balance after the transaction.
- `isFraud` — Target variable indicating fraudulent transactions.
- `isFlaggedFraud` — Indicates whether a transaction was flagged by the existing fraud-detection system.

## 🔄 Project Workflow

1. **Data Inspection:** Examined dataset structure, data types, and descriptive statistics.
2. **Data Preprocessing:** Checked missing values and outliers and prepared features for modeling.
3. **Exploratory Data Analysis:** Investigated fraud distribution, transaction types, account balances, and transaction amounts.
4. **Feature Engineering:** Analyzed account balance discrepancies and created additional features.
5. **Data Splitting:** Separated the dataset into training and testing sets.
6. **Handling Class Imbalance:** Applied SMOTE to the training data to balance the target classes.
7. **Model Building:** Trained and evaluated multiple classification algorithms.
8. **Model Comparison:** Compared performance using classification metrics and confusion matrices.
9. **Feature Importance:** Identified the features most influential in the selected model.

## 🤖 Machine Learning Models

- Logistic Regression
- XGBoost Classifier
- LightGBM Classifier

## 📈 Model Performance Comparison

| Metric | Logistic Regression | XGBoost | LightGBM |
|:---|---:|---:|---:|
| Accuracy | 98.33% | **99.82%** | 99.56% |
| Precision | 6.85% | **41.94%** | 22.49% |
| Recall | 95.77% | **99.22%** | 99.10% |
| F1-Score | 12.78% | **58.96%** | 36.66% |
| AUC | 97.05% | **99.52%** | 99.33% |

## 🏆 Best-Performing Model: XGBoost

XGBoost achieved the best overall performance among the three models based on the reported results.

- **Accuracy:** 99.82%
- **Precision:** 41.94%
- **Recall:** 99.22%
- **F1-Score:** 58.96%
- **AUC:** 99.52%

The model detected 2,416 fraudulent transactions, with 19 false negatives and 3,344 false positives, according to the notebook's reported confusion matrix.

Although accuracy was high, precision was considerably lower, indicating that some legitimate transactions were incorrectly classified as fraudulent.

## 🔍 Feature Importance

The feature-importance analysis identified the following influential features:

- `errorBalanceOrig`
- `type`
- `is_orig_zero`
- `oldbalanceOrg`
- `newbalanceOrig`
- `errorBalanceDest`
- `newbalanceDest`
- `amount`
- `step`

`errorBalanceOrig` was the most important feature in the reported analysis, followed by transaction type and `is_orig_zero`.

These findings highlight the predictive value of account balance discrepancies and transaction characteristics.

## 📌 Key Findings

- Fraudulent transactions in the dataset occurred in the TRANSFER and CASH_OUT categories.
- The target variable is highly imbalanced, making accuracy alone insufficient for evaluating model performance.
- SMOTE was used to address class imbalance in the training data.
- XGBoost achieved the highest reported F1-score and AUC among the evaluated models.
- Feature-importance analysis highlighted account balance discrepancies as important predictors of fraud.
