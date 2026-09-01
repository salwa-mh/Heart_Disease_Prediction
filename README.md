# 🩺 Heart Disease Prediction System

A comprehensive Machine Learning project designed to predict the presence of heart disease in patients based on medical and lifestyle features. The system incorporates data preprocessing, model selection, hyperparameter tuning, and an interactive Streamlit application with model interpretability via SHAP.

---

## 📌 Project Overview
Early detection of heart disease can help healthcare professionals take preventive actions and improve patient outcomes. This project evaluates 5 classification algorithms to build a reliable predictive system using clinical parameters like age, cholesterol, blood pressure, and smoking status.

For detailed technical analysis and full metrics, check out the [Project Report](Project_Report.pdf).

---

## 🛠️ Data Preprocessing & Pipeline
* **Data Cleaning:** Removed non-predictive `id` columns and duplicate records.
* **Imputation:** Handled missing values using statistical measures (median/mode) to prevent data leakage.
* **Outlier Handling:** Capped numerical outliers using the Interquartile Range (IQR) method.
* **Encoding & Scaling:** Applied One-Hot Encoding to categorical variables and normalized numerical features using `StandardScaler`.

---

## 📊 Model Evaluation Summary

We compared 5 machine learning models using standard metrics:

| Model | Accuracy | F1-Score | Optimal Configuration |
| :--- | :---: | :---: | :--- |
| **Random Forest** ⭐ | **89.0%** | **0.86** | `n_estimators=200`, `max_depth=5` |
| **Decision Tree** | 85.5% | 0.826 | `max_depth=3` |
| **SVM** | 83.6% | 0.809 | `kernel='rbf'`, `C=1` |
| **Logistic Regression** | 82.8% | 0.798 | `penalty='l2'`, `C=0.1` |
| **Gaussian Naive Bayes** | 81.0% | 0.800 | `SelectKBest (k=15)` |

**🏆 Final Selected Model:** **Random Forest** achieved the highest accuracy and best-balanced performance with minimal misclassification errors.

---

## 💻 Streamlit Web Application
The repository includes an interactive web interface (`app.py`) built with Streamlit. It enables users to:
1. Input patient metrics manually.
2. Get instant predictions on heart disease risk.
3. Understand prediction factors using **SHAP (SHapley Additive exPlanations)**.
4. View performance metrics and interactive Confusion Matrices.

---

## 📂 Repository Structure

```text
.
├── data/              # Processed train/test datasets
├── models/            # Saved .pkl models & preprocessing assets
├── notebooks/         # Model training & visualization notebooks
├── results/           # Confusion matrices and prediction logs
├── app.py             # Streamlit application
├── Project_Report.pdf # Comprehensive technical report
└── README.md          # Project documentation
