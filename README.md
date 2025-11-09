# 🧠 Customer Spending Prediction — Linear Regression + Streamlit Deployment

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Model-LinearRegression-yellow)
![Status](https://img.shields.io/badge/Deployment-Live-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit.app-brightgreen?logo=streamlit)](https://customer-spending-prediction-linreg-gd-ols-rajesh-arigala.streamlit.app)

---

### 🚀 End-to-End Machine Learning Project  
A complete ML workflow — from data preprocessing and model training to deployment using **Streamlit Cloud**.  
This app predicts a customer’s **Yearly Amount Spent** based on behavioral and membership features.

---

## 📌 **Project Overview**

This project demonstrates the full machine learning lifecycle:
1. **Model Development:** Built a Linear Regression model in Python to predict yearly customer spending.
2. **Model Evaluation:** Measured model performance using MSE, MAE, and R² metrics.
3. **Model Export:** Saved the trained model as a pickle file (`LR.pkl`).
4. **Feature Validation:** Verified and exported input features via `features.pkl`.
5. **Deployment:** Wrapped the model in a user-friendly Streamlit app (`app.py`).
6. **Prediction Modes:** Supports both single-customer input and batch CSV uploads.

---

## 🧮 **Model Performance**

| Metric | Score |
|---------|-------|
| **Mean Squared Error (MSE)** | 98.58 |
| **Mean Absolute Error (MAE)** | 7.89 |
| **R² (Model Accuracy)** | 0.98 ✅ |

📈 A high R² score (0.98) shows that the model explains ~98% of the variance in customer spending.

---

## 🧰 **Tech Stack**

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python 3 |
| **Data Analysis** | Pandas, NumPy |
| **Modeling** | Scikit-learn (LinearRegression) |
| **Deployment** | Streamlit |
| **Version Control** | Git, GitHub |

---

## 📂 **Repository Structure**
```
customer_spending_predictor/
│
├── app.py # Streamlit app for deployment
├── Linear_Model.ipynb # Model training and saving (creates LR.pkl)
├── check_features.ipynb # Verified and exported feature names
├── LR.pkl # Trained Linear Regression model
├── features.pkl # Stored feature names for the model
├── requirements.txt # Python dependencies
├── test_sample.csv # Single sample input for testing
├── Batch_Input.csv # Multiple customer records for batch testing
├── Batch_Predictions.csv # Batch data with predicted results
└── README.md # Project documentation
```
---