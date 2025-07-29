# 🛠️ Predictive Housing Insights — Built for Industry Leaders
From raw data to transparent decisions — all in one app.

## 🧭 Overview

This repository hosts a Streamlit-based web app designed for real estate developers, investors, analysts, and other stakeholders in the housing industry.

The app delivers a **three-in-one functionality**:

1. 📊 **Data Exploration** — Visualize patterns and trends in housing data  
2. 🧠 **Predictive Modeling** — Estimate house prices using a trained ML model  
3. 🔍 **Model Explainability** — Understand what features drive the predictions using SHAP

Whether you're forecasting development value or analyzing neighborhood trends, this app empowers data-driven decision-making at every level.

---
## 🚀 How to Use This Repository

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/shaligah/House-Price-Prediction-App.git
```
### 2. Install dependencies
pip install -r requirements.txt
### 3. Run the app
streamlit run streamlit_app.py

---
## 🧠 Model Info
- Model Type: XGBoost Regressor
- Target Variable: SalePrice (log-transformed during training)
- Explainability Tool: SHAP for local instance-based interpretability

---
## 📊 Data
- Source: Ames Housing Dataset, US Bereau of Economics
- Contains features like house size, quality, location, year built, and more
- Cleaned and preprocessed for use in prediction and visualization

---
## 🔗 Live App

