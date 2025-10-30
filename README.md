# Customer Churn Prediction Project

## 📌 Project Overview
This project predicts customer churn using machine learning. The goal is to analyze customer behavior and identify which users are likely to leave a service. Businesses can use this model to take proactive actions like retention offers and better support.

## 🧠 Problem Solved
Companies lose revenue when customers cancel subscriptions.  
This project helps identify customers at high risk of churn so the business can:

- Improve customer retention strategy
- Give personalized offers to risky customers
- Reduce overall churn rate
- Save marketing & re-acquisition cost

## 👤 Author
**Yuvraj**  
17-year-old aspiring ML engineer from India  
📧 Email: **y.india.main@gmail.com**

## ✅ Steps Followed
### 1. Problem Understanding
- Business problem: Identify customers likely to churn  
- Type: Binary classification (0 = stays, 1 = churn)

### 2. Data Loading & Exploration
- Loaded dataset
- Checked missing values & data types
- EDA: distributions, churn ratio, correlations

### 3. Data Preprocessing
- Missing value handling  
- Categorical encoding  
- Feature/target split  
- Train-test split (80-20)  
- StandardScaler normalization for numeric features  

### 4. Model Training
Trained models:

- Logistic Regression
- Random Forest Classifier ✅ selected

Metrics used:
- Accuracy
- ROC-AUC
- Confusion Matrix
- Precision/Recall/F1-score

### 5. Model Saving
Files saved in `models/`

| File | Description |
|------|-------------|
| churn_model.pkl | Trained model |
| scaler.pkl | StandardScaler |
| model_features.pkl | Feature list |

Saved using `joblib.dump(...)`.

### 6. Inference Pipeline
Created `predict.py` to:

- Load model + scaler + features
- Preprocess new data
- Predict churn + probability

### 7. Deployment Plan
- Streamlit interface
- Input form + results + charts
- Host on Streamlit Cloud / Render

## 📂 Project File Structure
```
project/
 ├── data/                  
 ├── notebooks/             
 ├── models/                
 ├── predict.py             
 ├── app/streamlit_app.py   
 ├── requirements.txt
 └── README.md
```

## 🛠️ Skills Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-Learn
- Machine Learning (Classification)
- Model Serialization (joblib)
- Streamlit (deployment)

## 🎯 Deliverables
- EDA + model training notebook
- Saved model and scaler
- Prediction script
- Streamlit web app

---

Thanks for reading!  
Have a productive day and keep learning 🔥
