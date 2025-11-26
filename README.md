# ML-Based Credit Scoring System  
A full-stack machine learning web application that predicts loan eligibility using a trained ML model, a Flask backend API, and a Flutter web frontend.

---

## 🚀 Project Overview

This project is an end-to-end **Credit Scoring System** that evaluates loan eligibility based on customer financial and demographic data.  
It includes:

- **Machine Learning Model** (trained using joblib)
- **Flask Backend API**  
- **SHAP Explainability** showing top 5 feature contributions  
- **Flutter Web Frontend** for user interaction  
- **Render Deployment** for hosting the backend + frontend

---

## 🧠 Features

### 🔍 ML Model  
- Trained classification model  
- Predicts `loan approved` or `loan rejected`  
- Exported using `joblib`

### 🧩 Explainability (SHAP)  
The API returns the **top 5 contributing features** for every prediction.

### 🌐 API Backend (Flask)
- Predict loan eligibility  
- Serve static Flutter web files  
- CORS enabled  
- Health check endpoint `/healthz` for Render

### 🎨 Flutter Web Frontend  
- User-friendly form  
- Sends data to the backend  
- Displays loan approval + SHAP insights  

---

## 🗂️ Repository Structure

