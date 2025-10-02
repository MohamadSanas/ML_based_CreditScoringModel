# train.py for isolation processing and model training
from preprocess import load_and_preprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


# 1. Load preprocessed dataset
df = load_and_preprocess("credit_dataset.csv")

# 2. Separate features 
X = df.values

# 3. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train Isolation Forest model
model = IsolationForest(n_estimator=100, contamination=0.05, random_state=42)
model.fit(X_scaled)



