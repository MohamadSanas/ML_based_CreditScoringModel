# train_unsupervised.py
from preprocess import load_and_preprocess
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 1. Load preprocessed data
df = load_and_preprocess("credit_dataset.csv")  # your preprocessed dataset

# 2. Features (all columns, no target column)
X = df.values  

# 3. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train Isolation Forest (unsupervised)
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_scaled)

# 5. Predict anomalies (-1 = anomaly, 1 = normal)
y_pred = model.predict(X_scaled)

# Add anomaly column to dataframe
df["anomaly"] = y_pred

# 6. Count anomalies
print("Normal points:", list(y_pred).count(1))
print("Anomalies:", list(y_pred).count(-1))

# 7. Optional: visualize first two features (if dataset is high-dimensional, you can use PCA)
if X.shape[1] == 2:  
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap="coolwarm")
    plt.title("Isolation Forest Anomaly Detection")
    plt.show()

print("Sample Normal Points (Eligible):")
print(df[df["anomaly"] == 1].head())

print("\nSample Anomalies (Not Eligible):")
print(df[df["anomaly"] == -1].head())