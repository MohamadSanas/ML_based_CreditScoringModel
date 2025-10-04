# train_unsupervised.py
from preprocess import load_and_preprocess
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns


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


print("Sample Normal Points (Eligible):")
print(df[df["anomaly"] == 1].head())

print("\nSample Anomalies (Not Eligible):")
print(df[df["anomaly"] == -1].head())

eligible_df = df[df["anomaly"] == 1]
not_eligible_df = df[df["anomaly"] == -1]

eligible_df.to_csv("eligible_applicants.csv", index=False)
print("Eligible applicants saved to 'eligible_applicants.csv'.")

not_eligible_df.to_csv("not_eligible_applicants.csv", index=False)
print("Not eligible applicants saved to 'not_eligible_applicants.csv'.")
