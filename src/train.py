# train_unsupervised.py
from preprocess import load_and_preprocess
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

# 1. Load preprocessed data
df = load_and_preprocess("data/credit_dataset.csv")

# 2. Select features
considering_features = [
    'young_age_score',
    'AMT_INCOME_TOTAL_log',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'DAYS_EMPLOYED',
    'household_size'
]
X = df[considering_features].values  

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

# 6. Apply manual rule: young_age_score(age >49) < -0.02 → mark as Not Eligible (-1)
mask = (df["young_age_score"] > -0.0167) & (df["anomaly"] == 1)
df.loc[mask, "anomaly"] = np.where(np.random.rand(mask.sum()) < 1.0, -1, 1)

#
mask = (df["young_age_score"] > -0.02) & (df["anomaly"] == 1)
df.loc[mask, "anomaly"] = np.where(np.random.rand(mask.sum()) < 0.7, -1, 1)

# Additional manual rule: If CODE_GENDER == 0 
mask = (df["CODE_GENDER"] == 0) & (df["anomaly"] == -1)
df.loc[mask, "anomaly"] = np.where(np.random.rand(mask.sum()) < 0.2, 1, -1)



# Additional manual rule: If FLAG_OWN_REALTY == 1 and marked as Not Eligible (-1), randomly reassign 50% to Eligible (1)
mask = (df["FLAG_OWN_REALTY"] == 1) & (df["anomaly"] == -1)
df.loc[mask, "anomaly"] = np.where(np.random.rand(mask.sum()) < 0.5, 1, -1)



# Additional manual rule: If FLAG_OWN_CAR == 1 and marked as Not Eligible (-1), randomly reassign 30% to Eligible (1)
mask = (df["FLAG_OWN_CAR"] == 1) & (df["anomaly"] == -1)
df.loc[mask, "anomaly"] = np.where(np.random.rand(mask.sum()) < 0.3, 1, -1)



# 7. Count anomalies
print("Normal points:", list(df["anomaly"]).count(1))
print("Anomalies:", list(df["anomaly"]).count(-1))

# 8. Show samples
print("\nSample Normal Points (Eligible):")
print(df[df["anomaly"] == 1].head())

print("\nSample Anomalies (Not Eligible):")
print(df[df["anomaly"] == -1].head())

# 9. Save results
eligible_df = df[df["anomaly"] == 1]
not_eligible_df = df[df["anomaly"] == -1]

eligible_df.to_csv("data/eligible_applicants.csv", index=False)
print("Eligible applicants saved to 'eligible_applicants.csv'.")

not_eligible_df.to_csv("data/not_eligible_applicants.csv", index=False)
print("Not eligible applicants saved to 'not_eligible_applicants.csv'.")


'''
# 10. Visualizations for insights
eligible_mean = df[df["anomaly"] == 1]["FLAG_OWN_CAR"].mean()
not_eligible_mean = df[df["anomaly"] == -1]["FLAG_OWN_CAR"].mean()
print(f"\nAverage income (eligible): {eligible_mean:.2f}")
print(f"Average income (not eligible): {not_eligible_mean:.2f}")
'''





# Save the trained model
dump(model, "model/credit_eligibility_model.joblib")

# Save the scaler
dump(scaler, "model/scaler.joblib")

print("Model and scaler saved successfully!")

