# predict_new_applicants_array.py
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1️⃣ Load trained model and scaler
model = load("model/credit_eligibility_model.joblib")
scaler = load("model/scaler.joblib")

# 2️⃣ Define new applicant data as a list of lists (each inner list = one applicant)
# Order of features must match training: ['young_age_score','AMT_INCOME_TOTAL_log','FLAG_OWN_CAR','FLAG_OWN_REALTY','DAYS_EMPLOYED','household_size']
new_applicants = [
    [-0.05, 10.5, 1, 1, 20000, 1],  # applicant 1
    [-0.012, 8.0, 0, 1, 1500, 2],   # applicant 2
    [-0.016, 9.0, 1, 1, 2500, 4],   # applicant 3
]

# 3️⃣ Convert to DataFrame (optional, for convenience)
considering_features = ['young_age_score','AMT_INCOME_TOTAL_log','FLAG_OWN_CAR','FLAG_OWN_REALTY','DAYS_EMPLOYED','household_size']
new_df = pd.DataFrame(new_applicants, columns=considering_features)

# 4️⃣ Scale the data
X_new_scaled = scaler.transform(new_df.values)

# 5️⃣ Predict eligibility (-1 = Not Eligible, 1 = Eligible)
y_new_pred = model.predict(X_new_scaled)

# 6️⃣ Map predictions to readable labels
new_df["eligibility"] = ["Eligible" if x == 1 else "Not Eligible" for x in y_new_pred]

# 7️⃣ Show results
new_df['age'] = ((-1)/new_df['young_age_score'] - 1).astype(int)
print(new_df)
