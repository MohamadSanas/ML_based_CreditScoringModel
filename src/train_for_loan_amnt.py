# train_max_or_min_loan_amnt.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

#  Load preprocessed data
df = pd.read_csv("data/aproved_loan.csv")


# Separate features and target
X = df.drop(columns=["loan_amnt_log","loan_status","previous_loan_defaults_on_file"])
Y = df["loan_amnt_log"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train RandomForestRegressor
rf = RandomForestRegressor(
    n_estimators=200,  
    max_depth=None,    
    random_state=42,
    n_jobs=-1       
)

rf.fit(X_train, Y_train)

# Evaluate model
Y_pred = rf.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Save the trained model
dump(rf, "model/loan_amount_model.joblib")
print("Model saved to 'model/loan_amount_model.joblib'.")
