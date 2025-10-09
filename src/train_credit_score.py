import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
from joblib import dump

df = pd.read_csv("data/preprocessed_data.csv")

X = df.drop(columns=["credit_score","loan_status","loan_amnt_log","loan_percent_income_log",
                     "loan_percent_income_bucket_Low","loan_percent_income_bucket_Medium",
                     "loan_percent_income_bucket_High","loan_percent_income_bucket_Very High"
                     ])

Y = df["credit_score"]


X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,test_size=0.2,random_state=42
    )


model_1=RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

model_1.fit(X_train,Y_train)

# Evaluate model
Y_pred = model_1.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Save the trained model
dump(model_1, "model/loan_amount_model.joblib")
print("Model saved to 'model/loan_amount_model.joblib'.")
