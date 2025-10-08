from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from joblib import dump


# Load preprocessed data
df = pd.read_csv("data/preprocessed_data.csv")

# Separate features and target
X = df.drop(columns=["loan_amnt_log"])
Y = df["loan_amnt_log"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# RandomForestRegressor model
rf = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=200)

# Optional: Hyperparameter tuning (RandomizedSearchCV)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid, 
    n_iter=20,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, Y_train)
best_rf = search.best_estimator_

# Evaluate model
Y_pred = best_rf.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Save the trained model
dump(best_rf, "model/loan_amount_model.joblib")
print("Model saved to 'model/loan_amount_model.joblib'.")