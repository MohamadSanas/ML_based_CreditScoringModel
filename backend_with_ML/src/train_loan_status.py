# train_xgboost.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load


# Load preprocessed data
df = pd.read_csv("data/preprocessed_data.csv")

# Separate features and target
X = df.drop(columns=["loan_status"])
Y = df["loan_status"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(X_train, Y_train)

print("Before SMOTE:", Y_train.value_counts())
print("After SMOTE:", pd.Series(Y_res).value_counts())

# XGBoost classifier
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

#Hyperparameter tuning (RandomizedSearchCV)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3]
}

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X_res, Y_res)
best_model = search.best_estimator_

# Predict and evaluate
y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))

# Plot feature importance
importances = pd.Series(best_model.feature_importances_, index=X.columns)
importances.nlargest(15).plot(kind='barh', figsize=(10,6))
plt.title('Top 15 Feature Importances (XGBoost)')
plt.show()

dump(best_model, 'model/credit_eligibility_model.joblib')
