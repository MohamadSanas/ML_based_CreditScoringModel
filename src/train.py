# train_supervised.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# 1. Load preprocessed data
df = pd.read_csv("data/preprocessed_data.csv")
#print(df.columns)


# 2. Select features

X = df.drop(columns=["loan_status"])
Y = df["loan_status"]

# 3. Split data into training and testing sets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=42,stratify=Y)

# 4. Train RandomForest model classifier
model = RandomForestClassifier(
    n_estimators=200, 
    random_state=42,
    max_depth=None,
    n_jobs=-1,
    class_weight={0:3,1:1}    # Adjust weights to handle class imbalance
    ) 


model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))


plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(15).plot(kind='barh', figsize=(10,6) )
plt.title('Top 15 Feature Importances')
plt.show()   

"""
# Save the trained model
dump(model, "model/credit_eligibility_model.joblib")

# Save the scaler
dump(scaler, "model/scaler.joblib")

print("Model and scaler saved successfully!")

"""