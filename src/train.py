#Train ML model
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from load_data import load_data 
from preprocess import preprocess
import joblib

# Load and preprocess data
data = load_data('data/raw/data.csv')       

# Preprocess the data
data = preprocess(data) 
# Define features and target variable
X = data.drop(columns=['TARGET'])   

#


