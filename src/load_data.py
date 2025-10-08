import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def load_data(filepath="data/loan_data.csv"):
    df = pd.read_csv(filepath)
    
    # Drop ID if present
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    
    # Encode binary columns
    binary_cols = ['person_gender', 'previous_loan_defaults_on_file', 'loan_status']
    le = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    
    # One-hot encode categorical columns only
    df = pd.get_dummies(df, columns=[
        'person_education',
        'person_home_ownership',
        'loan_intent'
    ])
    
    #df.drop(columns=['cb_person_cred_hist_length'], inplace=True)
    # Drop missing values
    df.dropna(inplace=True)
    
    print("Data loaded successfully")
    print("Dataset shape after preprocessing:", df.shape)
    
    return df


def readDF(filepath):
    return pd.read_csv(filepath)


if __name__ == "__main__":
    df = load_data("data/loan_data.csv")
