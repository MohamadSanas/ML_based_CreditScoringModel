import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


def load_data(filepath="/data/loan_data.csv"):
    
    df=pd.read_csv(filepath)
    
    if 'ID' in df.columns:
        df=df.drop(columns=['ID'])  
        
    binary_cols = ['person_gender', 'previous_loan_defaults_on_file', 'loan_status']
    
    le=LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
        
    df=pd.get_dummies(df, columns=[
        'person_age',
        'person_education',  
        'person_income',
        'person_emp_exp',
        'person_home_ownership',
        'loan_amnt',
        'loan_intent',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        'credit_score'
        
    ],drop_first=True)
    
    print("data loaded successfully")
    print("Dataset shape after preprocessing:", df.shape)
    df.dropna()
    
    
    return df


def readDF(filepath):
    df=pd.read_csv(filepath)
    return df



if __name__ == "__main__":
    #df = load_data("data/loan_data.csv")
    df = readDF("data/loan_data.csv")
    
    
