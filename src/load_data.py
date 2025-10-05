import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


def load_data(filepath="/data/credit_dataset.csv"):
    
    df=pd.read_csv(filepath)
    
    if 'ID' in df.columns:
        df=df.drop(columns=['ID'])  
        
    df['OCCUPATION_TYPE']=df['OCCUPATION_TYPE'].fillna('Unknown')
    
    binary_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL']
    
    le=LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
        
    df=pd.get_dummies(df, columns=[
        'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE',  
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE',
        'OCCUPATION_TYPE'
    ],drop_first=True)
    
    print("data loaded and preprocessed successfully")
    print("Dataset shape after preprocessing:", df.shape)
    df.dropna()
    
    
    return df



if __name__ == "__main__":
    df = load_data("data/credit_dataset.csv")
    
