from eda import eda_process
from load_data import load_data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def find_outliers(df, column):
    """Count number of outliers in a column using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    return len(outliers)


def cap_outliers(df, column):
    """Cap outliers at lower/upper IQR bounds."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# check outliers for numeric11al columns after capping
def find_outliers_count(df):
    numaric_cols= df.select_dtypes(include=['int64', 'float64'])
    count=0
    for col in numaric_cols:
        if find_outliers(df, col) !=0:
            count=+1
            return col,find_outliers(df, col)
    if count==0:
        return "No outlters found"


def preprocess(df):
    """
    Clean and transform raw dataframe into model-ready format.
    """

    # --- Feature Engineering ---

    # Drop irrelevant columns
    

    # Log transformation for income
    for col in ["loan_amnt","cb_person_cred_hist_length","loan_int_rate"]:
        df[col + '_log'] = np.log1p(df[col])
        df.drop(columns=[col], inplace=True)

    

    # --- Outlier capping (can be change) ---
    
    for col in ["person_age","person_income","person_emp_exp","loan_amnt_log","credit_score","cb_person_cred_hist_length_log"]:
        df = cap_outliers(df, col)
        
    df['loan_percent_income_log'] = np.log1p(df['loan_percent_income'])
    
    df['loan_percent_income_bucket'] = pd.cut(
    df['loan_percent_income'], 
    bins=[0, 10, 20, 30, 50, 100], 
    labels=['Very Low','Low','Medium','High','Very High']
    )
    df=df.drop(columns=['loan_percent_income'], inplace=True)

    df = pd.get_dummies(df, columns=['loan_percent_income_bucket'], drop_first=True)

    df=df.dropna(inplace=True)
    """
    
    numaric_cols= df.select_dtypes(include=['int64', 'float64'])
    coo_matrix= numaric_cols.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(coo_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, linecolor='black', cbar=True)
    plt.title('Correlation Matrix(after preprocess);')
    plt.show()
    
    
    
    # Correlation Matrix split into two stages for better readability

    # Stage 1: print first half of the columns
    print("=== Correlation Matrix Part 1 ===")
    print(coo_matrix.iloc[:, :len(coo_matrix)//2])

    # Stage 2: print second half of the columns
    print("\n=== Correlation Matrix Part 2 ===")
    print(coo_matrix.iloc[:, len(coo_matrix)//2:])
    """
    
    return df




def load_and_preprocess(path):
    """
    Load dataset and apply preprocessing pipeline.
    """
    df = load_data(path)
    df_new= eda_process(df)
    df_preprocessed = preprocess(df_new)
    print("Dataset shape after full preprocessing:", df_preprocessed.shape)
    
    df_preprocessed.dropna(inplace=True)
    df_preprocessed.to_csv("data/preprocessed_data.csv", index=False)
    print("Preprocessed data saved to '../data/preprocessed_data.csv'.")
    

    
    numaric_cols = df_preprocessed.select_dtypes(include=['int64', 'float64'])
    for col in numaric_cols.columns:
        outlier_count = find_outliers(df, col)
        if outlier_count != 0:
            print(col, outlier_count)
    
    return df_preprocessed


if __name__ == "__main__":
    df_preprocessed = load_and_preprocess("data/loan_data.csv")