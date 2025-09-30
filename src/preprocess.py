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
    df["household_size"] = df["CNT_FAM_MEMBERS"]

    # Convert days to age in years
    df["age"] = (df["DAYS_BIRTH"] / 365).astype(int)

    # Drop irrelevant columns
    df = df.drop(
        columns=[
            "DAYS_BIRTH",
            "CNT_FAM_MEMBERS",
            "CNT_CHILDREN",
            "FLAG_MOBIL",
            "FLAG_PHONE",
            "FLAG_EMAIL",
            "FLAG_WORK_PHONE",
        ]
    )

    

    # Log transformation for income
    df["AMT_INCOME_TOTAL_log"] = np.log1p(df["AMT_INCOME_TOTAL"])
    df = df.drop(columns=["AMT_INCOME_TOTAL"])
    #droping missing values
    df = df.dropna()

    # --- Outlier capping (can be change) ---
    for col in ["DAYS_EMPLOYED", "household_size", "AMT_INCOME_TOTAL_log"]:
        df = cap_outliers(df, col)
    
    numaric_cols= df.select_dtypes(include=['int64', 'float64'])
    coo_matrix= numaric_cols.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(coo_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, linecolor='black', cbar=True)
    plt.title('Correlation Matrix(after preprocess);')
    plt.show()
    
    find_outliers_count(df)
    
    # Correlation Matrix split into two stages for better readability

    # Stage 1: print first half of the columns
    print("=== Correlation Matrix Part 1 ===")
    print(coo_matrix.iloc[:, :len(coo_matrix)//2])

    # Stage 2: print second half of the columns
    print("\n=== Correlation Matrix Part 2 ===")
    print(coo_matrix.iloc[:, len(coo_matrix)//2:])
    
    return df




def load_and_preprocess(path):
    """
    Load dataset and apply preprocessing pipeline.
    """
    df = load_data(path)
    df_new= eda_process(df)
    df_preprocessed = preprocess(df_new)
    print("Dataset shape after full preprocessing:", df_preprocessed.shape)
    return df_preprocessed


if __name__ == "__main__":
    df_preprocessed = load_and_preprocess("credit_dataset.csv")