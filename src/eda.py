from load_data import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Load preprocessed data
df = load_data(filepath="data/loan_data.csv")


"""
print(df.info())
print(df.describe())
print(df.shape)
print(df.isnull().sum())
"""

def Visualize_distribution(df, columns):
    plt.figure(figsize=(20,12))
    plt.hist(df[columns], bins=50, color='blue', edgecolor='black')
    plt.title(f'Distribution of {columns}')
    plt.xlabel(columns)
    plt.ylabel('Frequency')
    plt.show()

def Visualize_boxplot(df, columns):
    plt.figure(figsize=(40,6))
    sns.boxplot(x=df[columns])
    plt.title(f'Boxplot of {columns}')
    plt.show()

def find_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    return len(outliers)

def eda_process(df):
    numaric_cols = df.select_dtypes(include=['int64', 'float64'])
    
    for col in numaric_cols.columns:
        outlier_count = find_outliers(df, col)
        if outlier_count != 0:
            print(col, outlier_count)
    

    print("------------------------------------------------------------------------------------------")
    
    coo_matrix = numaric_cols.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(coo_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                linewidths=0.5, linecolor='black', cbar=True)
    plt.title('Correlation Matrix (actual dataset)')
    plt.show()
    
    """

    print("=== Correlation Matrix Part 1 ===")
    print(coo_matrix.iloc[:, :len(coo_matrix)//2])
    print("\n=== Correlation Matrix Part 2 ===")
    print(coo_matrix.iloc[:, len(coo_matrix)//2:])
    """
    
    return df
    

if __name__ == "__main__":
    df= eda_process(df)
