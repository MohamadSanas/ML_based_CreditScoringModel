from load_data import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Load preprocessed data
df = load_data("credit_dataset.csv")

"""
print(df.info())
print(df.describe())
print(df.shape)
print(df.isnull().sum())
"""


def Visualize_distribution(df, columns):
    # 2. Visualize the distribution of a numerical column
    plt.figure(figsize=(20,12))
    plt.hist(df[columns], bins=50, color='blue', edgecolor='black')
    plt.title(f'Distribution of {columns}')
    plt.xlabel(columns)
    plt.ylabel('Frequency')
    plt.show()


def Visualize_countplot(df, columns):
    # 3. Boxplot to identify outliers in a numerical column
    plt.figure(figsize=(40,6))
    sns.boxplot(x=df[columns])
    plt.title(f'Boxplot of {columns}')
    plt.show()
    
def find_outliers(df, column):
    """Count number of outliers in a column using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    return len(outliers)

def eda_process(df):
    # change to num values for better visualization
    numaric_cols= df.select_dtypes(include=['int64', 'float64'])

    # check outliers for numerical columns
    for col in numaric_cols:
        if find_outliers(df, col) !=0:
            print( col,find_outliers(df, col))
    print ("--------------------------------------------------")


    # 4. Correlation heatmap for numerical features
    coo_matrix= numaric_cols.corr()

    plt.figure(figsize=(12,10))
    sns.heatmap(coo_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, linecolor='black', cbar=True)
    plt.title('Correlation Matrix (actual dataset);')
    plt.show()


    # Correlation Matrix split into two stages for better readability

    # Stage 1: print first half of the columns
    print("=== Correlation Matrix Part 1 ===")
    print(coo_matrix.iloc[:, :len(coo_matrix)//2])

    # Stage 2: print second half of the columns
    print("\n=== Correlation Matrix Part 2 ===")
    print(coo_matrix.iloc[:, len(coo_matrix)//2:])
    
    return df

if __name__ == "__main__":
    eda_process(df)


