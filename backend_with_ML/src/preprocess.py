from eda import eda_process
from load_data import load_data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ----------------------------
# Outlier handling functions
# ----------------------------
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


# ----------------------------
# Preprocessing function
# ----------------------------
def preprocess(df):
    """Clean and transform raw dataframe into model-ready format."""

    # Strip column names
    df.columns = df.columns.str.strip()

    # Drop irrelevant columns safely
    if "luxury_assets_value" in df.columns:
        df = df.drop(columns=["luxury_assets_value"])

    # Log transformation for skewed numeric columns

    # Cap outliers in all numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        df = cap_outliers(df, col)

    # Correlation matrix
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="black",
        cbar=True,
    )
    plt.title("Correlation Matrix (After Preprocessing)")
    plt.show()

    # Print correlation in two parts for readability
    mid = len(corr_matrix.columns) // 2
    print("=== Correlation Matrix Part 1 ===")
    print(corr_matrix.iloc[:, :mid])
    print("\n=== Correlation Matrix Part 2 ===")
    print(corr_matrix.iloc[:, mid:])

    return df


# ----------------------------
# Load and preprocess pipeline
# ----------------------------
def load_and_preprocess(path):
    """Load dataset, apply EDA, preprocess, and save to CSV."""

    # Load and encode dataset
    df = load_data(path)

    # Apply any EDA process (if needed)
    df = eda_process(df)

    # Preprocess dataset
    df_preprocessed = preprocess(df)

    # Drop remaining NaNs if any
    df_preprocessed.dropna(inplace=True)

    # Save to CSV
    df_preprocessed.to_csv("data/preprocessed_data.csv", index=False)
    print("Preprocessed data saved to 'data/preprocessed_data.csv'.")
    print("Dataset shape after preprocessing:", df_preprocessed.shape)

    # Print outlier counts after preprocessing
    numeric_cols = df_preprocessed.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        outlier_count = find_outliers(df_preprocessed, col)
        if outlier_count != 0:
            print(f"Outliers in {col}: {outlier_count}")

    return df_preprocessed


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    df_preprocessed = load_and_preprocess("data/loan_approval_dataset.csv")
