import pandas as pd
import numpy as np


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

    # Drop missing values
    df = df.dropna()

    # Log transformation for income
    df["AMT_INCOME_TOTAL_log"] = np.log1p(df["AMT_INCOME_TOTAL"])
    df = df.drop(columns=["AMT_INCOME_TOTAL"])

    # --- Outlier capping ---
    for col in ["DAYS_EMPLOYED", "household_size", "AMT_INCOME_TOTAL_log"]:
        df = cap_outliers(df, col)

    return df


def load_and_preprocess(path="credit_dataset.csv"):
    """
    Load dataset and apply preprocessing pipeline.
    """
    df = pd.read_csv(path)
    df = preprocess(df)
    return df
