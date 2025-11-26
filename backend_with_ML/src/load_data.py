import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(filepath="data/loan_approval_dataset.csv"):
    df = pd.read_csv(filepath)

    # Fix column names by removing extra spaces
    df.columns = df.columns.str.strip()

    # Drop ID if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Encode binary columns
    binary_cols = ["education", "self_employed", "loan_status"]
    le = LabelEncoder()

    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # Drop missing values
    df.dropna(inplace=True)

    print("Data loaded successfully")
    print("Dataset shape after preprocessing:", df.shape)

    return df


def readDF(filepath):
    return pd.read_csv(filepath)


if __name__ == "__main__":
    df = load_data("data/loan_approval_dataset.csv")
