import pandas as pd
import numpy as np


def load_data(path):
    """
    Load dataset from given path
    """
    return pd.read_csv(path)


def clean_data(df):
    """
    Perform basic data cleaning:
    - Replace '?' with NaN
    - Drop unnecessary columns
    - Convert target variable
    - Handle missing values
    """

    df = df.copy()

    # 1. Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # 2. Drop ID columns (not useful for ML)
    cols_to_drop = ["encounter_id", "patient_nbr"]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # 3. Convert target variable
    # '<30' → 1 (readmitted)
    # '>30' or 'NO' → 0
    df["readmitted"] = df["readmitted"].apply(
        lambda x: 1 if x == "<30" else 0
    )

    # 4. Handle missing values
    # Fill categorical with "Unknown"
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # Fill numerical with median
    num_cols = df.select_dtypes(exclude=["object"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df