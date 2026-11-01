"""
Data Preparation script for the "Visit with Us" Wellness Tourism Package project.

Loads the registered dataset, cleans it, removes unnecessary columns, and
splits it into train/test sets. The splits are saved to the current working
directory (repo root) so the GitHub Actions workflow can upload them as a
workflow artifact for the model-training job to consume.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")

TARGET_COLUMN = "ProdTaken"

# Identifier / index columns that carry no predictive signal.
COLUMNS_TO_DROP = ["Unnamed: 0", "CustomerID"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop identifier / index columns that are not useful for modelling.
    df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns])

    # Fix known data-entry inconsistencies in categorical columns.
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Drop exact duplicate rows, if any.
    df = df.drop_duplicates()

    return df


def prepare_data(data_path: str = DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(data_path)
    df = clean_data(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete.")
    print(f"Dropped columns: {COLUMNS_TO_DROP}")
    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Xtrain: {Xtrain.shape}  Xtest: {Xtest.shape}")
    print(f"ytrain: {ytrain.shape}  ytest: {ytest.shape}")
    print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

    return Xtrain, Xtest, ytrain, ytest


if __name__ == "__main__":
    prepare_data()
