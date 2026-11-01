"""
Data Registration script for the "Visit with Us" Wellness Tourism Package project.

Reads tourism.csv from the repository's data folder, validates that every
expected column is present, and prints a short summary of the dataset so
that data drift / schema issues are caught early in the pipeline.
"""

import os
import sys

import pandas as pd

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


def register_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find the dataset at '{data_path}'. "
            "Make sure tourism.csv has been uploaded to tourism_project/data/."
        )

    df = pd.read_csv(data_path)

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    print("Dataset registered successfully from:", data_path)
    print("-" * 60)
    print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
    print("-" * 60)
    print("Expected columns present:", len(EXPECTED_COLUMNS), "/", len(EXPECTED_COLUMNS))
    print("-" * 60)
    print("Column dtypes:")
    print(df[EXPECTED_COLUMNS].dtypes)
    print("-" * 60)
    print("Missing values per column:")
    print(df[EXPECTED_COLUMNS].isna().sum())
    print("-" * 60)
    print("Target distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True).rename("proportion"))
    print("-" * 60)
    print("Dataset registration complete.")

    return df


if __name__ == "__main__":
    try:
        register_dataset()
    except Exception as exc:
        print(f"Dataset registration failed: {exc}", file=sys.stderr)
        sys.exit(1)
