import os
from typing import Any

import pandas as pd

from config import PathsConfig

REQUIRED_COLUMNS = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)] + ['Class']


def load_credit_card_data(filepath: str | None = None) -> pd.DataFrame:
    if filepath is None:
        raw_path = PathsConfig.RAW_DATA_DIR / "creditcard.csv"
        root_data_path = PathsConfig.DATA_DIR / "creditcard.csv"
        if raw_path.exists():
            filepath = str(raw_path)
        elif root_data_path.exists():
            filepath = str(root_data_path)
        else:
            filepath = str(raw_path)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Please download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in data/raw/creditcard.csv or data/creditcard.csv"
        )

    df = pd.read_csv(filepath)
    validate_dataset(df)
    return df



def validate_dataset(df: pd.DataFrame) -> None:
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")

    if 'Class' not in df.columns:
        raise ValueError("Dataset must contain 'Class' column (0=normal, 1=fraud)")

    fraud_count = df[df['Class'] == 1].shape[0]
    normal_count = df[df['Class'] == 0].shape[0]

    if fraud_count == 0:
        raise ValueError("Dataset contains no fraud cases (Class=1)")

    print(f"Dataset validated: {len(df)} rows, {normal_count} normal, {fraud_count} fraud cases")


def inspect_data(df: pd.DataFrame) -> dict[str, Any]:
    fraud_cases = df[df['Class'] == 1].shape[0]
    normal_cases = df[df['Class'] == 0].shape[0]

    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "fraud_cases": fraud_cases,
        "normal_cases": normal_cases,
        "fraud_percentage": round(fraud_cases / len(df) * 100, 4),
        "columns_list": df.columns.tolist()
    }
    return stats


def get_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != 'Class']
    stats = df[feature_cols].describe()
    return stats
