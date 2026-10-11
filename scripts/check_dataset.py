#!/usr/bin/env python
"""
Download and verify the Kaggle Credit Card Fraud Detection dataset.
This script helps users download the dataset from Kaggle.
"""
import sys
from pathlib import Path

# Add project root to path (required before local imports)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PathsConfig  # noqa: E402
from ml.data_loader import inspect_data, load_credit_card_data  # noqa: E402


def check_dataset():
    """Check if dataset exists and inspect it."""
    filepath = PathsConfig.RAW_DATA_DIR / "creditcard.csv"

    if not filepath.exists():
        print(f"Dataset not found at {filepath}")
        print("\nTo download the dataset:")
        print("1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("2. Download creditcard.csv")
        print("3. Place it in data/raw/creditcard.csv")
        print("\nOr use Kaggle API:")
        print("  pip install kaggle")
        print("  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip")
        return False

    print(f"Dataset found at {filepath}")
    try:
        df = load_credit_card_data(str(filepath))
        stats = inspect_data(df)
        print("\nDataset Statistics:")
        print(f"  Rows: {stats['rows']:,}")
        print(f"  Columns: {stats['columns']}")
        print(f"  Missing values: {stats['missing_values']}")
        print(f"  Duplicates: {stats['duplicates']}")
        print(f"  Normal cases: {stats['normal_cases']:,}")
        print(f"  Fraud cases: {stats['fraud_cases']:,}")
        print(f"  Fraud percentage: {stats['fraud_percentage']}%")
        print(f"\nColumns: {stats['columns_list']}")
        return True
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False


if __name__ == "__main__":
    check_dataset()
