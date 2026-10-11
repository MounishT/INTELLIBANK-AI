from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config import MLConfig, PathsConfig

MODELS_DIR: Path = PathsConfig.MODELS_DIR  # type: ignore[assignment]


def get_model_path(model_name: str) -> Path:
    return MODELS_DIR / f"{model_name}.pkl"


def get_scaler_path() -> Path:
    return MODELS_DIR / "scaler.pkl"


def predict_fraud(transaction_data: dict[str, Any], threshold: float | None = None) -> dict[str, Any]:
    if threshold is None:
        threshold = MLConfig.FRAUD_THRESHOLD

    required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]

    missing_cols = [col for col in required_cols if col not in transaction_data]
    if missing_cols:
        return {
            "error": "Missing features required for ML prediction. Expected Time, Amount, and V1-V28.",
            "missing": missing_cols
        }

    scaler_path = get_scaler_path()
    model_path = get_model_path("random_forest")

    if not scaler_path.exists() or not model_path.exists():
        return {
            "error": "ML Models not found. Please train the models first."
        }

    try:
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)

        df = pd.DataFrame([transaction_data], columns=required_cols)

        df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])

        probability = model.predict_proba(df)[0][1]
        is_fraud = bool(probability >= threshold)

        return {
            "risk_score": float(probability),
            "threshold_used": threshold,
            "is_suspicious": is_fraud,
            "recommendation": "Review immediately." if is_fraud else "Normal activity."
        }

    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}"
        }


def predict_fraud_batch(transactions: list[dict[str, Any]], threshold: float | None = None) -> list[dict[str, Any]]:
    if threshold is None:
        threshold = MLConfig.FRAUD_THRESHOLD

    required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]

    scaler_path = get_scaler_path()
    model_path = get_model_path("random_forest")

    if not scaler_path.exists() or not model_path.exists():
        return [{"error": "ML Models not found. Please train the models first."}] * len(transactions)

    try:
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)

        results: list[dict[str, Any]] = []
        for tx in transactions:
            missing_cols = [col for col in required_cols if col not in tx]
            if missing_cols:
                results.append({
                    "error": "Missing features required for ML prediction.",
                    "missing": missing_cols
                })
                continue

            df = pd.DataFrame([tx], columns=required_cols)
            df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])

            probability = model.predict_proba(df)[0][1]
            is_fraud = bool(probability >= threshold)

            results.append({
                "risk_score": float(probability),
                "threshold_used": threshold,
                "is_suspicious": is_fraud,
                "recommendation": "Review immediately." if is_fraud else "Normal activity."
            })

        return results

    except Exception as e:
        return [{"error": f"Batch prediction failed: {str(e)}"}] * len(transactions)
