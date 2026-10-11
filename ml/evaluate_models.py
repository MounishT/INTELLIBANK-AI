from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import MLConfig, PathsConfig


def evaluate_all_models(threshold: float | None = None) -> pd.DataFrame:
    if threshold is None:
        threshold = MLConfig.FRAUD_THRESHOLD

    x_test_path = PathsConfig.PROCESSED_DATA_DIR / "X_test.csv"
    y_test_path = PathsConfig.PROCESSED_DATA_DIR / "y_test.csv"

    if not x_test_path.exists() or not y_test_path.exists():
        return pd.DataFrame({"error": ["Test data not found. Run train_model.py first."]})

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)['Class']

    models = {
        "Logistic Regression": PathsConfig.MODELS_DIR / "logistic_regression.pkl",
        "Random Forest": PathsConfig.MODELS_DIR / "random_forest.pkl",
        "XGBoost": PathsConfig.MODELS_DIR / "xgboost.pkl"
    }

    results: list[dict[str, Any]] = []

    for name, path in models.items():
        if not path.exists():
            print(f"Model {name} not found at {path}, skipping...")
            continue

        try:
            model = joblib.load(path)
            y_probs = model.predict_proba(X_test)[:, 1]

            y_pred = (y_probs >= threshold).astype(int)

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = fp = fn = tp = 0

            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            pr_auc = average_precision_score(y_test, y_probs)
            roc_auc = roc_auc_score(y_test, y_probs)

            results.append({
                "Model": name,
                "Threshold": threshold,
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(f1, 4),
                "PR-AUC": round(pr_auc, 4),
                "ROC-AUC": round(roc_auc, 4),
                "False Positives": int(fp),
                "False Negatives": int(fn),
                "True Positives": int(tp),
                "True Negatives": int(tn)
            })

        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            results.append({
                "Model": name,
                "Threshold": threshold,
                "Error": str(e)
            })

    df_results = pd.DataFrame(results)
    PathsConfig.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(PathsConfig.REPORTS_DIR / "model_comparison.csv", index=False)

    return df_results


def find_optimal_threshold(model_name: str = "Random Forest") -> dict[str, Any]:
    x_test_path = PathsConfig.PROCESSED_DATA_DIR / "X_test.csv"
    y_test_path = PathsConfig.PROCESSED_DATA_DIR / "y_test.csv"

    if not x_test_path.exists() or not y_test_path.exists():
        return {"error": "Test data not found. Run train_model.py first."}

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)['Class']

    model_path = PathsConfig.MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.pkl"
    if not model_path.exists():
        return {"error": f"Model {model_name} not found."}

    model = joblib.load(model_path)
    y_probs = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = MLConfig.FRAUD_THRESHOLD

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    precision, recall, _ = precision_recall_curve(y_test, y_probs)

    return {
        "optimal_threshold": round(best_threshold, 2),
        "best_f1": round(best_f1, 4),
        "precision_recall_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist()
        }
    }


if __name__ == "__main__":
    res = evaluate_all_models()
    print(res)
    print("\nFinding optimal threshold...")
    opt = find_optimal_threshold()
    print(opt)
