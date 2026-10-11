import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import MLConfig, PathsConfig
from ml.data_loader import load_credit_card_data


def train_and_save_models(filepath: str | None = None) -> bool:
    try:
        df = load_credit_card_data(filepath)
    except FileNotFoundError as e:
        print(e)
        return False

    print("Dataset loaded successfully. Preparing data...")

    X = df.drop(columns=['Class'])
    y = df['Class']

    split_params = MLConfig.SPLIT_PARAMS
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_params.get("test_size", 0.2),
        stratify=y if split_params.get("stratify", True) else None,
        random_state=split_params.get("random_state", 42)
    )

    scaler = StandardScaler()
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

    PathsConfig.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, PathsConfig.MODELS_DIR / "scaler.pkl")

    rf_params = MLConfig.RF_PARAMS
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=rf_params.get("n_estimators", 200),
        max_depth=rf_params.get("max_depth", 15),
        min_samples_split=rf_params.get("min_samples_split", 5),
        min_samples_leaf=rf_params.get("min_samples_leaf", 2),
        random_state=rf_params.get("random_state", 42),
        n_jobs=rf_params.get("n_jobs", -1),
        class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, PathsConfig.MODELS_DIR / "random_forest.pkl")

    lr_params = MLConfig.LR_PARAMS
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=lr_params.get("max_iter", 1000),
        random_state=lr_params.get("random_state", 42),
        class_weight=lr_params.get("class_weight", "balanced")
    )
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, PathsConfig.MODELS_DIR / "logistic_regression.pkl")

    xgb_params = MLConfig.XGB_PARAMS
    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=xgb_params.get("n_estimators", 200),
        max_depth=xgb_params.get("max_depth", 8),
        learning_rate=xgb_params.get("learning_rate", 0.1),
        subsample=xgb_params.get("subsample", 0.8),
        colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
        random_state=xgb_params.get("random_state", 42),
        n_jobs=xgb_params.get("n_jobs", -1),
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, PathsConfig.MODELS_DIR / "xgboost.pkl")

    PathsConfig.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_test.to_csv(PathsConfig.PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_test.to_csv(PathsConfig.PROCESSED_DATA_DIR / "y_test.csv", index=False)

    print("All models trained and saved to models/ directory.")
    return True


if __name__ == "__main__":
    train_and_save_models()
