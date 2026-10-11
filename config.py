import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

PROJECT_ROOT = Path(__file__).parent


def get_project_root() -> Path:
    return PROJECT_ROOT


@lru_cache
def load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


config: dict[str, Any] = load_config()


def get_config(key: str, default: Any = None) -> Any:
    keys = key.split(".")
    value: Any = config
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
        else:
            return default
        if value is None:
            return default
    return value


class AppConfig:
    NAME = get_config("app.name", "IntelliBank AI")
    VERSION = get_config("app.version", "1.0.0")
    DEBUG = get_config("app.debug", True)


class LLMConfig:
    MODEL = os.getenv("OPENAI_MODEL", get_config("llm.model", "gpt-4o-mini"))
    TEMPERATURE = get_config("llm.temperature", 0)
    MAX_TOKENS = get_config("llm.max_tokens", 2000)


class AuthConfig:
    DEFAULT_USERNAME = get_config("auth.default_username", "admin@intellibank.com")
    SESSION_TIMEOUT_MINUTES = get_config("auth.session_timeout_minutes", 60)


class PathsConfig:
    DATA_DIR = PROJECT_ROOT / get_config("paths.data_dir", "data")
    MODELS_DIR = PROJECT_ROOT / get_config("paths.models_dir", "models")
    REPORTS_DIR = PROJECT_ROOT / get_config("paths.reports_dir", "reports")
    RAW_DATA_DIR = PROJECT_ROOT / get_config("paths.raw_data_dir", "data/raw")
    PROCESSED_DATA_DIR = PROJECT_ROOT / get_config("paths.processed_data_dir", "data/processed")


class MLConfig:
    FRAUD_THRESHOLD = get_config("ml.fraud_threshold", 0.4)
    RF_PARAMS = get_config("ml.random_forest", {})
    XGB_PARAMS = get_config("ml.xgboost", {})
    LR_PARAMS = get_config("ml.logistic_regression", {})
    SPLIT_PARAMS = get_config("ml.train_test_split", {})


class LoanConfig:
    MAX_DTI_RATIO = get_config("loan.max_dti_ratio", 0.4)
    INTEREST_RATE_ANNUAL = get_config("loan.interest_rate_annual", 0.12)
    LOAN_TERM_MONTHS = get_config("loan.loan_term_months", 60)
    MIN_CREDIT_SCORE_PRIME = get_config("loan.min_credit_score_prime", 700)
    MIN_CREDIT_SCORE_SUBPRIME = get_config("loan.min_credit_score_subprime", 600)


class FraudRulesConfig:
    LARGE_WITHDRAWAL_THRESHOLD = get_config("fraud_rules.large_withdrawal_threshold", -1000)
    LARGE_WITHDRAWAL_SCORE = get_config("fraud_rules.large_withdrawal_score", 0.4)
    UNKNOWN_MERCHANT_SCORE = get_config("fraud_rules.unknown_merchant_score", 0.3)
    HIGH_RISK_COUNTRIES = get_config("fraud_rules.high_risk_countries", ["Unknown", "International"])


class LoggingConfig:
    LEVEL = get_config("logging.level", "INFO")
    FORMAT = get_config("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    FILE = PROJECT_ROOT / get_config("logging.file", "logs/intellibank.log")


class CustomerInfo(BaseModel):
    customer_id: str
    name: str
    email: str
    accounts: list[dict[str, Any]]
    credit_score: int
    monthly_income: float
    existing_loan_emi: float


class AccountInfo(BaseModel):
    account_id: str
    type: str
    balance: float


class TransactionInfo(BaseModel):
    transaction_id: str
    account_id: str
    date: str
    description: str
    amount: float
    type: str


class BalanceRequest(BaseModel):
    customer_id: str


class BalanceResponse(BaseModel):
    accounts: list[AccountInfo]


class TransactionsRequest(BaseModel):
    account_id: str
    limit: int = 5


class TransactionsResponse(BaseModel):
    transactions: list[TransactionInfo]


class FraudCheckRequest(BaseModel):
    amount: float
    description: str


class FraudCheckResponse(BaseModel):
    risk_score: float
    threshold_used: float
    is_suspicious: bool
    reason: str


class RealMLFraudRequest(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class RealMLFraudResponse(BaseModel):
    risk_score: float
    threshold_used: float
    is_suspicious: bool
    recommendation: str
    error: str | None = None


class LoanRequest(BaseModel):
    customer_id: str
    amount: float


class LoanResponse(BaseModel):
    eligible: bool
    max_approved_amount: float | None = None
    estimated_emi: float | None = None
    reason: str | None = None
    error: str | None = None


class ToolResult(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


def safe_json_loads(data: Any) -> Any:
    """Safely parse JSON string, or return data if already dict/list/int/float."""
    import json
    if isinstance(data, (dict, list)):
        return data
    if isinstance(data, (str, bytes, bytearray)):
        try:
            return json.loads(data)
        except Exception:
            return data
    return data

