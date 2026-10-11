import json
import os
from typing import Any

from config import FraudRulesConfig, LoanConfig, PathsConfig


def load_data(filepath: str) -> list[dict[str, Any]]:
    if not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        data = json.load(f)
        return data if isinstance(data, list) else []


def get_customer_accounts(customer_id: str) -> dict[str, Any]:
    customers = load_data(str(PathsConfig.DATA_DIR / "customers.json"))
    for cust in customers:
        if cust["customer_id"] == customer_id:
            return {"accounts": cust["accounts"]}
    return {"error": "Customer not found."}


def get_recent_transactions(account_id: str, limit: int = 5) -> list[dict[str, Any]]:
    transactions = load_data(str(PathsConfig.DATA_DIR / "transactions.json"))
    account_txns = [tx for tx in transactions if tx["account_id"] == account_id]
    account_txns.sort(key=lambda x: x["date"], reverse=True)
    return account_txns[:limit]


def calculate_emi(principal: float, annual_rate: float, months: int) -> float:
    if principal <= 0 or months <= 0:
        return 0.0
    monthly_rate = annual_rate / 12
    if monthly_rate == 0:
        return principal / months
    emi = principal * monthly_rate * (1 + monthly_rate) ** months / ((1 + monthly_rate) ** months - 1)
    return round(emi, 2)


def check_fraud_risk(transaction_amount: float, transaction_description: str) -> dict[str, Any]:
    risk_score = 0.1
    reasons: list[str] = []

    if transaction_amount < FraudRulesConfig.LARGE_WITHDRAWAL_THRESHOLD:
        risk_score += FraudRulesConfig.LARGE_WITHDRAWAL_SCORE
        reasons.append(f"Large withdrawal (>{abs(FraudRulesConfig.LARGE_WITHDRAWAL_THRESHOLD)})")

    desc_lower = transaction_description.lower()
    if "unknown" in desc_lower:
        risk_score += FraudRulesConfig.UNKNOWN_MERCHANT_SCORE
        reasons.append("Unknown merchant")

    for country in FraudRulesConfig.HIGH_RISK_COUNTRIES:
        if country.lower() in desc_lower:
            risk_score += 0.2
            reasons.append(f"High-risk country/region: {country}")
            break

    threshold = getattr(FraudRulesConfig, 'THRESHOLD', 0.4)
    is_fraudulent = risk_score >= threshold

    return {
        "risk_score": round(min(risk_score, 1.0), 2),
        "threshold_used": threshold,
        "is_suspicious": is_fraudulent,
        "reason": "; ".join(reasons) if reasons else "Normal pattern",
        "method": "rule_based"
    }


def calculate_loan_eligibility(customer_id: str, requested_amount: float) -> dict[str, Any]:
    customers = load_data(str(PathsConfig.DATA_DIR / "customers.json"))
    customer = next((c for c in customers if c["customer_id"] == customer_id), None)

    if not customer:
        return {"error": "Customer not found."}

    monthly_income = customer["monthly_income"]
    existing_emi = customer["existing_loan_emi"]
    credit_score = customer["credit_score"]

    max_emi = (monthly_income * LoanConfig.MAX_DTI_RATIO) - existing_emi

    estimated_emi = calculate_emi(
        requested_amount,
        LoanConfig.INTEREST_RATE_ANNUAL,
        LoanConfig.LOAN_TERM_MONTHS
    )

    if credit_score >= LoanConfig.MIN_CREDIT_SCORE_PRIME and estimated_emi <= max_emi:
        return {
            "eligible": True,
            "max_approved_amount": requested_amount,
            "estimated_emi": estimated_emi,
            "credit_tier": "prime",
            "dti_ratio": round((existing_emi + estimated_emi) / monthly_income, 4)
        }
    elif credit_score >= LoanConfig.MIN_CREDIT_SCORE_SUBPRIME and estimated_emi <= max_emi * 0.8:
        reduced_amount = max_emi * 0.8 / (LoanConfig.INTEREST_RATE_ANNUAL / 12 * (1 + LoanConfig.INTEREST_RATE_ANNUAL / 12) ** LoanConfig.LOAN_TERM_MONTHS / ((1 + LoanConfig.INTEREST_RATE_ANNUAL / 12) ** LoanConfig.LOAN_TERM_MONTHS - 1))
        return {
            "eligible": True,
            "max_approved_amount": round(reduced_amount, 2),
            "estimated_emi": calculate_emi(reduced_amount, LoanConfig.INTEREST_RATE_ANNUAL, LoanConfig.LOAN_TERM_MONTHS),
            "credit_tier": "subprime",
            "reason": "Approved for reduced amount due to credit score",
            "dti_ratio": round((existing_emi + calculate_emi(reduced_amount, LoanConfig.INTEREST_RATE_ANNUAL, LoanConfig.LOAN_TERM_MONTHS)) / monthly_income, 4)
        }
    elif credit_score < LoanConfig.MIN_CREDIT_SCORE_SUBPRIME:
        return {
            "eligible": False,
            "reason": f"Credit score too low ({credit_score}). Minimum required: {LoanConfig.MIN_CREDIT_SCORE_SUBPRIME}",
            "credit_tier": "declined"
        }
    else:
        return {
            "eligible": False,
            "reason": "Debt-to-income ratio too high for requested amount",
            "max_affordable_emi": round(max_emi, 2),
            "requested_emi": estimated_emi,
            "dti_ratio": round((existing_emi + estimated_emi) / monthly_income, 4)
        }


def get_customer_by_id(customer_id: str) -> dict[str, Any] | None:
    customers = load_data(str(PathsConfig.DATA_DIR / "customers.json"))
    return next((c for c in customers if c["customer_id"] == customer_id), None)


def get_all_customers() -> list[dict[str, Any]]:
    return load_data(str(PathsConfig.DATA_DIR / "customers.json"))


def get_all_transactions() -> list[dict[str, Any]]:
    return load_data(str(PathsConfig.DATA_DIR / "transactions.json"))
