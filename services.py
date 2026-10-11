import logging
from typing import Any

from config import ToolResult
from ml.predict import predict_fraud, predict_fraud_batch
from tools.banking_api import (
    calculate_loan_eligibility,
    check_fraud_risk,
    get_all_customers,
    get_all_transactions,
    get_customer_accounts,
    get_customer_by_id,
    get_recent_transactions,
)

logger = logging.getLogger(__name__)

# banking_api functions return Python dicts directly, not JSON strings
# So no json.loads() needed in service layer


class CustomerService:
    @staticmethod
    def get_customer(customer_id: str) -> dict[str, Any] | None:
        return get_customer_by_id(customer_id)

    @staticmethod
    def get_all_customers() -> list[dict[str, Any]]:
        return get_all_customers()

    @staticmethod
    def get_customer_accounts(customer_id: str) -> ToolResult:
        result = get_customer_accounts(customer_id)
        if "error" in result:
            return ToolResult(success=False, error=result["error"])
        return ToolResult(success=True, data={"accounts": result["accounts"]})

    @staticmethod
    def get_account_transactions(account_id: str, limit: int = 5) -> ToolResult:
        result = get_recent_transactions(account_id, limit)
        return ToolResult(success=True, data={"transactions": result})


class FraudService:
    @staticmethod
    def check_rule_based(amount: float, description: str) -> ToolResult:
        result = check_fraud_risk(amount, description)
        return ToolResult(success=True, data=result)

    @staticmethod
    def check_ml(transaction_data: dict[str, Any], threshold: float | None = None) -> ToolResult:
        result = predict_fraud(transaction_data, threshold)
        if "error" in result:
            return ToolResult(success=False, error=result["error"])
        return ToolResult(success=True, data=result)

    @staticmethod
    def check_batch_ml(transactions: list[dict[str, Any]], threshold: float | None = None) -> list[ToolResult]:
        results = predict_fraud_batch(transactions, threshold)
        return [ToolResult(success="error" not in r, data=r if "error" not in r else None, error=r.get("error")) for r in results]


class CreditService:
    @staticmethod
    def check_eligibility(customer_id: str, amount: float) -> ToolResult:
        result = calculate_loan_eligibility(customer_id, amount)
        if "error" in result:
            return ToolResult(success=False, error=result["error"])
        return ToolResult(success=True, data=result)


class DashboardService:
    @staticmethod
    def get_summary_stats() -> dict[str, Any]:
        customers = get_all_customers()
        transactions = get_all_transactions()
        total_balance = sum(
            acc["balance"]
            for cust in customers
            for acc in cust.get("accounts", [])
        )
        return {
            "total_customers": len(customers),
            "total_transactions": len(transactions),
            "total_balance": round(total_balance, 2),
            "active_customers": len([c for c in customers if c.get("accounts")])
        }

    @staticmethod
    def get_recent_transactions(limit: int = 10) -> list[dict[str, Any]]:
        transactions = get_all_transactions()
        transactions.sort(key=lambda x: x["date"], reverse=True)
        return transactions[:limit]

    @staticmethod
    def get_fraud_analysis() -> list[dict[str, Any]]:
        transactions = get_all_transactions()
        results = []
        for tx in transactions:
            fraud_result = check_fraud_risk(tx["amount"], tx["description"])
            # check_fraud_risk returns a dict directly, not a JSON string
            fraud_data = fraud_result
            results.append({
                **tx,
                "risk_score": fraud_data.get("risk_score", 0),
                "is_suspicious": fraud_data.get("is_suspicious", False),
                "reason": fraud_data.get("reason", "")
            })
        return results
