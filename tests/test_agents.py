import json
from typing import Any

from agents.credit_agent import check_loan_qualification
from agents.customer_agent import check_balance, get_transactions
from agents.risk_agent import analyze_real_ml_transaction_risk, analyze_transaction_risk
from config import ToolResult


def parse_tool_result(response: str) -> ToolResult:
    data = json.loads(response)
    return ToolResult(**data)


def assert_data(result: ToolResult) -> dict[str, Any]:
    """Helper to assert result.data is not None and return it."""
    assert result.data is not None
    return result.data


class TestCustomerAgent:
    def test_check_balance_valid_customer(self):
        response = check_balance.invoke({"customer_id": "CUST-001"})
        result = parse_tool_result(response)
        assert result.success is True
        data = assert_data(result)
        assert "accounts" in data
        accounts = data["accounts"]
        assert len(accounts) == 2
        assert accounts[0]["account_id"] == "ACC-1001"
        assert accounts[0]["balance"] == 15420.50

    def test_check_balance_invalid_customer(self):
        response = check_balance.invoke({"customer_id": "INVALID-ID"})
        result = parse_tool_result(response)
        assert result.success is False
        assert result.error is not None
        assert "Customer not found" in result.error

    def test_get_transactions_valid_account(self):
        response = get_transactions.invoke({"account_id": "ACC-1002", "limit": 5})
        result = parse_tool_result(response)
        assert result.success is True
        data = assert_data(result)
        assert "transactions" in data
        transactions = data["transactions"]
        assert len(transactions) == 2
        assert transactions[0]["amount"] == -5.25


class TestRiskAgent:
    def test_analyze_transaction_risk_safe(self):
        response = analyze_transaction_risk.invoke({
            "amount": -50.0,
            "description": "Coffee Shop"
        })
        result = parse_tool_result(response)
        assert result.success is True
        data = assert_data(result)
        assert data["is_suspicious"] is False
        assert data["risk_score"] < 0.4

    def test_analyze_transaction_risk_suspicious_large_amount(self):
        response = analyze_transaction_risk.invoke({
            "amount": -2500.0,
            "description": "Regular Store"
        })
        result = parse_tool_result(response)
        assert result.success is True
        data = assert_data(result)
        assert data["is_suspicious"] is True
        assert data["risk_score"] >= 0.4

    def test_analyze_transaction_risk_suspicious_unknown_merchant(self):
        response = analyze_transaction_risk.invoke({
            "amount": -100.0,
            "description": "Unknown Merchant"
        })
        result = parse_tool_result(response)
        assert result.success is True
        data = assert_data(result)
        assert data["is_suspicious"] is True

    def test_analyze_real_ml_transaction_risk_missing_features(self):
        response = analyze_real_ml_transaction_risk.invoke({
            "Time": 1.0, "Amount": 100.0,
            "V1": 0.1, "V2": 0.2, "V3": 0.3, "V4": 0.4, "V5": 0.5,
            "V6": 0.6, "V7": 0.7, "V8": 0.8, "V9": 0.9, "V10": 1.0,
            "V11": 1.1, "V12": 1.2, "V13": 1.3, "V14": 1.4, "V15": 1.5,
            "V16": 1.6, "V17": 1.7, "V18": 1.8, "V19": 1.9, "V20": 2.0,
            "V21": 2.1, "V22": 2.2, "V23": 2.3, "V24": 2.4, "V25": 2.5,
            "V26": 2.6, "V27": 2.7
        })
        result = parse_tool_result(response)
        assert result.success is False
        assert result.error is not None
        assert "V28" in result.error or "Missing features" in result.error


class TestCreditAgent:
    def test_check_loan_qualification_eligible_prime(self):
        response = check_loan_qualification.invoke({"customer_id": "CUST-001", "amount": 5000})
        result = parse_tool_result(response)
        assert result.success is True
        data = assert_data(result)
        assert data["eligible"] is True
        assert data["credit_tier"] == "prime"

    def test_check_loan_qualification_eligible_subprime_reduced(self):
        response = check_loan_qualification.invoke({"customer_id": "CUST-002", "amount": 1000})
        result = parse_tool_result(response)
        assert result.success is True
        data = assert_data(result)
        assert data["eligible"] is True
        assert data["credit_tier"] == "subprime"

    def test_check_loan_qualification_ineligible_dti(self):
        # CUST-002 has credit_score=620 (subprime) but 50000 loan exceeds DTI
        response = check_loan_qualification.invoke({"customer_id": "CUST-002", "amount": 50000})
        result = parse_tool_result(response)
        assert result.success is True
        data = assert_data(result)
        assert data["eligible"] is False
        assert "Debt-to-income ratio too high" in data["reason"]
        assert "dti_ratio" in data

    def test_check_loan_qualification_invalid_customer(self):
        response = check_loan_qualification.invoke({"customer_id": "INVALID-ID", "amount": 5000})
        result = parse_tool_result(response)
        assert result.success is False
        assert result.error is not None
        assert "Customer not found" in result.error


class TestToolResultSchema:
    def test_all_tools_return_tool_result(self):
        responses = [
            check_balance.invoke({"customer_id": "CUST-001"}),
            get_transactions.invoke({"account_id": "ACC-1001"}),
            analyze_transaction_risk.invoke({"amount": -100, "description": "Test"}),
            check_loan_qualification.invoke({"customer_id": "CUST-001", "amount": 1000}),
        ]
        for response in responses:
            result = parse_tool_result(response)
            assert isinstance(result, ToolResult)
            assert hasattr(result, "success")
            assert hasattr(result, "data")
            assert hasattr(result, "error")


class TestOrchestratorUseCases:
    """Verify all 5 core banking use cases execute end-to-end with real tools."""

    def setup_method(self):
        from orchestrator import create_orchestrator
        self.orchestrator = create_orchestrator()

    def test_use_case_1_account_balance(self):
        from orchestrator import run_orchestrator
        res = run_orchestrator(self.orchestrator, "What is the account balance for CUST-001?", [])
        assert "output" in res
        assert "CUST-001" in res["output"] or "Savings" in res["output"]
        assert len(res["intermediate_steps"]) >= 1
        tools_called = [s["tool"] for s in res["intermediate_steps"]]
        assert "check_balance" in tools_called

    def test_use_case_2_recent_transactions(self):
        from orchestrator import run_orchestrator
        res = run_orchestrator(self.orchestrator, "Show my last 5 transactions.", [])
        assert "output" in res
        assert len(res["intermediate_steps"]) >= 1
        tools_called = [s["tool"] for s in res["intermediate_steps"]]
        assert "get_transactions" in tools_called

    def test_use_case_3_suspicious_activity_txn9004(self):
        from orchestrator import run_orchestrator
        res = run_orchestrator(self.orchestrator, "Check transaction TXN-9004 for suspicious activity.", [])
        assert "output" in res
        assert "Fraud Risk Assessment" in res["output"] or "Risk Score" in res["output"]
        tools_called = [s["tool"] for s in res["intermediate_steps"]]
        assert "analyze_transaction_risk" in tools_called

    def test_use_case_4_loan_eligibility(self):
        from orchestrator import run_orchestrator
        res = run_orchestrator(self.orchestrator, "Check whether CUST-002 is eligible for a $5000 loan.", [])
        assert "output" in res
        assert "Loan Eligibility" in res["output"] or "Approved" in res["output"]
        tools_called = [s["tool"] for s in res["intermediate_steps"]]
        assert "check_loan_qualification" in tools_called

    def test_use_case_5_multi_action_workflow(self):
        from orchestrator import run_orchestrator
        res = run_orchestrator(
            self.orchestrator,
            "Check CUST-001's balance, show recent transactions, and check for suspicious activity.",
            []
        )
        assert "output" in res
        tools_called = [s["tool"] for s in res["intermediate_steps"]]
        assert "check_balance" in tools_called
        assert "get_transactions" in tools_called
        assert "analyze_transaction_risk" in tools_called


class TestSafeJsonHelper:
    def test_safe_json_loads(self):
        from config import safe_json_loads
        assert safe_json_loads({"a": 1}) == {"a": 1}
        assert safe_json_loads('{"a": 1}') == {"a": 1}
        assert safe_json_loads([1, 2, 3]) == [1, 2, 3]
        assert safe_json_loads("not valid json") == "not valid json"
        assert safe_json_loads(None) is None

