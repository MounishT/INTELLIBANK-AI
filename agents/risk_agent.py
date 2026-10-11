import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import ToolResult
from ml.predict import predict_fraud
from tools.banking_api import check_fraud_risk


class AnalyzeTransactionRiskInput(BaseModel):
    amount: float = Field(description="Transaction amount (negative for debits)")
    description: str = Field(description="Transaction description/merchant name")


class AnalyzeRealMLTransactionRiskInput(BaseModel):
    Time: float = Field(description="Time feature from dataset")
    Amount: float = Field(description="Transaction amount")
    V1: float | None = Field(default=None, description="PCA feature V1")
    V2: float | None = Field(default=None, description="PCA feature V2")
    V3: float | None = Field(default=None, description="PCA feature V3")
    V4: float | None = Field(default=None, description="PCA feature V4")
    V5: float | None = Field(default=None, description="PCA feature V5")
    V6: float | None = Field(default=None, description="PCA feature V6")
    V7: float | None = Field(default=None, description="PCA feature V7")
    V8: float | None = Field(default=None, description="PCA feature V8")
    V9: float | None = Field(default=None, description="PCA feature V9")
    V10: float | None = Field(default=None, description="PCA feature V10")
    V11: float | None = Field(default=None, description="PCA feature V11")
    V12: float | None = Field(default=None, description="PCA feature V12")
    V13: float | None = Field(default=None, description="PCA feature V13")
    V14: float | None = Field(default=None, description="PCA feature V14")
    V15: float | None = Field(default=None, description="PCA feature V15")
    V16: float | None = Field(default=None, description="PCA feature V16")
    V17: float | None = Field(default=None, description="PCA feature V17")
    V18: float | None = Field(default=None, description="PCA feature V18")
    V19: float | None = Field(default=None, description="PCA feature V19")
    V20: float | None = Field(default=None, description="PCA feature V20")
    V21: float | None = Field(default=None, description="PCA feature V21")
    V22: float | None = Field(default=None, description="PCA feature V22")
    V23: float | None = Field(default=None, description="PCA feature V23")
    V24: float | None = Field(default=None, description="PCA feature V24")
    V25: float | None = Field(default=None, description="PCA feature V25")
    V26: float | None = Field(default=None, description="PCA feature V26")
    V27: float | None = Field(default=None, description="PCA feature V27")
    V28: float | None = Field(default=None, description="PCA feature V28")


@tool(args_schema=AnalyzeTransactionRiskInput)
def analyze_transaction_risk(amount: float, description: str) -> str:
    """Analyze a mock transaction for potential fraud risk using rule-based logic."""
    result = check_fraud_risk(amount, description)
    return json.dumps(ToolResult(success=True, data=result).model_dump())


@tool(args_schema=AnalyzeRealMLTransactionRiskInput)
def analyze_real_ml_transaction_risk(
    Time: float, Amount: float,
    V1: float | None = None, V2: float | None = None, V3: float | None = None,
    V4: float | None = None, V5: float | None = None, V6: float | None = None,
    V7: float | None = None, V8: float | None = None, V9: float | None = None,
    V10: float | None = None, V11: float | None = None, V12: float | None = None,
    V13: float | None = None, V14: float | None = None, V15: float | None = None,
    V16: float | None = None, V17: float | None = None, V18: float | None = None,
    V19: float | None = None, V20: float | None = None, V21: float | None = None,
    V22: float | None = None, V23: float | None = None, V24: float | None = None,
    V25: float | None = None, V26: float | None = None, V27: float | None = None,
    V28: float | None = None
) -> str:
    """Analyze a real transaction for fraud using the trained ML model.
    Input must contain Time, Amount, and V1-V28 features from the Kaggle dataset."""
    transaction_data = {
        "Time": Time, "Amount": Amount,
        "V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5, "V6": V6,
        "V7": V7, "V8": V8, "V9": V9, "V10": V10, "V11": V11, "V12": V12,
        "V13": V13, "V14": V14, "V15": V15, "V16": V16, "V17": V17, "V18": V18,
        "V19": V19, "V20": V20, "V21": V21, "V22": V22, "V23": V23, "V24": V24,
        "V25": V25, "V26": V26, "V27": V27, "V28": V28
    }
    missing = [k for k, v in transaction_data.items() if v is None]
    if missing:
        return json.dumps(ToolResult(success=False, error=f"Missing features: {missing}").model_dump())
    result = predict_fraud(transaction_data)
    if "error" in result:
        return json.dumps(ToolResult(success=False, error=result["error"]).model_dump())
    return json.dumps(ToolResult(success=True, data=result).model_dump())


def get_risk_tools():
    return [analyze_transaction_risk, analyze_real_ml_transaction_risk]
