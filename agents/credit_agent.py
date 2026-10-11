import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import ToolResult
from tools.banking_api import calculate_loan_eligibility


class CheckLoanQualificationInput(BaseModel):
    customer_id: str = Field(description="The customer ID")
    amount: float = Field(description="The requested loan amount")


@tool(args_schema=CheckLoanQualificationInput)
def check_loan_qualification(customer_id: str, amount: float) -> str:
    """Check if a customer is eligible for a loan of a specific amount."""
    result = calculate_loan_eligibility(customer_id, amount)
    if "error" in result:
        return json.dumps(ToolResult(success=False, error=result["error"]).model_dump())
    return json.dumps(ToolResult(success=True, data=result).model_dump())


def get_credit_tools():
    return [check_loan_qualification]
