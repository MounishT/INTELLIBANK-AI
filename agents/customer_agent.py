import json

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import ToolResult
from tools.banking_api import get_customer_accounts, get_recent_transactions


class CheckBalanceInput(BaseModel):
    customer_id: str = Field(description="The customer ID to check balance for")


class GetTransactionsInput(BaseModel):
    account_id: str = Field(description="The account ID to get transactions for")
    limit: int = Field(default=5, description="Maximum number of transactions to return")


@tool(args_schema=CheckBalanceInput)
def check_balance(customer_id: str) -> str:
    """Get the account balance(s) for a customer. Input should be the customer ID."""
    result = get_customer_accounts(customer_id)
    if "error" in result:
        return json.dumps(ToolResult(success=False, error=result["error"]).model_dump())
    return json.dumps(ToolResult(success=True, data={"accounts": result["accounts"]}).model_dump())


@tool(args_schema=GetTransactionsInput)
def get_transactions(account_id: str, limit: int = 5) -> str:
    """Get the recent transactions for a specific account. Input should be the account ID."""
    result = get_recent_transactions(account_id, limit)
    return json.dumps(ToolResult(success=True, data={"transactions": result}).model_dump())


def get_customer_tools():
    return [check_balance, get_transactions]
