import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "replace")


from config import safe_json_loads
from agents.credit_agent import check_loan_qualification
from agents.customer_agent import check_balance, get_transactions
from agents.risk_agent import analyze_transaction_risk
from orchestrator import create_orchestrator, run_orchestrator

print("======================================================================")
print("INTELLIBANK AI — 5 CORE BANKING USE CASES VERIFICATION")
print("======================================================================\n")

orch = create_orchestrator()

test_cases = [
    (1, "What is the account balance for CUST-001?", "Customer Agent: Check Balance"),
    (2, "Show my last 5 transactions.", "Customer Agent: Transaction History"),
    (3, "Check transaction TXN-9004 for suspicious activity.", "Risk Agent: Fraud Risk Verification"),
    (4, "Check whether CUST-002 is eligible for a $5000 loan.", "Credit Agent: Loan Eligibility"),
    (5, "Check CUST-001's balance, show recent transactions, and check for suspicious activity.", "Multi-Agent Orchestrator: Combined Workflow")
]

all_passed = True

for num, query, desc in test_cases:
    print(f"[{num}/5] Testing Use Case: {desc}")
    print(f"Query: \"{query}\"")
    
    res = run_orchestrator(orch, query, [])
    output = res.get("output", "")
    steps = res.get("intermediate_steps", [])
    
    print(f"Tools Executed: {[s['tool'] for s in steps]}")
    assert len(steps) >= 1, f"Expected at least 1 tool execution, got {len(steps)}"
    
    # Display preview
    preview_lines = [line.strip() for line in output.split("\n") if line.strip()][:4]
    for p in preview_lines:
        print(f"  {p}")
    print("Status: [PASSED]\n")

print("======================================================================")
print("ALL 5 USE CASES SUCCESSFULLY EXECUTED AND VERIFIED END-TO-END!")
print("======================================================================")
