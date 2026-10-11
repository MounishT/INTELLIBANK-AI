# IntelliBank AI — Intelligent Banking System with Agent Orchestration

**IntelliBank AI** is an enterprise-grade banking intelligence platform featuring autonomous Multi-Agent Orchestration, automated risk surveillance, rule-based credit underwriting, and Machine Learning fraud detection on the real Kaggle credit card dataset.

Designed with an accessible, high-contrast fintech design system in Streamlit, IntelliBank AI bridges retail banking workflows with verifiable machine learning models.

---

## Architecture Overview

1. **User Interface (Streamlit):**
   - High-contrast, light-themed fintech styling with slate-900 typography, royal blue actions, and subtle borders.
   - 11 dedicated operations pages covering accounts, transactions, AI assistance, surveillance, loan decisioning, and ML benchmarks.
   - Built-in session security with PBKDF2 password hashing and token expiry.

2. **Multi-Agent Orchestrator (Supervisor Pattern):**
   - **LangGraph ReAct Agent:** When configured with an OpenAI API key, uses LangGraph's dynamic supervisor to reason, route, and invoke specialized tools.
   - **Deterministic Supervisor Router:** Offline-capable fallback engine that accurately parses intents and executes the exact same underlying tools (`check_balance`, `get_transactions`, `analyze_transaction_risk`, `check_loan_qualification`), generating full execution audit traces without requiring paid API keys.

3. **Specialized Banking Agents:**
   - **Customer Agent:** Interacts with core account ledgers to retrieve balances and statement activity.
   - **Risk Agent:** Dual-mode surveillance providing rule-based anomaly detection on retail transactions and PCA vector predictions via trained ML classifiers.
   - **Credit Agent:** Underwrites consumer loan requests against credit score tiers, debt-to-income (DTI) ceilings, and compound interest EMI calculations.

4. **Real Machine Learning Pipeline:**
   - Preprocessing, scaling (Time and Amount only, preserving PCA V1–V28), stratified splitting, and training for **Logistic Regression**, **Random Forest**, and **XGBoost**.
   - Performance evaluation comparing Precision, Recall, F1-Score, PR-AUC, ROC-AUC, and full confusion matrices.
   - Optimal threshold tuner balancing false positive rate and fraud catch rate.

---

## 11 Operations Workspaces

| # | Workspace Page | Key Functionality |
|---|---|---|
| 1 | **Login & Logout** | Secure portal sign-in, session tokens, pre-filled dev credentials |
| 2 | **Dashboard Overview** | Executive KPI metric cards, portfolio balance totals, recent activity feed |
| 3 | **AI Banking Assistant** | Chat interface with quick prompt chips, customer context, and expandable tool execution trace |
| 4 | **Customer Accounts** | Client profiles, credit tiers, income, debt commitments, and account balances |
| 5 | **Transaction History** | Searchable and filterable ledger with account and type selectors |
| 6 | **Fraud Detection** | Simulated retail surveillance with live transaction risk evaluator |
| 7 | **Fraud Investigation Center** | Dataset explorer (Kaggle), model score evaluator, threshold slider, and analyst disposition workflow |
| 8 | **Credit & Loan Assessment** | Interactive loan simulator, EMI computation, DTI analysis, and policy guidelines |
| 9 | **ML Model Performance** | Cross-model benchmark table, confusion matrix cards, and threshold optimizer |
| 10 | **Agent Execution Logs** | Chronological audit trail of all tool invocations with arguments and outputs |
| 11 | **About & Settings** | System architecture, environment status, configuration viewer, and API key manager |

---

## Setup & Quick Start

### 1. Prerequisites & Installation
Ensure Python 3.10+ is installed on your system.

```bash
# Clone the repository and navigate to root directory
cd "IntelliBank AI"

# Install production dependencies
pip install -r requirements.txt
```

### 2. Verify or Train ML Models
The project comes with trained models in `models/` and preprocessed data in `data/raw/creditcard.csv`. To inspect or re-train:

```bash
# Verify dataset presence and class distribution
python scripts/check_dataset.py

# Train Logistic Regression, Random Forest, and XGBoost
python -m ml.train_model

# Evaluate all models and generate reports/model_comparison.csv
python -m ml.evaluate_models
```

### 3. Run Automated Tests
Execute the comprehensive test suite (18 automated tests covering tools, schemas, use cases, and JSON handlers):

```bash
python -m pytest tests/ -v
```

### 4. Verify 5 Core Banking Use Cases
Run the dedicated test script that executes all 5 core use cases end-to-end:

```bash
python scripts/test_use_cases.py
```

### 5. Launch the Application
Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## Authentication & Credentials

- **Portal URL:** `http://localhost:8501`
- **Default Staff Email:** `admin@intellibank.com`
- **Default Password:** `admin123`
- **OpenAI API Key:** Optional. Leave empty to use the built-in Deterministic Multi-Agent Supervisor, or paste a key to activate live LangGraph ReAct model calling.

---

## Verified Test Identifiers

- **Customers:**
  - `CUST-001` (Jane Doe) — Credit Score: 780 (Prime), Income: $6,500/mo, EMI: $500/mo. Accounts: Savings `ACC-1001` ($15,420.50), Checking `ACC-1002` ($2,350.00).
  - `CUST-002` (John Smith) — Credit Score: 620 (Subprime), Income: $3,200/mo, EMI: $800/mo. Account: Checking `ACC-2001` ($450.25).
- **Transactions:**
  - `TXN-9001` ($6,500.00 Salary Deposit, ACC-1001)
  - `TXN-9002` (-$5.25 Coffee House, ACC-1002)
  - `TXN-9003` (-$85.50 Grocery Store, ACC-1002)
  - `TXN-9004` (-$2,500.00 Unknown International Merchant, ACC-1001) — *Flagged as High Risk / Suspicious*

---

## 5 Core Banking Use Cases

The AI Banking Assistant natively resolves the following benchmark scenarios:

1. **Check Account Balance:**
   - *Prompt:* `"What is the account balance for CUST-001?"`
   - *Tool:* `check_balance`
   - *Result:* Returns Savings ($15,420.50), Checking ($2,350.00), and combined portfolio total ($17,770.50).

2. **View Recent Transactions:**
   - *Prompt:* `"Show my last 5 transactions."`
   - *Tool:* `get_transactions`
   - *Result:* Returns timestamped transactions with merchant descriptions and debit/credit formatting.

3. **Detect Suspicious Activity:**
   - *Prompt:* `"Check transaction TXN-9004 for suspicious activity."`
   - *Tool:* `analyze_transaction_risk`
   - *Result:* Evaluates transaction `TXN-9004`, flags high risk (`risk_score=1.0`), and identifies triggered heuristic rules (large withdrawal, unknown merchant, international).

4. **Assess Loan Qualification:**
   - *Prompt:* `"Check whether CUST-002 is eligible for a $5000 loan."`
   - *Tool:* `check_loan_qualification`
   - *Result:* Calculates DTI and EMI, approves facility under subprime guidelines with max approved limit and monthly installment breakdown.

5. **Combined Multi-Agent Workflow:**
   - *Prompt:* `"Check CUST-001's balance, show recent transactions, and check for suspicious activity."`
   - *Tools:* Sequentially invokes `check_balance`, `get_transactions`, and `analyze_transaction_risk`, consolidating the outputs into a unified response with complete execution trace.

---

## Project Structure

```
IntelliBank AI/
├── .streamlit/
│   └── config.toml             # Accessible light fintech theme configuration
├── app.py                      # Main Streamlit application with 11 pages
├── orchestrator.py             # LangGraph & Deterministic Supervisor Orchestrator
├── auth.py                     # PBKDF2 authentication and session management
├── config.py                   # Central settings, paths, and safe JSON utilities
├── config.yaml                 # Configurable thresholds and model parameters
├── services.py                 # Clean business logic and API abstraction layer
├── logging_config.py           # Structured logging configuration
├── agents/
│   ├── customer_agent.py       # Account balance and transaction tools
│   ├── risk_agent.py           # Rule-based and real ML fraud analysis tools
│   └── credit_agent.py         # Underwriting and loan qualification tools
├── tools/
│   └── banking_api.py          # Simulated banking ledger operations
├── ml/
│   ├── data_loader.py          # Kaggle dataset loader and validation
│   ├── train_model.py          # Model training pipeline (LR, RF, XGB)
│   ├── evaluate_models.py      # Cross-model evaluation and metric reporting
│   └── predict.py              # Real-time PCA vector fraud prediction
├── data/
│   ├── customers.json          # Mock customer database
│   ├── transactions.json       # Mock transaction database
│   ├── users.json              # Staff user credentials (salted hash)
│   ├── sessions.json           # Active session storage
│   ├── raw/creditcard.csv      # Kaggle credit card dataset
│   └── processed/              # Saved test splits (X_test.csv, y_test.csv)
├── models/
│   ├── logistic_regression.pkl # Trained Logistic Regression model
│   ├── random_forest.pkl       # Trained Random Forest model
│   ├── xgboost.pkl             # Trained XGBoost model
│   └── scaler.pkl              # Fitted StandardScaler (Time & Amount)
├── reports/
│   └── model_comparison.csv    # Evaluated benchmark metrics
├── scripts/
│   ├── check_dataset.py        # Inspects dataset statistics and distribution
│   └── test_use_cases.py       # Validates all 5 use cases end-to-end
├── tests/
│   └── test_agents.py          # Pytest suite (18 unit and integration tests)
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```