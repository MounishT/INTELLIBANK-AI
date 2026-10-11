# Project Strategy & Architecture: Multi-Agent Banking Application

## 1. Problem Statement & Research
**Problem Statement:** Traditional banking systems are highly siloed. Customer queries, fraud detection, and loan approvals are handled by different departments, leading to slow response times, poor customer experience, and high operational costs.
**Research Context:** Recent IEEE research on Multi-Agent Systems (MAS) in FinTech shows that distributed AI agents can reduce processing time significantly by handling concurrent tasks (e.g., checking credit score while verifying identity).

## 2. Target Audience & Users (Who will use this?)
*   **Primary Users (B2C):** Retail banking customers who want instant answers, seamless transfers, and quick loan eligibility checks via a natural language interface.
*   **Secondary Users (B2B):** Bank administrators and compliance officers who monitor the dashboard for flagged fraud transactions and AI audit logs.

## 3. Pain Points (Root Cause)
*   **High Turnaround Time:** Customers wait days for loan approvals due to manual underwriting.
*   **False Positives in Fraud:** Rigid rule-based systems freeze accounts unnecessarily, frustrating customers. 
*   **Fragmented Experience:** Customers have to navigate complex menus instead of just asking for what they want.

## 4. Dataset Strategy (No Synthetic Data)
Instead of generating fake data, we will use open-source, real-world anonymized datasets to power our agents' backend logic:
*   **Risk/Fraud Agent:** **Kaggle Credit Card Fraud Detection Dataset** (European cardholders). This provides real transaction features (V1-V28, Time, Amount).
*   **Credit/Loan Agent:** **Kaggle Loan Prediction Dataset** (Applicant income, Co-applicant income, Loan amount, Credit history).

## 5. Agentic AI & ML Algorithms (Existing vs. Proposing)
*   **Existing System:** Rule-based routing (e.g., "Press 1 for loans"). Static threshold for fraud (e.g., >$5000 is flagged).
*   **Proposing System (Agentic AI):**
    *   **Supervisor Agent (Algorithm: LLM Intent Classification):** Uses a Large Language Model (e.g., GPT-4 / Gemini) with Zero-Shot classification to dynamically route user text to the correct sub-agent.
    *   **Risk Agent (Algorithm: XGBoost / Random Forest):** Instead of rules, we train an ML model on the Fraud dataset. 
    *   **Algorithm Justification:** We will test Logistic Regression, Random Forest, and XGBoost. Random Forest usually provides the best balance of accuracy and recall for highly imbalanced fraud data.

## 6. Threshold Justification (Why 0.4 instead of 0.3?)
In our Fraud Detection model, the output is a probability (0.0 to 1.0). 
*   If we set the threshold to **0.3**, the model is overly sensitive. It catches more fraud (High Recall) but flags too many normal transactions (False Positives), angering customers.
*   If we set the threshold to **0.8**, it misses too much fraud (Low Recall).
*   **Optimal Threshold (e.g., 0.4 or 0.5):** We will determine this using the **F1-Score** and the **Precision-Recall Curve** on our real dataset to balance customer experience with bank security.

## 7. Business Analysis & Revenue Stream
*   **Revenue Model:** **B2B SaaS (Software as a Service) with Tiered API Usage.**
*   Instead of charging retail customers, we sell this Agentic Architecture to mid-sized Credit Unions and Regional Banks. They pay a base subscription fee + a "pay-per-API-call" model for high-volume agent routing.
*   **Capital Impact:** Reduces customer support headcount costs by 40% while increasing loan origination rates due to instant pre-approvals.

## 8. Additional Features (Scale-Up for Real Life)
*   **Voice Integration:** Allowing users to speak to the agents.
*   **Biometric Auth:** Secure login before the Supervisor agent accepts commands.
*   **Human-in-the-Loop (HITL):** If the Risk Agent's confidence is between 0.4 and 0.6, it escalates to a human employee rather than auto-rejecting.

## 9. Tools & Strategy Summary
*   **UI/Frontend:** `Streamlit` (User-friendly, centralized, secure dashboard).
*   **Agentic Framework:** `LangGraph` and `LangChain` (Manages state and workflow between Supervisor and Sub-agents).
*   **Machine Learning (for specialized agents):** `Scikit-learn` and `Pandas` (for training the Fraud/Loan models on real datasets).
*   **LLM Engine:** OpenAI API or Gemini API (for natural language comprehension).
