import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from auth import auth_manager
from config import AppConfig, LLMConfig, LoanConfig, PathsConfig, config, safe_json_loads
from logging_config import get_logger, setup_logging
from orchestrator import create_orchestrator, run_orchestrator
from services import CreditService, CustomerService, DashboardService, FraudService

setup_logging()
logger = get_logger(__name__)

st.set_page_config(
    page_title=f"{AppConfig.NAME} — Intelligent Banking Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# PROFESSIONAL FINTECH THEME & ACCESSIBILITY STYLING
# ==============================================================================
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Enforce light background and dark navy text everywhere */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Force high-contrast text on all standard elements */
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown, .stText {
        color: #0f172a !important;
    }

    /* Secondary muted text */
    .text-muted {
        color: #475569 !important;
        font-size: 0.875rem;
    }

    /* Main container max width & spacing */
    .main .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #1e293b !important;
        font-weight: 500;
        padding: 4px 0;
    }

    /* Premium Fintech Cards */
    .ib-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1.25rem;
    }

    .ib-card-compact {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }

    /* Metrics Cards */
    .ib-metric-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        border-top: 4px solid #1d4ed8;
    }
    .ib-metric-card-danger {
        border-top-color: #dc2626 !important;
    }
    .ib-metric-card-success {
        border-top-color: #059669 !important;
    }
    .ib-metric-card-amber {
        border-top-color: #d97706 !important;
    }

    .ib-metric-label {
        font-size: 0.8125rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569 !important;
        margin-bottom: 0.25rem;
    }
    .ib-metric-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: #0f172a !important;
        line-height: 1.2;
    }
    .ib-metric-sub {
        font-size: 0.8125rem;
        color: #64748b !important;
        margin-top: 0.35rem;
    }

    /* Section Headers */
    .ib-page-title {
        font-size: 1.625rem;
        font-weight: 700;
        color: #0f172a !important;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .ib-page-subtitle {
        font-size: 0.9375rem;
        color: #475569 !important;
        margin-bottom: 1.5rem;
    }

    /* Status Badges */
    .ib-badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1.2;
    }
    .ib-badge-blue { background: #eff6ff !important; color: #1d4ed8 !important; border: 1px solid #bfdbfe; }
    .ib-badge-green { background: #ecfdf5 !important; color: #047857 !important; border: 1px solid #a7f3d0; }
    .ib-badge-red { background: #fef2f2 !important; color: #b91c1c !important; border: 1px solid #fecaca; }
    .ib-badge-amber { background: #fffbeb !important; color: #b45309 !important; border: 1px solid #fde68a; }
    .ib-badge-purple { background: #faf5ff !important; color: #7e22ce !important; border: 1px solid #e9d5ff; }

    /* Buttons */
    div.stButton > button {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button:hover {
        background-color: #1e40af !important;
        color: #ffffff !important;
    }
    div.stButton > button:active {
        background-color: #1e3a8a !important;
    }

    /* Secondary outline buttons */
    .stButton.btn-secondary > button {
        background-color: #ffffff !important;
        color: #1d4ed8 !important;
        border: 1px solid #cbd5e1 !important;
    }

    /* Input & Textarea Elements */
    input, textarea, select {
        color: #0f172a !important;
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    input:focus, textarea:focus, select:focus {
        border-color: #1d4ed8 !important;
        box-shadow: 0 0 0 1px #1d4ed8 !important;
    }

    /* Dataframe container */
    div[data-testid="stDataFrame"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* Expander styling */
    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        margin-bottom: 0.75rem !important;
    }

    /* Chat Messages styling */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    }

    /* Code blocks */
    pre, code {
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE MANAGEMENT
# ==============================================================================
def init_session_state():
    defaults = {
        "logged_in": False,
        "session_token": None,
        "username": None,
        "messages": [],
        "orchestrator": None,
        "current_customer_id": "CUST-001",
        "chat_prompt_input": "",
        "dispositions": {}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def login_user(username: str, session_token: str, api_key: str | None = None):
    if api_key and api_key.strip():
        os.environ["OPENAI_API_KEY"] = api_key.strip()
    st.session_state["logged_in"] = True
    st.session_state["session_token"] = session_token
    st.session_state["username"] = username
    try:
        st.session_state["orchestrator"] = create_orchestrator()
    except Exception as e:
        logger.warning("Orchestrator initialization fallback: %s", e)
    st.rerun()


def logout():
    if st.session_state.get("session_token"):
        auth_manager.logout(st.session_state["session_token"])
    st.session_state["logged_in"] = False
    st.session_state["session_token"] = None
    st.session_state["username"] = None
    st.session_state["messages"] = []
    st.session_state["orchestrator"] = None
    st.rerun()


def validate_session() -> bool:
    token = st.session_state.get("session_token")
    if not token:
        return False
    username = auth_manager.validate_session(token)
    if not username:
        st.session_state["logged_in"] = False
        st.session_state["session_token"] = None
        st.session_state["username"] = None
        return False
    st.session_state["username"] = username
    return True


# ==============================================================================
# PAGE 1: LOGIN & AUTHENTICATION
# ==============================================================================
def render_login_page():
    col_left, col_center, col_right = st.columns([1, 1.8, 1])

    with col_center:
        st.markdown("<div style='height: 4vh;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; margin-bottom: 1.5rem;'>
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🏦</div>
            <h1 style='font-size: 2rem; font-weight: 800; color: #0f172a; margin: 0;'>IntelliBank AI</h1>
            <p style='color: #475569; font-size: 0.95rem; margin-top: 0.25rem;'>
                Enterprise Banking Intelligence & Multi-Agent Operations
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.25rem; font-weight: 700; margin-bottom: 0.25rem;'>Staff Portal Sign In</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748b; font-size: 0.85rem; margin-bottom: 1.25rem;'>Enter your banking credentials to access operational services.</p>", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Staff Email / ID", value="admin@intellibank.com", placeholder="admin@intellibank.com")
            password = st.text_input("Password", value="admin123", type="password", placeholder="••••••••")
            api_key_input = st.text_input(
                "OpenAI API Key (Optional — for live LLM ReAct agent)",
                type="password",
                placeholder="sk-... (Leave empty to use built-in Deterministic Supervisor)"
            )

            submit = st.form_submit_button("Sign In to Portal", use_container_width=True)

            if submit:
                if not username or not password:
                    st.error("Please provide both email and password.")
                else:
                    token = auth_manager.authenticate(username, password)
                    if token:
                        login_user(username, token, api_key_input)
                    else:
                        st.error("Invalid credentials. Use admin@intellibank.com / admin123")

        st.markdown("""
        <div style='background: #f1f5f9; padding: 0.85rem; border-radius: 8px; margin-top: 1rem; border: 1px solid #cbd5e1;'>
            <p style='margin: 0; font-size: 0.8rem; color: #334155;'>
                <strong>Development Credentials:</strong><br>
                Email: <code>admin@intellibank.com</code> &nbsp;|&nbsp; Password: <code>admin123</code><br>
                <em>AI Mode: Built-in Multi-Agent Supervisor is active offline without API keys.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# NAVIGATION SIDEBAR
# ==============================================================================
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid #e2e8f0;'>
            <div style='font-size: 1.8rem;'>🏦</div>
            <div>
                <div style='font-weight: 700; font-size: 1.05rem; color: #0f172a;'>IntelliBank AI</div>
                <div style='font-size: 0.75rem; color: #64748b;'>Core Operations MVP</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        user = st.session_state.get("username", "Staff")
        st.markdown(f"""
        <div style='background: #f8fafc; padding: 0.65rem 0.85rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1rem;'>
            <div style='font-size: 0.75rem; color: #64748b;'>ACTIVE SESSION</div>
            <div style='font-size: 0.85rem; font-weight: 600; color: #0f172a;'>{user}</div>
        </div>
        """, unsafe_allow_html=True)

        # Active Customer Selector
        st.markdown("<p style='font-size: 0.8rem; font-weight: 600; color: #475569; margin-bottom: 4px;'>ACTIVE CUSTOMER CONTEXT</p>", unsafe_allow_html=True)
        selected_cust = st.selectbox(
            "Customer Context",
            ["CUST-001 (Jane Doe)", "CUST-002 (John Smith)"],
            index=0 if st.session_state.get("current_customer_id") == "CUST-001" else 1,
            label_visibility="collapsed"
        )
        st.session_state["current_customer_id"] = "CUST-001" if "CUST-001" in selected_cust else "CUST-002"

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.8rem; font-weight: 600; color: #475569; margin-bottom: 4px;'>OPERATIONS WORKSPACE</p>", unsafe_allow_html=True)

        pages = [
            "Dashboard Overview",
            "AI Banking Assistant",
            "Customer Accounts",
            "Transaction History",
            "Fraud Detection",
            "Fraud Investigation Center",
            "Credit and Loan Assessment",
            "ML Model Performance",
            "Agent Execution Logs",
            "About and Settings"
        ]

        page = st.radio("Navigation", pages, label_visibility="collapsed")

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.divider()

        col_lo1, col_lo2 = st.columns([1, 1])
        with col_lo1:
            api_status = "Online (LLM)" if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY", "").startswith("your_") else "Offline (Rule-Agent)"
            st.caption(f"Router: **{api_status}**")
        with col_lo2:
            if st.button("Sign Out", use_container_width=True):
                logout()

    return page


# ==============================================================================
# PAGE 2: DASHBOARD OVERVIEW
# ==============================================================================
def render_dashboard_overview():
    st.markdown("<div class='ib-page-title'>Operations Dashboard Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='ib-page-subtitle'>Real-time summary of portfolio accounts, simulated liquidity, transaction streams, and risk surveillance.</div>", unsafe_allow_html=True)

    stats = DashboardService.get_summary_stats()
    fraud_records = DashboardService.get_fraud_analysis()
    suspicious_count = len([f for f in fraud_records if f.get("is_suspicious")])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='ib-metric-card'>
            <div class='ib-metric-label'>Total Customers</div>
            <div class='ib-metric-value'>{stats['total_customers']}</div>
            <div class='ib-metric-sub'>Active client profiles: {stats['active_customers']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='ib-metric-card ib-metric-card-success'>
            <div class='ib-metric-label'>Portfolio Balance</div>
            <div class='ib-metric-value'>${stats['total_balance']:,.2f}</div>
            <div class='ib-metric-sub'>Combined simulated deposits</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='ib-metric-card'>
            <div class='ib-metric-label'>Recorded Transactions</div>
            <div class='ib-metric-value'>{stats['total_transactions']}</div>
            <div class='ib-metric-sub'>Audit trail active</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='ib-metric-card ib-metric-card-danger'>
            <div class='ib-metric-label'>Fraud Alerts</div>
            <div class='ib-metric-value'>{suspicious_count} Flagged</div>
            <div class='ib-metric-sub'>Pending analyst review</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Secondary Dashboard Panels
    col_chart, col_quick = st.columns([2, 1])

    with col_chart:
        st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;'>Recent System Activity</h3>", unsafe_allow_html=True)
        recent_txns = DashboardService.get_recent_transactions(5)
        if recent_txns:
            tx_df = pd.DataFrame(recent_txns)[["transaction_id", "date", "description", "amount", "type"]]
            tx_df.columns = ["Transaction ID", "Date", "Description", "Amount ($)", "Type"]
            st.dataframe(tx_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_quick:
        st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; margin-bottom: 0.75rem;'>Surveillance Summary</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 0.85rem; margin-bottom: 0.75rem;'>
            <div style='font-weight: 600; color: #b91c1c; font-size: 0.85rem;'>🚨 High Risk Item Detected</div>
            <div style='font-size: 0.8rem; color: #7f1d1d; margin-top: 2px;'>
                Transaction <code>TXN-9004</code> (-$2,500.00) flagged: Unknown International Merchant.
            </div>
        </div>
        <div style='background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 0.85rem;'>
            <div style='font-weight: 600; color: #1d4ed8; font-size: 0.85rem;'>🤖 Multi-Agent Engine Ready</div>
            <div style='font-size: 0.8rem; color: #1e3a8a; margin-top: 2px;'>
                Customer, Risk, and Credit agents synchronized with core tools.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: AI BANKING ASSISTANT
# ==============================================================================
def render_ai_assistant():
    st.markdown("<div class='ib-page-title'>AI Banking Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='ib-page-subtitle'>Autonomous multi-agent orchestration for customer queries, transaction audits, fraud checks, and lending qualification.</div>", unsafe_allow_html=True)

    col_chat, col_context = st.columns([2.2, 1])

    with col_context:
        st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 1rem; font-weight: 700; margin-bottom: 0.5rem;'>💡 Suggested Banking Queries</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.8rem; color: #64748b; margin-bottom: 0.75rem;'>Click any button below to execute the end-to-end multi-agent workflow:</p>", unsafe_allow_html=True)

        suggestions = [
            ("💰 1. Check CUST-001 Balance", "What is the account balance for CUST-001?"),
            ("📜 2. Show Last 5 Transactions", "Show my last 5 transactions."),
            ("🚨 3. Check TXN-9004 for Fraud", "Check transaction TXN-9004 for suspicious activity."),
            ("🏦 4. Check Loan Eligibility", "Check whether CUST-002 is eligible for a $5000 loan."),
            ("⚡ 5. Combined Multi-Agent Audit", "Check CUST-001's balance, show recent transactions, and check for suspicious activity.")
        ]

        for label, q_text in suggestions:
            if st.button(label, key=f"sug_{label}", use_container_width=True):
                st.session_state.messages.append(HumanMessage(content=q_text))
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Context Card
        cust_id = st.session_state.get("current_customer_id", "CUST-001")
        cust = CustomerService.get_customer(cust_id)
        if cust:
            st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 1rem; font-weight: 700; margin-bottom: 0.5rem;'>👤 Active Profile: {cust['name']}</h4>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='font-size: 0.85rem; color: #334155;'>
                <strong>Customer ID:</strong> <code>{cust_id}</code><br>
                <strong>Credit Score:</strong> <span class='ib-badge ib-badge-blue'>{cust['credit_score']}</span><br>
                <strong>Monthly Income:</strong> ${cust['monthly_income']:,.2f}<br>
                <strong>Current EMI:</strong> ${cust['existing_loan_emi']:,.2f}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_chat:
        st.markdown("<div class='ib-card' style='min-height: 500px;'>", unsafe_allow_html=True)

        # Render conversation history
        messages: list[BaseMessage] = st.session_state.get("messages", [])

        if not messages:
            st.info("👋 Hello! I am the IntelliBank AI Assistant. Ask a question or click any suggested prompt on the right to start.")

        for idx, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(msg.content)
                    steps = getattr(msg, "intermediate_steps", [])
                    if steps:
                        with st.expander(f"🔍 Agent Execution Trace ({len(steps)} Tool Call{'s' if len(steps)>1 else ''})"):
                            for s in steps:
                                tool_name = s.get("tool", "tool")
                                args = s.get("args", {})
                                result = s.get("result", "")
                                badge_class = "ib-badge-blue"
                                if "risk" in tool_name or "fraud" in tool_name:
                                    badge_class = "ib-badge-red"
                                elif "loan" in tool_name:
                                    badge_class = "ib-badge-purple"
                                st.markdown(f"<span class='ib-badge {badge_class}'>Tool Call: <code>{tool_name}</code></span>", unsafe_allow_html=True)
                                st.code(f"Arguments: {json.dumps(args, indent=2)}\nResponse: {result}", language="json")

        st.markdown("</div>", unsafe_allow_html=True)

        user_input = st.chat_input("Enter your banking question (e.g., 'What is my balance for CUST-001?')...")
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            st.rerun()

    # Process pending human message
    if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
        user_msg = str(st.session_state.messages[-1].content)
        with st.spinner("Multi-Agent Supervisor is routing and executing banking tools..."):
            orch = st.session_state.get("orchestrator")
            if not orch:
                orch = create_orchestrator()
                st.session_state["orchestrator"] = orch

            chat_history: list[BaseMessage] = st.session_state.messages[:-1]
            response = run_orchestrator(orch, user_msg, chat_history)

            ai_msg = AIMessage(content=response.get("output", "No response generated"))
            ai_msg.intermediate_steps = response.get("intermediate_steps", [])  # type: ignore[attr-defined]
            st.session_state.messages.append(ai_msg)
            st.rerun()


# ==============================================================================
# PAGE 4: CUSTOMER ACCOUNTS
# ==============================================================================
def render_customer_accounts():
    st.markdown("<div class='ib-page-title'>Customer Accounts Directory</div>", unsafe_allow_html=True)
    st.markdown("<div class='ib-page-subtitle'>Centralized client ledger displaying simulated accounts, liquidity positions, credit tiers, and debt commitments.</div>", unsafe_allow_html=True)

    customers = CustomerService.get_all_customers()

    for cust in customers:
        st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
        col_hdr1, col_hdr2 = st.columns([2, 1])
        with col_hdr1:
            st.markdown(f"<h3 style='margin: 0; font-size: 1.2rem; color: #0f172a;'>{cust['name']} <span class='ib-badge ib-badge-blue'>{cust['customer_id']}</span></h3>", unsafe_allow_html=True)
            st.markdown(f"<p class='text-muted' style='margin-top: 4px;'>Contact: {cust['email']}</p>", unsafe_allow_html=True)
        with col_hdr2:
            tier_badge = "ib-badge-green" if cust['credit_score'] >= 700 else "ib-badge-amber"
            st.markdown(f"""
            <div style='text-align: right;'>
                <span class='ib-badge {tier_badge}'>Credit Score: {cust['credit_score']}</span>
                <div style='font-size: 0.8rem; color: #64748b; margin-top: 4px;'>Monthly Income: ${cust['monthly_income']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        accs = cust.get("accounts", [])
        if accs:
            acc_df = pd.DataFrame(accs)
            acc_df.columns = ["Account ID", "Account Type", "Current Balance ($)"]
            acc_df["Current Balance ($)"] = acc_df["Current Balance ($)"].apply(lambda x: f"${x:,.2f}")
            st.dataframe(acc_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 5: TRANSACTION HISTORY
# ==============================================================================
def render_transaction_history():
    st.markdown("<div class='ib-page-title'>Transaction History Ledger</div>", unsafe_allow_html=True)
    st.markdown("<div class='ib-page-subtitle'>Auditable ledger of simulated banking transactions with filtering across accounts and classifications.</div>", unsafe_allow_html=True)

    transactions = DashboardService.get_recent_transactions(50)
    if not transactions:
        st.info("No transaction records found.")
        return

    df = pd.DataFrame(transactions)

    # Filter Controls
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        account_filter = st.selectbox("Filter by Account", ["All Accounts"] + sorted(list(df["account_id"].unique())))
    with col_f2:
        type_filter = st.selectbox("Transaction Type", ["All Types"] + sorted(list(df["type"].unique())))
    with col_f3:
        search_kw = st.text_input("Search Description", placeholder="e.g., Salary, Coffee, International...")

    filtered_df = df.copy()
    if account_filter != "All Accounts":
        filtered_df = filtered_df[filtered_df["account_id"] == account_filter]
    if type_filter != "All Types":
        filtered_df = filtered_df[filtered_df["type"] == type_filter]
    if search_kw:
        filtered_df = filtered_df[filtered_df["description"].str.contains(search_kw, case=False, na=False)]

    st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
    st.markdown(f"<p class='text-muted'>Displaying {len(filtered_df)} matching transactions.</p>", unsafe_allow_html=True)

    display_df = filtered_df[["transaction_id", "account_id", "date", "description", "amount", "type"]].copy()
    display_df.columns = ["Txn ID", "Account", "Date / Time", "Description", "Amount ($)", "Type"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 6: FRAUD DETECTION (SIMULATED BANKING)
# ==============================================================================
def render_fraud_detection():
    st.markdown("<div class='ib-page-title'>Fraud Detection & Surveillance</div>", unsafe_allow_html=True)
    st.markdown("<div class='ib-page-subtitle'>Rule-based behavioral anomaly detection for active simulated banking accounts.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;'>Simulated Transaction Surveillance</h3>", unsafe_allow_html=True)
        fraud_data = DashboardService.get_fraud_analysis()
        df = pd.DataFrame(fraud_data)

        if not df.empty:
            summary_cols = ["transaction_id", "description", "amount", "risk_score", "is_suspicious", "reason"]
            df_disp = df[summary_cols].copy()
            df_disp["Status"] = df_disp["is_suspicious"].apply(lambda x: "🚨 Suspicious" if x else "✅ Normal")
            df_disp = df_disp.drop(columns=["is_suspicious"])
            df_disp.columns = ["Txn ID", "Description", "Amount ($)", "Risk Score", "Reason", "Status"]
            st.dataframe(df_disp, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;'>Live Risk Analyzer</h3>", unsafe_allow_html=True)
        st.markdown("<p class='text-muted'>Test any mock transaction against banking heuristic risk rules:</p>", unsafe_allow_html=True)

        with st.form("risk_test_form"):
            test_amt = st.number_input("Transaction Amount ($)", value=-2500.0, step=50.0, help="Negative for debits/withdrawals")
            test_desc = st.text_input("Merchant Description", value="Unknown International Merchant")
            run_btn = st.form_submit_button("Evaluate Risk", use_container_width=True)

            if run_btn:
                res = FraudService.check_rule_based(test_amt, test_desc)
                if res.success and res.data:
                    d = res.data
                    badge_style = "ib-badge-red" if d['is_suspicious'] else "ib-badge-green"
                    status_text = "🚨 HIGH RISK / FLAGGED" if d['is_suspicious'] else "✅ NORMAL TRANSACTION"
                    st.markdown(f"""
                    <div style='background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin-top: 1rem;'>
                        <div style='font-weight: 700; margin-bottom: 6px;'><span class='ib-badge {badge_style}'>{status_text}</span></div>
                        <div style='font-size: 0.85rem; color: #334155;'>
                            <strong>Risk Score:</strong> <code>{d['risk_score']}</code> (Threshold: {d['threshold_used']})<br>
                            <strong>Triggered Flags:</strong> {d['reason']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 7: FRAUD INVESTIGATION CENTER (REAL KAGGLE ML)
# ==============================================================================
def render_fraud_investigation():
    st.markdown("<div class='ib-page-title'>Fraud Investigation Center (Kaggle ML)</div>", unsafe_allow_html=True)
    st.markdown("<div class='ib-page-subtitle'>Interactive analyst workspace powered by trained Machine Learning models on the anonymized Kaggle Credit Card dataset.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 0.85rem; margin-bottom: 1.25rem;'>
        <p style='margin: 0; font-size: 0.85rem; color: #1e3a8a;'>
            <strong>ℹ️ Dataset Separation Notice:</strong> This investigation console evaluates genuine PCA anonymized features
            (<code>V1–V28</code>, <code>Time</code>, <code>Amount</code>) from the Kaggle Credit Card Fraud dataset. It operates completely
            independently from the mock retail banking accounts in the preceding tabs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    models_dir = PathsConfig.MODELS_DIR
    model_files = {
        "Random Forest": models_dir / "random_forest.pkl",
        "Logistic Regression": models_dir / "logistic_regression.pkl",
        "XGBoost": models_dir / "xgboost.pkl",
    }
    available_models = {name: path for name, path in model_files.items() if path.exists()}
    scaler_path = models_dir / "scaler.pkl"

    if not available_models or not scaler_path.exists():
        st.warning("⚠️ No trained ML models found. Please run ML training (`python -m ml.train_model`) first.")
        return

    import joblib
    from ml.data_loader import inspect_data, load_credit_card_data

    try:
        df = load_credit_card_data()
        stats = inspect_data(df)
    except Exception as e:
        st.error(f"Unable to load creditcard dataset: {e}")
        return

    # Model and threshold selector
    col_ctl1, col_ctl2, col_ctl3 = st.columns([1.5, 1.5, 1])
    with col_ctl1:
        sel_model = st.selectbox("Inference Model", list(available_models.keys()), index=0)
    with col_ctl2:
        threshold = st.slider("Decision Threshold", min_value=0.05, max_value=0.95, value=0.40, step=0.05)
    with col_ctl3:
        sample_size = st.selectbox("Sample Inspection Size", [25, 50, 100, 200], index=1)

    model = joblib.load(available_models[sel_model])
    scaler = joblib.load(scaler_path)

    # Filter by true class
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        class_filter = st.radio("Filter Samples by True Class", ["All Records", "Confirmed Frauds (Class=1)", "Normal Transactions (Class=0)"], horizontal=True)

    subset_df = df.copy()
    if "Confirmed Frauds" in class_filter:
        subset_df = subset_df[subset_df["Class"] == 1]
    elif "Normal" in class_filter:
        subset_df = subset_df[subset_df["Class"] == 0]

    sample_df = subset_df.head(sample_size).copy()

    # Preprocessing
    feature_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    X_sample = sample_df[feature_cols].copy()
    X_sample[['Time', 'Amount']] = scaler.transform(X_sample[['Time', 'Amount']])

    probs = model.predict_proba(X_sample)[:, 1]
    preds = (probs >= threshold).astype(int)

    sample_df["Predicted Risk"] = probs
    sample_df["Prediction"] = ["🚨 Fraud" if p == 1 else "✅ Normal" for p in preds]
    sample_df["True Class"] = ["Fraud" if c == 1 else "Normal" for c in sample_df["Class"]]

    st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;'>Dataset Inference Results ({len(sample_df)} records)</h3>", unsafe_allow_html=True)

    disp_cols = ["Time", "Amount", "Predicted Risk", "Prediction", "True Class", "V1", "V2", "V3", "V4"]
    st.dataframe(sample_df[disp_cols].style.format({"Amount": "${:,.2f}", "Predicted Risk": "{:.4f}"}), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Analyst Review Flow
    st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;'>Analyst Review & Case Disposition</h3>", unsafe_allow_html=True)

    col_rev1, col_rev2 = st.columns([1, 2])
    with col_rev1:
        rec_index = st.selectbox("Select Record Index for Disposition", sample_df.index.tolist())
    with col_rev2:
        disp_choice = st.selectbox("Disposition Verdict", ["Pending Review", "Confirmed Fraud", "False Positive (Whitelisted)", "Escalate to Tier 2"])

    analyst_notes = st.text_area("Analyst Investigation Notes", placeholder="Document rationale for disposition...")
    if st.button("Save Analyst Disposition"):
        st.session_state["dispositions"][rec_index] = {
            "verdict": disp_choice,
            "notes": analyst_notes,
            "timestamp": datetime.now().isoformat()
        }
        st.success(f"Disposition for record {rec_index} saved successfully as '{disp_choice}'.")

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 8: CREDIT AND LOAN ASSESSMENT
# ==============================================================================
def render_credit_assessment():
    st.markdown("<div class='ib-page-title'>Credit and Loan Assessment</div>", unsafe_allow_html=True)
    st.markdown("<div class='ib-page-subtitle'>Rule-based consumer lending evaluation engine computing EMI, debt-to-income limits, and underwriting credit tiers.</div>", unsafe_allow_html=True)

    col_app, col_rules = st.columns([1.5, 1])

    with col_app:
        st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;'>Loan Application Underwriting</h3>", unsafe_allow_html=True)

        cust_id = st.selectbox("Applicant Profile", ["CUST-001 (Jane Doe)", "CUST-002 (John Smith)"])
        actual_cust_id = "CUST-001" if "CUST-001" in cust_id else "CUST-002"

        loan_amount = st.number_input("Requested Facility Amount ($)", min_value=500.0, max_value=50000.0, value=5000.0, step=500.0)

        if st.button("Evaluate Loan Qualification", use_container_width=True):
            res = CreditService.check_eligibility(actual_cust_id, loan_amount)
            if res.success and res.data:
                d = res.data
                if d.get("eligible"):
                    tier = str(d.get("credit_tier")).upper()
                    st.markdown(f"""
                    <div style='background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 1.25rem; margin-top: 1rem;'>
                        <div style='font-weight: 700; font-size: 1.1rem; color: #047857; margin-bottom: 6px;'>
                            ✅ Application Approved
                        </div>
                        <div style='font-size: 0.9rem; color: #065f46;'>
                            <strong>Approved Principal:</strong> ${d.get('max_approved_amount', loan_amount):,.2f}<br>
                            <strong>Estimated Monthly EMI:</strong> ${d.get('estimated_emi', 0):,.2f}/month<br>
                            <strong>Credit Underwriting Tier:</strong> <span class='ib-badge ib-badge-green'>{tier}</span><br>
                            <strong>Post-Facility DTI Ratio:</strong> {float(d.get('dti_ratio', 0))*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 1.25rem; margin-top: 1rem;'>
                        <div style='font-weight: 700; font-size: 1.1rem; color: #b91c1c; margin-bottom: 6px;'>
                            ❌ Application Declined
                        </div>
                        <div style='font-size: 0.9rem; color: #7f1d1d;'>
                            <strong>Primary Reason:</strong> {d.get('reason', 'Policy constraint')}<br>
                            <strong>Max Affordable Monthly EMI:</strong> ${d.get('max_affordable_emi', 0):,.2f}/month
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(f"Underwriting Error: {res.error}")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_rules:
        st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;'>Underwriting Policy Rules</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size: 0.85rem; color: #334155; line-height: 1.6;'>
            <strong>Prime Credit Score Threshold:</strong> &ge; {LoanConfig.MIN_CREDIT_SCORE_PRIME}<br>
            <strong>Subprime Credit Score Threshold:</strong> &ge; {LoanConfig.MIN_CREDIT_SCORE_SUBPRIME}<br>
            <strong>Maximum Debt-to-Income (DTI):</strong> {LoanConfig.MAX_DTI_RATIO*100:.0f}% of Gross Income<br>
            <strong>Annual Percentage Rate (APR):</strong> {LoanConfig.INTEREST_RATE_ANNUAL*100:.1f}% Fixed<br>
            <strong>Standard Term Length:</strong> {LoanConfig.LOAN_TERM_MONTHS} Months (5 Years)
        </div>
        <hr style='border: none; border-top: 1px solid #e2e8f0; margin: 12px 0;'>
        <div style='font-size: 0.75rem; color: #64748b;'>
            <em>Regulatory Note: Automated decisioning demonstrated for interview MVP. Real-world lending requires Fair Lending compliance and FCRA credit verification.</em>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 9: ML MODEL PERFORMANCE
# ==============================================================================
def render_ml_performance():
    st.markdown("<div class='ib-page-title'>Machine Learning Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='ib-page-subtitle'>Rigorous evaluation metrics comparing Logistic Regression, Random Forest, and XGBoost on credit card fraud detection.</div>", unsafe_allow_html=True)

    report_file = PathsConfig.REPORTS_DIR / "model_comparison.csv"

    if not report_file.exists():
        st.warning("""
        ⚠️ **No evaluation report found.**
        Click the button below to execute leakage-safe cross-model evaluation against the dataset test split.
        """)
        if st.button("Run Model Evaluation"):
            from ml.evaluate_models import evaluate_all_models
            with st.spinner("Evaluating models..."):
                evaluate_all_models()
                st.rerun()
        return

    df_models = pd.read_csv(report_file)

    st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; margin-bottom: 0.75rem;'>Comparative Evaluation Matrix</h3>", unsafe_allow_html=True)
    st.dataframe(df_models, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Detailed Confusion Matrix Cards
    st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; margin: 1rem 0 0.5rem;'>Model Confusion Breakdown</h3>", unsafe_allow_html=True)
    cols = st.columns(len(df_models))

    for idx, (_, row) in enumerate(df_models.iterrows()):
        with cols[idx]:
            m_name = row["Model"]
            st.markdown(f"""
            <div class='ib-card-compact'>
                <div style='font-weight: 700; color: #0f172a; margin-bottom: 8px;'>{m_name}</div>
                <div style='font-size: 0.825rem; color: #334155;'>
                    <strong>Precision:</strong> {row['Precision']:.4f}<br>
                    <strong>Recall:</strong> {row['Recall']:.4f}<br>
                    <strong>F1-Score:</strong> {row['F1-Score']:.4f}<br>
                    <strong>PR-AUC:</strong> {row['PR-AUC']:.4f}<br>
                    <hr style='margin: 6px 0; border: none; border-top: 1px solid #e2e8f0;'>
                    <strong>True Positives (TP):</strong> {int(row['True Positives'])}<br>
                    <strong>False Positives (FP):</strong> {int(row['False Positives'])}<br>
                    <strong>False Negatives (FN):</strong> {int(row['False Negatives'])}<br>
                    <strong>True Negatives (TN):</strong> {int(row['True Negatives'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("Optimize Operating Threshold", use_container_width=True):
            from ml.evaluate_models import find_optimal_threshold
            opt = find_optimal_threshold()
            st.success(f"Optimal Threshold: {opt.get('optimal_threshold')} (Best F1: {opt.get('best_f1')})")
    with c_btn2:
        if st.button("Re-evaluate All Models", use_container_width=True):
            from ml.evaluate_models import evaluate_all_models
            with st.spinner("Re-evaluating models..."):
                evaluate_all_models()
                st.rerun()


# ==============================================================================
# PAGE 10: AGENT EXECUTION LOGS
# ==============================================================================
def render_agent_logs():
    st.markdown("<div class='ib-page-title'>Agent Execution Audit Logs</div>", unsafe_allow_html=True)
    st.markdown("<div class='ib-page-subtitle'>Complete chronological audit trail of all specialized tools invoked by the Multi-Agent Orchestrator.</div>", unsafe_allow_html=True)

    messages = st.session_state.get("messages", [])
    logs = []

    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, "intermediate_steps") and msg.intermediate_steps:
            user_query = messages[i-1].content if i > 0 and isinstance(messages[i-1], HumanMessage) else "Direct execution"
            logs.append({"query": user_query, "steps": msg.intermediate_steps})

    if not logs:
        st.info("ℹ️ No agent tool calls executed yet in this session. Head to the **AI Banking Assistant** to run queries.")
        return

    for idx, item in enumerate(reversed(logs), 1):
        st.markdown(f"""
        <div class='ib-card'>
            <div style='font-size: 0.8rem; color: #64748b; font-weight: 600;'>LOG ENTRY #{len(logs)-idx+1}</div>
            <div style='font-size: 1rem; font-weight: 700; color: #0f172a; margin: 4px 0 10px;'>Query: <code>{item['query']}</code></div>
        """, unsafe_allow_html=True)

        for s in item["steps"]:
            tool_name = s.get("tool", "unknown")
            args = s.get("args", {})
            result = s.get("result", "")
            st.code(f"Tool Executed: {tool_name}\nArguments: {json.dumps(args, indent=2)}\nOutput: {result}", language="json")

        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 11: ABOUT AND SETTINGS
# ==============================================================================
def render_about_settings():
    st.markdown("<div class='ib-page-title'>About and System Settings</div>", unsafe_allow_html=True)
    st.markdown("<div class='ib-page-subtitle'>Platform architecture, environment configuration, credentials status, and operational controls.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;'>System Specifications</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size: 0.85rem; color: #334155; line-height: 1.8;'>
            <strong>Application:</strong> {AppConfig.NAME} v{AppConfig.VERSION}<br>
            <strong>Assignment:</strong> Wrench Wise EduTech Technical Project<br>
            <strong>Multi-Agent Framework:</strong> LangGraph & ReAct Supervisor Router<br>
            <strong>ML Algorithms:</strong> Logistic Regression, Random Forest, XGBoost<br>
            <strong>Dataset:</strong> Kaggle Credit Card Fraud Detection (284,807 transactions)<br>
            <strong>UI Framework:</strong> Streamlit (Accessible Fintech Design System)
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;'>Environment Status</h3>", unsafe_allow_html=True)
        api_key_set = bool(os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY", "").startswith("your_"))
        models_ready = (PathsConfig.MODELS_DIR / "random_forest.pkl").exists()

        st.markdown(f"""
        <div style='font-size: 0.85rem; color: #334155; line-height: 1.8;'>
            <strong>OpenAI API Key:</strong> {'<span class="ib-badge ib-badge-green">Configured</span>' if api_key_set else '<span class="ib-badge ib-badge-amber">Offline (Deterministic Supervisor Active)</span>'}<br>
            <strong>ML Models Artifacts:</strong> {'<span class="ib-badge ib-badge-green">Trained & Loaded</span>' if models_ready else '<span class="ib-badge ib-badge-red">Missing</span>'}<br>
            <strong>Active Session User:</strong> <code>{st.session_state.get('username', 'N/A')}</code><br>
            <strong>Fraud Heuristic Threshold:</strong> <code>{config.get('ml.fraud_threshold', 0.4)}</code>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # API Key Update
    st.markdown("<div class='ib-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;'>LLM Key Management</h3>", unsafe_allow_html=True)
    new_key = st.text_input("Update OpenAI API Key", type="password", placeholder="sk-...")
    if st.button("Apply API Key"):
        if new_key.strip():
            os.environ["OPENAI_API_KEY"] = new_key.strip()
            try:
                st.session_state["orchestrator"] = create_orchestrator()
                st.success("API Key updated and LangGraph orchestrator re-initialized successfully!")
            except Exception as e:
                st.error(f"Failed to re-initialize orchestrator: {e}")
        else:
            st.warning("Please provide a valid key string.")
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# MAIN ROUTER
# ==============================================================================
def main():
    init_session_state()

    if not st.session_state["logged_in"]:
        render_login_page()
        return

    if not validate_session():
        st.error("Your session has expired. Please log in again.")
        render_login_page()
        return

    active_page = render_sidebar()

    page_map = {
        "Dashboard Overview": render_dashboard_overview,
        "AI Banking Assistant": render_ai_assistant,
        "Customer Accounts": render_customer_accounts,
        "Transaction History": render_transaction_history,
        "Fraud Detection": render_fraud_detection,
        "Fraud Investigation Center": render_fraud_investigation,
        "Credit and Loan Assessment": render_credit_assessment,
        "ML Model Performance": render_ml_performance,
        "Agent Execution Logs": render_agent_logs,
        "About and Settings": render_about_settings
    }

    handler = page_map.get(active_page)
    if handler:
        handler()
    else:
        render_dashboard_overview()


if __name__ == "__main__":
    main()
