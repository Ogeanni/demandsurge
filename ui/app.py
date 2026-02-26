"""
ui/app.py
DemandSurge Streamlit Chat Interface — FastAPI-backed version.

All ML and agent calls go through the FastAPI backend at API_BASE_URL.
Streamlit is purely a presentation layer 

Usage:
    # Terminal 1 — start API
    uvicorn api.main:app --reload --port 8000

    # Terminal 2 — start UI
    streamlit run ui/app.py
"""
import os
import time
import logging

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Page config — must be first Streamlit call ────────────────────────
st.set_page_config(
    page_title="DemandSurgeß — AI Pricing Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# API CLIENT HELPERS
# ══════════════════════════════════════════════════════════════════════

def api_get(path: str, params: dict = None) -> dict | list | None:
    """GET request to FastAPI. Returns parsed JSON or None on error."""
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            " Cannot reach the DemandSurge API. "
            "Make sure it's running: `uvicorn app.main:app --reload --port 8000`"
        )
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text[:200]}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None
    

def api_post(path: str, payload: dict) -> dict | str | None:
    """POST request to FastAPI. Returns parsed JSON or None on error."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}{path}",
            json=payload,
            timeout=120,   # Agent can take a while
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return resp.text
    except requests.exceptions.ConnectionError:
        st.error(
            " Cannot reach the DemandSurge API. "
            "Make sure it's running: `uvicorn app.main:app --reload --port 8000`"
        )
        return None
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", e.response.text[:200])
        except Exception:
            detail = e.response.text[:200]
        st.error(f"API error {e.response.status_code}: {detail}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None
    

# ══════════════════════════════════════════════════════════════════════
# CUSTOM CSS  (identical palette to previous version)
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --navy:   #0D1B2A;
    --teal:   #1B7F79;
    --gold:   #E8A020;
    --slate:  #2C3E50;
    --muted:  #8899AA;
    --bg:     #F7F9FC;
    --card:   #FFFFFF;
    --border: #E2E8F0;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--navy); }
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: var(--bg); }

section[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: #E8EDF2 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMarkdown p {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

div[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 1px 3px rgba(13,27,42,0.06);
}
div[data-testid="metric-container"] label {
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: var(--navy) !important;
}

.stButton > button {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 20px;
    color: var(--slate) !important;
    font-size: 0.8rem;
    padding: 4px 14px;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--teal);
    color: var(--teal) !important;
    background: rgba(27,127,121,0.05);
}

.section-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 8px;
    font-weight: 500;
}
.brand-title { font-size: 1.5rem; font-weight: 600; color: white; letter-spacing: -0.02em; }
.brand-sub   { font-size: 0.75rem; color: rgba(255,255,255,0.45); margin-top: 2px; }
.status-dot  {
    width: 7px; height: 7px; border-radius: 50%;
    background: #48BB78; display: inline-block;
    margin-right: 6px; box-shadow: 0 0 6px rgba(72,187,120,0.6);
}
.sidebar-divider { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 16px 0; }

div[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 2px rgba(27,127,121,0.15) !important;
}
/* ── Chat message bubbles ─────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 1px 3px rgba(13,27,42,0.06) !important;
}

/* Assistant bubble — slightly tinted */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #EEF4FF !important;
    border-color: #C7D9F5 !important;
}

/* Response text inside assistant bubble */
[data-testid="stChatMessage"] p {
    color: var(--navy) !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "messages":        [],
        "selected_product": None,
        "pending_query":   None,
        "api_healthy":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ══════════════════════════════════════════════════════════════════════
# CACHED API CALLS
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30, show_spinner=False)
def check_health():
    return api_get("/health")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_products():
    return api_get("/products") or []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_recommendation(product_id: int):
    return api_get(f"/products/{product_id}/recommend")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_forecast(category: str):
    return api_get(f"/forecast/{category}")


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:

        st.markdown("""
        <div style="padding: 20px 0 8px 0;">
            <div class="brand-title">📈 DemandSurge</div>
            <div class="brand-sub">AI Pricing Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Health check ──────────────────────────────────────────────
        health = check_health()
        if health and health.get("status") == "ok":
            agent_ready = health.get("agent_ready", False)
            if agent_ready:
                st.markdown(
                    '<span class="status-dot"></span> API & Agent online',
                    unsafe_allow_html=True
                )
            else:
                st.markdown("🟡 API online — Agent loading...")
                err = health.get("agent_error")
                if err:
                    st.caption(f"Agent error: {err}")
        else:
            st.markdown("🔴 API offline")
            st.caption(f"Start with: `uvicorn app.main:app --port 8000`")

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ── Product selector ──────────────────────────────────────────
        st.markdown('<div class="section-label">Select Product</div>', unsafe_allow_html=True)

        products = fetch_products()
        if not products:
            st.caption("No products — check API connection.")
            return

        product_labels = [f"{p['name']} ({p['category']})" for p in products]
        selected_idx   = st.selectbox(
            "Product",
            range(len(products)),
            format_func=lambda i: product_labels[i],
            label_visibility="collapsed",
        )
        selected = products[selected_idx]
        st.session_state.selected_product = selected

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ── Live metrics ──────────────────────────────────────────────
        st.markdown('<div class="section-label">Live Metrics</div>', unsafe_allow_html=True)

        with st.spinner(""):
            rec = fetch_recommendation(selected["id"])
            fc  = fetch_forecast(selected["category"])

        if rec:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Price", f"${rec['current_price']:.2f}")
            with col2:
                st.metric(
                    "Recommended",
                    f"${rec['recommended_price']:.2f}",
                    delta=f"{rec['pct_vs_current']:+.1f}%",
                )

            col3, col4 = st.columns(2)
            with col3:
                if fc:
                    direction = fc.get("trend_direction", "stable")
                    icon = "↑" if direction == "rising" else ("↓" if direction == "falling" else "→")
                    st.metric(
                        "Demand Trend",
                        f"{fc['current_index']:.0f}/100",
                        delta=f"{icon} {direction}",
                    )
                else:
                    st.metric("Demand Trend", "N/A")
            with col4:
                comp_med = rec.get("comp_price_med", 0)
                st.metric("Comp. Median", f"${comp_med:.2f}" if comp_med else "N/A")
        else:
            st.caption("Metrics unavailable.")

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ── Quick actions ─────────────────────────────────────────────
        st.markdown('<div class="section-label">Quick Actions</div>', unsafe_allow_html=True)

        pname = selected["name"]
        pcat  = selected["category"]

        if st.button("📊 Price this product",  use_container_width=True):
            st.session_state.pending_query = f"What price should I set for the {pname}?"
        if st.button("📈 Demand outlook",       use_container_width=True):
            st.session_state.pending_query = f"What is the demand forecast for {pcat} over the next 30 days?"
        if st.button("🔍 Competitor check",     use_container_width=True):
            st.session_state.pending_query = f"How do my {pname} prices compare to competitors?"
        if st.button("📋 Weekly review",        use_container_width=True):
            st.session_state.pending_query = "Give me a full weekly pricing review for all products."

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.cache_data.clear()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════

def render_main():

    st.markdown("## Pricing Intelligence")
    st.caption("Ask me anything about your product pricing, demand trends, or competitor positioning.")
    st.divider()

    # ── Suggested questions (shown only on empty chat) ────────────────
    if not st.session_state.messages:
        st.markdown('<div class="section-label">Try asking</div>', unsafe_allow_html=True)

        suggestions = [
            ("💰", "Leather wallet price?",      "What price should I set for the leather wallet?"),
            ("📈", "Electronics demand outlook",  "What is the demand outlook for electronics over the next 30 days?"),
            ("🔍", "Yoga mat vs competitors",     "How do my yoga mat prices compare to competitors?"),
            ("📦", "Clear running shoe inventory","I have 200 units of running shoes. What price clears them in 3 weeks?"),
            ("📋", "Weekly pricing review",       "Give me a full weekly pricing review — what should I change this week?"),
        ]

        cols = st.columns(len(suggestions))
        for col, (icon, label, query) in zip(cols, suggestions):
            with col:
                if st.button(f"{icon} {label}", key=f"sug_{label}"):
                    st.session_state.pending_query = query

        st.divider()

    # ── Chat history ──────────────────────────────────────────────────
    for msg in st.session_state.messages:
        avatar = "💬" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── Handle pending query from sidebar / suggestion buttons ────────
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)
        _process_query(query)
        st.rerun()

    # ── Chat input ────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about pricing, demand, competitors..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        _process_query(prompt)


def _process_query(query: str):
    """POSTs the query to FastAPI /chat and renders the response."""
    with st.chat_message("assistant", avatar="💬"):
        with st.spinner("Thinking..."):
            result = api_post("/chat", {"message": query})

        if result is None:
            return

        # Handle both plain text (str) and JSON (dict) responses
        if isinstance(result, str):
            response = result
        elif isinstance(result, dict):
            response = result.get("response", "No response.")
        else:
            response = str(result)

        if not response:
            st.warning("The agent returned an empty response. Please try again.")
            return

        st.markdown(response)

        st.session_state.messages.append({
            "role":    "assistant",
            "content": response,
        })

    # Render response OUTSIDE the chat bubble in a styled container
    st.markdown(
        f"""
        <div style="
            background-color: #1e1e2e;
            color: #e0e0e0;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #4a9eff;
            font-size: 15px;
            line-height: 1.7;
            margin: 10px 0;
            white-space: pre-wrap;
        ">{response}</div>
        """,
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════
# HOW IT WORKS EXPANDER
# ══════════════════════════════════════════════════════════════════════

def render_explainer():
    with st.expander("⚙️ How DemandSurge works", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("** Data Sources**")
            st.markdown("Google Trends · eBay API · Etsy API · Shopify sandbox")
        with col2:
            st.markdown("** ML Models**")
            st.markdown("Prophet demand forecast · XGBoost pricing · 28 engineered features")
        with col3:
            st.markdown("**🧠 Agent**")
            st.markdown("LangChain ReAct · GPT-4o-mini · 4 tools · 5-turn memory")
        with col4:
            st.markdown("**🔌 Architecture**")
            st.markdown("Streamlit UI → FastAPI → Agent → PostgreSQL")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    render_sidebar()
    render_main()
    render_explainer()


if __name__ == "__main__":
    main()






