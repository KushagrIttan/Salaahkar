"""
Salaahkar - Streamlit Chat Interface
A modern chat UI that connects to the RAG agent for academic Q&A.
"""

import streamlit as st
from agent import build_chain


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salaahkar — Academic Tutor",
    page_icon="📚",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header area */
    .header-container {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .header-container h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF 0%, #48C6EF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .header-container p {
        color: #9ca3af;
        font-size: 1rem;
        margin-top: 0;
    }

    /* Chat bubbles */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
    }

    /* Input bar */
    .stChatInput > div {
        border-radius: 12px !important;
    }

    /* Divider */
    .divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #6C63FF44, transparent);
        margin: 0.5rem 0 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <h1>📚 Salaahkar</h1>
    <p>Your AI-powered academic tutor — ask anything from your notes</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    with st.spinner("🔄 Loading knowledge base..."):
        st.session_state.chain = build_chain()

# ── Render Chat History ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "📚"):
        st.markdown(msg["content"])

# ── Chat Input ───────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask Salaahkar a question..."):
    # Display user message
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get assistant response
    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke({"input": user_input})
            answer = response["answer"]
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
