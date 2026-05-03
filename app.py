"""
Salaahkar - Phase 1 Streamlit Chat Interface
Modern chat UI with sidebar controls, citation expander, and persistent history.
"""

import streamlit as st
from agent import setup_salaahkar_agent


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

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        background: linear-gradient(135deg, #6C63FF 0%, #48C6EF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.4rem;
    }

    /* Source expander */
    .stExpander {
        border: 1px solid #6C63FF33 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Salaahkar Settings")
    st.markdown("---")

    temperature = st.slider(
        "🌡️ LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower = more factual & deterministic. Higher = more creative.",
    )

    top_k = st.slider(
        "🔎 Top-K Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of document chunks each retriever returns.",
    )

    st.markdown("---")
    st.markdown(
        "<p style='color:#9ca3af; font-size:0.8rem; text-align:center;'>"
        "Powered by Llama 3.2 · nomic-embed-text<br>Phase 1 — Hybrid RAG</p>",
        unsafe_allow_html=True,
    )

    # Rebuild chain when settings change
    settings_key = f"{temperature}_{top_k}"
    if st.session_state.get("_settings_key") != settings_key:
        st.session_state["_settings_key"] = settings_key
        st.session_state.pop("chain", None)  # force rebuild

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
    with st.spinner("🔄 Loading knowledge base (hybrid search + multi-query) ..."):
        st.session_state.chain = setup_salaahkar_agent(
            temperature=temperature,
            top_k=top_k,
        )

# ── Helper: render source documents inside an expander ───────────────────────
def render_sources(sources):
    """Display retrieved source chunks in a collapsible expander."""
    if not sources:
        return
    with st.expander("📚 View Retrieved Sources", expanded=False):
        for i, doc in enumerate(sources, 1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "—")
            st.markdown(f"**Chunk {i}** · `{src}` · page {page}")
            st.caption(doc.page_content[:500])
            if i < len(sources):
                st.markdown("---")


# ── Render Chat History ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "📚"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # Re-render sources for assistant messages that have them
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

# ── Chat Input ───────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask Salaahkar a question..."):
    # Display user message
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get assistant response (returns 'answer' + 'context')
    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke({"input": user_input})
            answer = response["answer"]
            sources = response.get("context", [])
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
