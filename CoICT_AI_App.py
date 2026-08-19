"""
CoICT AI Assistant
===================
A local, offline AI chatbot built with Streamlit and Ollama (qwen2.5:0.5b).
Designed to answer questions related to the College of Information and
Communication Technologies (CoICT): Data Science, Computer Science,
Computer Networking, Telecommunication, ICT, and related fields.

No external API is required — the model runs fully locally through Ollama.

Author: CoICT Student Project
"""

import streamlit as st
import ollama
from datetime import datetime

# ----------------------------------------------------------------------
# App configuration
# ----------------------------------------------------------------------
MODEL_NAME = "qwen2.5:0.5b"

SYSTEM_PROMPT = (
    "You are CoICT AI Assistant, a helpful and knowledgeable virtual "
    "assistant for the College of Information and Communication "
    "Technologies (CoICT). Your expertise covers Data Science, Computer "
    "Science, Computer Networking, Telecommunication Engineering, "
    "Information and Communication Technology (ICT), Software "
    "Engineering, Cybersecurity, Databases, Artificial Intelligence, "
    "and all related technology fields taught within a CoICT department. "
    "Answer questions clearly, accurately, and in a friendly academic "
    "tone. When a question is outside the ICT/technology domain, "
    "politely mention that you specialize in CoICT-related topics, "
    "but still try to be helpful. Keep answers well organized, using "
    "short paragraphs, bullet points, or numbered steps when useful."
)

st.set_page_config(
    page_title="CoICT AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Custom styling (background, chat bubbles, fonts)
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Animated gradient background */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #16222a);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
    color: #eaf6ff;
}

/* Force readable light text everywhere inside the main app,
   including chat bubbles and streamed markdown responses */
.stApp, .stApp p, .stApp span, .stApp div, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp li, .stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] * {
    color: #eaf6ff !important;
}

/* Code blocks inside chat responses stay readable too */
.stApp code {
    color: #7be3ff !important;
    background: rgba(0, 0, 0, 0.35) !important;
}

.stApp pre {
    background: rgba(0, 0, 0, 0.35) !important;
    border-radius: 10px;
}

/* Chat input box container and text the user types */
[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.15);
}
[data-testid="stChatInput"] > div {
    background: transparent !important;
}
[data-testid="stChatInput"] textarea {
    background: rgba(15, 32, 39, 0.75) !important;
    color: #eaf6ff !important;
    caret-color: #eaf6ff !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(234, 246, 255, 0.55) !important;
}
[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
}
[data-testid="stChatInputContainer"] {
    background: rgba(15, 32, 39, 0.85) !important;
}

/* ------------------------------------------------------------------
   Fix: Streamlit dims/fades the whole app (reduces opacity) while
   the script is running — this includes the entire time the AI is
   streaming its answer. Against our dark background this dimming
   looks like the screen "going dark". We force full opacity and
   remove the fade transition so the interface stays bright and
   stable the whole time a response is being generated.
   ------------------------------------------------------------------ */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
section.main,
.main,
.block-container,
[data-testid="stBottomBlockContainer"],
[data-testid="stSidebar"] {
    opacity: 1 !important;
    filter: none !important;
    transition: none !important;
}

/* Some Streamlit versions apply the fade via a wrapping div with
   this data attribute during a running script — neutralize it too */
div[data-stale="true"] {
    opacity: 1 !important;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Header banner */
.coict-header {
    text-align: center;
    padding: 1.6rem 1rem 1.2rem 1rem;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}

.coict-header h1 {
    color: #FFFFFF;
    font-weight: 700;
    font-size: 2.1rem;
    margin-bottom: 0.2rem;
    letter-spacing: 0.5px;
}

.coict-header p {
    color: #cfe8ff;
    font-size: 1rem;
    margin: 0;
    font-weight: 300;
}

.coict-badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.6rem;
    letter-spacing: 0.5px;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 16px;
    padding: 0.4rem 0.6rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027 0%, #203a43 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #e6f1ff !important;
}

.sidebar-card {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 0.9rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.10);
}

.status-online {
    color: #38ff9a !important;
    font-weight: 600;
}

.status-offline {
    color: #ff6b6b !important;
    font-weight: 600;
}

/* Chat input box */
[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.15);
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def check_ollama_status():
    """Check whether Ollama is running and whether the required model exists."""
    try:
        models_response = ollama.list()
        available_models = [m.get("model", m.get("name", "")) for m in models_response.get("models", [])]
        model_found = any(MODEL_NAME in m for m in available_models)
        return True, model_found
    except Exception:
        return False, False


def get_ai_response(chat_history):
    """Stream a response from the local Ollama model."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history
    stream = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content


# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 CoICT AI Assistant")
    st.markdown(
        """
        <div class="sidebar-card">
        <b>Coverage areas:</b><br>
        • Data Science<br>
        • Computer Science<br>
        • Computer Networking<br>
        • Telecommunication<br>
        • ICT &amp; Software Engineering<br>
        • Cybersecurity &amp; Databases<br>
        • Artificial Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

    ollama_running, model_ready = check_ollama_status()

    st.markdown("### ⚙️ System Status")
    if ollama_running and model_ready:
        st.markdown(f'<span class="status-online">● Ollama running — {MODEL_NAME} ready</span>', unsafe_allow_html=True)
    elif ollama_running and not model_ready:
        st.markdown('<span class="status-offline">● Ollama running, but model not found</span>', unsafe_allow_html=True)
        st.caption(f"Run this in your terminal: `ollama pull {MODEL_NAME}`")
    else:
        st.markdown('<span class="status-offline">● Ollama is not running</span>', unsafe_allow_html=True)
        st.caption("Start it with `ollama serve`, then reload this page.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(f"Model: {MODEL_NAME} (runs 100% locally)")
    st.caption(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="coict-header">
        <h1>🎓 CoICT AI Assistant</h1>
        <p>Your offline study companion for Data Science, Computer Science, Networking, Telecommunication &amp; ICT</p>
        <span class="coict-badge">Powered by Ollama · qwen2.5:0.5b · No internet required</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Display chat history
# ----------------------------------------------------------------------
for message in st.session_state.messages:
    avatar = "🧑‍🎓" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ----------------------------------------------------------------------
# Chat input and response generation
# ----------------------------------------------------------------------
user_prompt = st.chat_input("Ask a question about Data Science, Networking, ICT, or any CoICT topic...")

if user_prompt:
    ollama_running, model_ready = check_ollama_status()

    if not ollama_running:
        st.error(
            "⚠️ Cannot reach Ollama. Please make sure Ollama is installed and "
            "running (`ollama serve`) before asking a question."
        )
    elif not model_ready:
        st.error(
            f"⚠️ The model '{MODEL_NAME}' was not found. Please run "
            f"`ollama pull {MODEL_NAME}` in your terminal first."
        )
    else:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_prompt)

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full_response = ""
            try:
                for chunk in get_ai_response(st.session_state.messages):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as error:
                full_response = (
                    "❌ Sorry, something went wrong while generating a response.\n\n"
                    f"Details: {error}"
                )
                placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})