import streamlit as st
import time
import sys
import tempfile
import traceback
from pathlib import Path

# Add project root to sys.path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ─── Fix: Streamlit's file-watcher crashes when it inspects torch.classes ──────
# (torch.classes.__path__ has a custom __getattr__ that Streamlit's watchdog
# module-scanner chokes on -> the whole process dies silently, which is what
# causes the browser's "Connection error" with NO Python traceback in the
# terminal). This must run BEFORE torch/whisper get imported anywhere below.
try:
    import torch
    torch.classes.__path__ = []  # type: ignore[attr-defined]
    # Fix for `CUBLAS_STATUS_NOT_SUPPORTED` on some GPU/CUDA setups
    torch.backends.cuda.matmul.allow_tf32 = False
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv

from pipeline import run_pipeline, PipelineResult
from core.rag_engine import ask_question
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.card:hover { border-color: var(--accent); }
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
}

/* ── Accent Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan { background: rgba(6,182,212,0.15); color: var(--accent-2); border: 1px solid rgba(6,182,212,0.3); }
.badge-green { background: rgba(16,185,129,0.15); color: var(--success); border: 1px solid rgba(16,185,129,0.3); }

/* ── Input & Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
}

/* ── Progress / Status ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.4rem 0;
    border: 1px solid var(--border);
    font-size: 0.8rem;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-active { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done { background: var(--success); }
.dot-pending { background: var(--border); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ── Chat ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}
.chat-msg { margin-bottom: 1rem; display: flex; flex-direction: column; gap: 0.2rem; }
.chat-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; }
.chat-bubble { display: inline-block; padding: 0.6rem 1rem; border-radius: 10px; font-size: 0.85rem; line-height: 1.6; max-width: 90%; }
.user-label { color: var(--accent-glow); }
.bot-label { color: var(--accent-2); }
.user-bubble { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }
.bot-bubble { background: rgba(6,182,212,0.1); border: 1px solid rgba(6,182,212,0.2); align-self: flex-start; }

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Stale Streamlit elements ── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_running": False,
    "rag_chain": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_status(steps: dict, key: str) -> str:
    return "dot-active" if steps.get(key) else "dot-pending"


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    source_tab, options_tab = st.tabs(["🎬 Input", "⚙️ Options"])
    with source_tab:
        st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
        source = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
        uploaded_file = st.file_uploader("Or upload a video/audio file", type=["mp4", "mkv", "avi", "mov", "mp3", "wav", "m4a"])

    with options_tab:
        st.selectbox(
            "Language",
            ["english", "hinglish", "bangla", "banglish"],
            index=0,
            key="language",
            help="Select the primary language spoken in the video."
        )

    run_btn = st.button("⚡ Process Video", use_container_width=True)

    if st.session_state.pipeline_running or st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-bar"><div class="status-dot { "dot-done" if st.session_state.pipeline_done else "dot-active" }"></div><span>{"✅ Done" if st.session_state.pipeline_done else "🏃 Running..."}</span></div>', unsafe_allow_html=True)


# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your videos</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip() and not uploaded_file:
        st.error("Please provide a YouTube URL or upload a file.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.rag_chain = None
        st.session_state.pipeline_running = True

        try:
            with st.spinner("⚙️ Pipeline running: Processing, Transcribing, Analyzing..."):
                # The new pipeline handles everything.
                # For local files, we need to save them first to get a path.
                input_source = source
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        input_source = tmp_file.name


                result = run_pipeline(
                    source=input_source,
                    language=st.session_state.language,
                )

                st.session_state.result = result
                st.session_state.pipeline_done = True
                st.session_state.rag_chain = getattr(result, "rag_chain", None)
                st.session_state.pipeline_running = False
                st.rerun()

        except (ImportError, AttributeError) as e:
            # Catch errors from missing dependencies or CUDA/torch issues
            st.session_state.pipeline_running = False
            st.session_state.pipeline_done = False
            st.error("❌ Environment Error")
            st.error(f"A required library or configuration is causing an issue: `{e}`")
            st.info("Please ensure all dependencies from `requirements.txt` are installed in your virtual environment and that your system (e.g., CUDA drivers) is correctly configured.")
            traceback.print_exc()

        except Exception as e:
            # Print the FULL traceback to the terminal (not just str(e)) so
            # the real cause is visible next time something breaks.
            st.session_state.pipeline_running = False
            st.session_state.pipeline_done = False
            print("\n" + "=" * 70)
            print("PIPELINE ERROR — full traceback:")
            traceback.print_exc()
            print("=" * 70 + "\n")
            # Check for the most common error: Invalid API Key
            if "authenticationerror" in str(e).lower() or "invalid api key" in str(e).lower():
                st.error("❌ Authentication Error: Invalid LLM API Key")
                st.info("Your LLM API key is missing or incorrect. Please follow these steps:\n1. Copy `.env.example` to a new file named `.env`.\n2. Get an API key from your provider (e.g., Groq).\n3. Paste the key into the `.env` file as `LLM_API_KEY`.")
            else:
                st.error(f"❌ An unexpected error occurred: {e}")
            st.caption("Full details were printed to the terminal running `streamlit run app.py`.")

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    video_info = getattr(r, "video_info", None)
    video_title = getattr(video_info, "title", "Untitled") if video_info else "Untitled"

    # Title banner
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Video Title</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
            {video_title}
        </div>
    </div>""", unsafe_allow_html=True)

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Chat with this Video</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for turn in st.session_state.chat_history:
            if isinstance(turn, HumanMessage):
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-end">
                    <span class="chat-label user-label">You</span>
                    <div class="chat-bubble user-bubble">{turn.content}</div>
                </div>"""
            elif isinstance(turn, AIMessage):
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-start">
                    <span class="chat-label bot-label">🤖 Assistant</span>
                    <div class="chat-bubble bot-bubble">{turn.content}</div>
                </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_input = st.text_input("Your question", placeholder="What is this video about?", label_visibility="collapsed")
    with chat_col2:
        send_btn = st.button("Ask", use_container_width=True)

    if send_btn and user_input.strip():
        if st.session_state.rag_chain is None:
            st.warning("Chat isn't available for this result (rag_chain missing).")
        else:
            with st.spinner("Thinking…"):
                try:
                    rag_answer = ask_question(
                        st.session_state.rag_chain, user_input.strip()
                    )
                    st.session_state.chat_history.append(HumanMessage(content=user_input.strip()))
                    st.session_state.chat_history.append(AIMessage(content=rag_answer))
                    st.rerun()
                except Exception as e:
                    print("\n" + "=" * 70)
                    print("CHAT ERROR — full traceback:")
                    traceback.print_exc()
                    print("=" * 70 + "\n")
                    st.error(f"❌ Chat error: {e}")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")

    # ── Feature Tabs ──────────────────────────────────────────────────────────
    summary_tab, actions_tab, decisions_tab, questions_tab, transcript_tab = st.tabs(
        ["📋 Summary", "✅ Action Items", "🔑 Key Decisions", "❓ Open Questions", "📜 Full Transcript"]
    )
    with summary_tab:
        st.markdown(f"""
        <div class="card-content">
            {getattr(r, "summary", "—")}
        </div>
        """, unsafe_allow_html=True)

    with actions_tab:
        st.markdown(f'<div class="card-content">{getattr(r, "action_items", "—")}</div>', unsafe_allow_html=True)

    with decisions_tab:
        st.markdown(f'<div class="card-content">{getattr(r, "key_decisions", "—")}</div>', unsafe_allow_html=True)

    with questions_tab:
        st.markdown(f'<div class="card-content">{getattr(r, "open_questions", "—")}</div>', unsafe_allow_html=True)

    with transcript_tab:
        st.markdown(f'<div class="transcript-box">{getattr(r, "transcript", "—")}</div>', unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Process
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
            Paste a YouTube URL or upload a video/audio file in the sidebar,
            and hit <strong>Process Video</strong> to get started.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)