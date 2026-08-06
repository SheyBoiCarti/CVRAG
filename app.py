import streamlit as st
import os
import sys
import textwrap
from pathlib import Path

# Add project root to system path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import Config
from rag.engine import CVChatEngine

# Configure Streamlit page layout and theme
st.set_page_config(
    page_title="AI CV Assistant // Industrial Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Complete Industrial Skeuomorphism Design System Styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-chassis: #e0e5ec;
        --panel-fg: #f0f2f5;
        --recessed-muted: #d1d9e6;
        --text-primary: #2d3436;
        --text-muted: #4a5568;
        --accent-orange: #ff4757;
        --accent-orange-hover: #ff6b81;
        --accent-fg: #ffffff;
        --shadow-dark: #babecc;
        --shadow-light: #ffffff;
        --border-deep: #a3b1c6;
        --dark-panel: #2d3436;
        --led-green: #22c55e;
    }

    /* Overall App Chassis Background with Noise Simulation */
    .stApp {
        background-color: var(--bg-chassis) !important;
        background-image: 
            radial-gradient(circle at 12px 12px, rgba(0,0,0,0.03) 1px, transparent 1px),
            linear-gradient(rgba(255,255,255,0.4) 1px, transparent 1px);
        background-size: 24px 24px, 100% 40px;
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide default Streamlit header bar decoration */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Streamlit Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-chassis) !important;
        border-right: 2px solid var(--border-deep) !important;
        box-shadow: 6px 0 16px var(--shadow-dark);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        border-bottom: 1px solid var(--border-deep);
        padding-bottom: 6px;
        margin-top: 18px;
    }

    /* Industrial Panel Module Card */
    .ind-panel {
        background: var(--bg-chassis);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 8px 8px 18px var(--shadow-dark), -8px -8px 18px var(--shadow-light);
        border: 1px solid rgba(255, 255, 255, 0.6);
        position: relative;
    }

    /* Screws & Vent Slots Decoration */
    .screw-tl, .screw-tr, .screw-bl, .screw-br {
        position: absolute;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: radial-gradient(circle at 3px 3px, #ffffff 0%, var(--shadow-dark) 80%);
        box-shadow: inset 1px 1px 2px rgba(0,0,0,0.4), 0.5px 0.5px 1px #fff;
    }
    .screw-tl { top: 10px; left: 10px; }
    .screw-tr { top: 10px; right: 10px; }
    .screw-bl { bottom: 10px; left: 10px; }
    .screw-br { bottom: 10px; right: 10px; }

    .vent-slots {
        position: absolute;
        top: 14px;
        right: 32px;
        display: flex;
        gap: 5px;
    }
    .vent-slot {
        width: 4px;
        height: 16px;
        border-radius: 4px;
        background: var(--recessed-muted);
        box-shadow: inset 1.5px 1.5px 3px var(--shadow-dark), inset -1.5px -1.5px 3px var(--shadow-light);
    }

    /* LED Status Indicator */
    .led-container {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: var(--bg-chassis);
        padding: 6px 14px;
        border-radius: 20px;
        box-shadow: inset 3px 3px 6px var(--shadow-dark), inset -3px -3px 6px var(--shadow-light);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .led-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: var(--led-green);
        box-shadow: 0 0 10px 2px rgba(34, 197, 94, 0.7);
        animation: pulse-led 2s infinite ease-in-out;
    }
    .led-dot.orange {
        background-color: var(--accent-orange);
        box-shadow: 0 0 10px 2px rgba(255, 71, 87, 0.7);
    }
    @keyframes pulse-led {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(0.88); }
    }

    /* Header Title Typography */
    .header-brand {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        color: var(--accent-orange);
        text-transform: uppercase;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .header-title-text {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.03em;
        text-shadow: 0 1px 0 #ffffff;
        margin: 4px 0 10px 0;
    }
    .header-subtext {
        color: var(--text-muted);
        font-size: 1rem;
        line-height: 1.6;
        max-width: 800px;
    }

    /* CRT Digital Screen Visualization Panel */
    .crt-screen {
        background: #1e293b;
        background-image: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.35) 50%);
        background-size: 100% 4px;
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 16px;
        border: 2px solid var(--border-deep);
        box-shadow: inset 4px 4px 10px rgba(0,0,0,0.6), inset -2px -2px 6px rgba(255,255,255,0.1);
        font-family: 'JetBrains Mono', monospace;
        color: #38bdf8;
        font-size: 0.85rem;
    }
    .crt-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .crt-tag {
        color: #94a3b8;
        text-transform: uppercase;
    }
    .crt-val {
        color: #4ade80;
        font-weight: 700;
    }
    .crt-val.orange {
        color: var(--accent-orange);
    }

    /* Neumorphic Tactical Buttons */
    div.stButton > button {
        background-color: var(--bg-chassis) !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        border-radius: 10px !important;
        padding: 12px 18px !important;
        box-shadow: 5px 5px 12px var(--shadow-dark), -5px -5px 12px var(--shadow-light) !important;
        transition: all 0.15s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    }
    div.stButton > button:hover {
        color: var(--accent-orange) !important;
        box-shadow: 7px 7px 16px var(--shadow-dark), -7px -7px 16px var(--shadow-light) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:active {
        transform: translateY(2px) !important;
        box-shadow: inset 4px 4px 8px var(--shadow-dark), inset -4px -4px 8px var(--shadow-light) !important;
    }

    /* Primary Accent Safety-Orange Buttons */
    .stButton.accent-btn > button {
        background: var(--accent-orange) !important;
        color: var(--accent-fg) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 4px 4px 10px rgba(255, 71, 87, 0.4), -3px -3px 8px var(--shadow-light) !important;
    }
    .stButton.accent-btn > button:hover {
        background: var(--accent-orange-hover) !important;
        color: var(--accent-fg) !important;
    }

    /* Recessed Inputs & Text Area Wells */
    div[data-testid="stChatInput"] input {
        background-color: var(--bg-chassis) !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: inset 5px 5px 10px var(--shadow-dark), inset -5px -5px 10px var(--shadow-light) !important;
        padding: 16px 20px !important;
    }
    div[data-testid="stChatInput"] input:focus {
        box-shadow: inset 5px 5px 10px var(--shadow-dark), inset -5px -5px 10px var(--shadow-light), 0 0 0 2px var(--accent-orange) !important;
    }

    /* Chat Messages - Neumorphic Cards */
    div[data-testid="stChatMessage"] {
        background-color: var(--bg-chassis) !important;
        border-radius: 14px !important;
        padding: 18px 22px !important;
        margin-bottom: 16px !important;
        box-shadow: 6px 6px 14px var(--shadow-dark), -6px -6px 14px var(--shadow-light) !important;
        border: 1px solid rgba(255,255,255,0.7) !important;
    }

    /* User Message distinct style */
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {
        box-shadow: inset 4px 4px 8px var(--shadow-dark), inset -4px -4px 8px var(--shadow-light) !important;
        background-color: var(--panel-fg) !important;
    }

    /* Recessed Source Expanders */
    div[data-testid="stExpander"] {
        background-color: var(--bg-chassis) !important;
        border-radius: 10px !important;
        border: 1px solid var(--border-deep) !important;
        box-shadow: inset 3px 3px 6px var(--shadow-dark), inset -3px -3px 6px var(--shadow-light) !important;
        margin-top: 12px !important;
    }
    div[data-testid="stExpander"] summary {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        color: var(--text-muted) !important;
        text-transform: uppercase !important;
    }
    .source-box {
        background-color: #1e293b;
        color: #cbd5e1;
        border-left: 3px solid var(--accent-orange);
        padding: 12px 16px;
        border-radius: 8px;
        margin-top: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.5;
        box-shadow: inset 2px 2px 5px rgba(0,0,0,0.5);
    }

    /* Stamped Badge Pills */
    .stamped-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        background-color: var(--bg-chassis);
        box-shadow: inset 2px 2px 4px var(--shadow-dark), inset -2px -2px 4px var(--shadow-light);
        color: var(--accent-orange);
        border: 1px solid var(--border-deep);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of tuples: (human_msg, ai_msg, sources, docs)

if "engine" not in st.session_state or st.session_state.get("current_llm_model") != Config.LLM_MODEL:
    try:
        with st.spinner("INITIALIZING RAG ENGINE & NVIDIA HARDWARE ENDPOINTS..."):
            st.session_state.engine = CVChatEngine(auto_index=True)
            st.session_state.current_llm_model = Config.LLM_MODEL
    except Exception as e:
        st.error(f"SYSTEM ERROR: Failed to initialize RAG Engine: {e}")
        st.stop()

engine: CVChatEngine = st.session_state.engine

# Sidebar Industrial Control Monitor
with st.sidebar:
    st.markdown(textwrap.dedent("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #ff4757; font-weight: 800; letter-spacing: 0.1em; margin-bottom: 8px;">
        CONSOLE // SYSTEM STATS
    </div>
    """), unsafe_allow_html=True)
    
    # Document Count Status Badge
    doc_count = engine.vector_manager.get_document_count()
    st.markdown(textwrap.dedent(f"""
    <div class="led-container">
        <div class="led-dot"></div>
        <span>INDEXED CHUNKS: {doc_count}</span>
    </div>
    """), unsafe_allow_html=True)
    
    st.markdown("### SOURCE DOCUMENTS")
    doc_files = [f.name for f in Config.DOCUMENTS_DIR.glob("*") if f.is_file() and not f.name.startswith(".")]
    if doc_files:
        for fname in doc_files:
            st.markdown(textwrap.dedent(f"""
            <div class="stamped-badge" style="width: 100%; margin-bottom: 6px; color: var(--text-primary);">
                {fname}
            </div>
            """), unsafe_allow_html=True)
    else:
        st.warning("NO FILES FOUND IN DOCUMENT BAY.")

    st.markdown("### SYSTEM ARCHITECTURE")
    st.markdown(textwrap.dedent(f"""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; line-height: 1.8; color: var(--text-muted);">
        <div><b>LLM:</b> <span style="color: var(--accent-orange);">{Config.LLM_MODEL}</span></div>
        <div><b>EMBEDDINGS:</b> <span style="color: var(--text-primary);">{Config.EMBEDDING_MODEL}</span></div>
        <div><b>VECTOR DB:</b> <span style="color: var(--text-primary);">ChromaDB Persistent</span></div>
        <div><b>CHUNK SIZE:</b> <span style="color: var(--text-primary);">{Config.CHUNK_SIZE} chars</span></div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown("---")

    # Tactile Physical Action Buttons
    if st.button("RE-INDEX VECTOR DB", use_container_width=True):
        with st.spinner("RE-BUILDING VECTOR EMBEDDINGS..."):
            engine.ensure_indexed(force_reindex=True)
            st.success("VECTOR BAY RE-INDEXED!")
            st.rerun()

    if st.button("CLEAR CHAT LOGS", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main Header Component - Industrial Spacecraft Instrument Panel
st.markdown("""
<div class="ind-panel">
    <div class="screw-tl"></div>
    <div class="screw-tr"></div>
    <div class="screw-bl"></div>
    <div class="screw-br"></div>
    <div class="vent-slots">
        <div class="vent-slot"></div>
        <div class="vent-slot"></div>
        <div class="vent-slot"></div>
    </div>
    <div class="header-brand">
        <span>SYSTEM // CV-RAG-ASSISTANT v2.0</span>
        <div class="led-container" style="padding: 3px 8px; font-size: 0.7rem;">
            <div class="led-dot"></div>
            <span>ONLINE</span>
        </div>
    </div>
    <div class="header-title-text">Candidate Intelligence Console</div>
    <div class="header-subtext">
        Perform tactile, grounded query analysis on candidate professional backgrounds, skillsets, employment records, and achievements.
    </div>
    <div class="crt-screen">
        <div class="crt-row">
            <span><span class="crt-tag">[HARDWARE STATUS]:</span> <span class="crt-val">READY</span></span>
            <span><span class="crt-tag">[RETRIEVAL ENGINE]:</span> <span class="crt-val">CHROMA VECTOR SEARCH</span></span>
            <span><span class="crt-tag">[LLM BACKBONE]:</span> <span class="crt-val orange">LLAMA-3.3-70B</span></span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tactical Quick Action Chips
st.markdown(textwrap.dedent("""
<div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 800; color: var(--text-muted); letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px;">
    PRE-SET COMMAND CHIPS
</div>
"""), unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

selected_prompt = None
with col1:
    if st.button("SUMMARY", use_container_width=True):
        selected_prompt = "Give me a summary of the candidate's professional background and key strengths."
with col2:
    if st.button("SKILLS", use_container_width=True):
        selected_prompt = "What technical skills, programming languages, and tools does the candidate possess?"
with col3:
    if st.button("EXPERIENCE", use_container_width=True):
        selected_prompt = "What are the candidate's key employment roles and achievements?"
with col4:
    if st.button("EDUCATION", use_container_width=True):
        selected_prompt = "What education, degrees, or certifications does the candidate hold?"

st.markdown("<br>", unsafe_allow_html=True)

# Render Chat History
for item in st.session_state.chat_history:
    user_msg = item[0]
    ai_msg = item[1]
    sources = item[2] if len(item) > 2 else []
    docs = item[3] if len(item) > 3 else []

    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        st.write(ai_msg)
        if sources:
            with st.expander(f"CITED SOURCES ({', '.join(sources)})"):
                for i, doc in enumerate(docs, 1):
                    src_file = os.path.basename(doc.metadata.get('source', 'CV'))
                    st.markdown(f"**SNIPPET #{i}** *(Source: `{src_file}`)*:")
                    st.markdown(f"<div class='source-box'>{doc.page_content}</div>", unsafe_allow_html=True)

# Chat Input Well
user_input = st.chat_input("ENTER QUERY COMMAND...")

# Process Input Execution
active_query = user_input or selected_prompt

if active_query:
    with st.chat_message("user"):
        st.write(active_query)

    with st.chat_message("assistant"):
        try:
            # Limit memory history context to last 4 turns for optimal speed and context window control
            history_tuples = [(h[0], h[1]) for h in st.session_state.chat_history[-4:]]
            stream_gen, sources, retrieved_docs = engine.query_stream(question=active_query, chat_history=history_tuples)
            
            answer = st.write_stream(stream_gen)

            if sources:
                with st.expander(f"CITED SOURCES ({', '.join(sources)})"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        src_file = os.path.basename(doc.metadata.get('source', 'CV'))
                        st.markdown(f"**SNIPPET #{i}** *(Source: `{src_file}`)*:")
                        st.markdown(f"<div class='source-box'>{doc.page_content}</div>", unsafe_allow_html=True)
        except Exception as e:
            answer = "**Service Notice**: The AI service is currently busy or rate-limited. Please wait a moment and try your question again."
            st.markdown(answer)
            sources = []
            retrieved_docs = []

    st.session_state.chat_history.append((active_query, answer, sources, retrieved_docs))

