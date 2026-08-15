import sys
from pathlib import Path
import streamlit as st

# Ensure repository root is on sys.path for direct 'streamlit run app.py' invocation
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import Config
from rag.engine import CVChatEngine
from ui.chat_feed import get_human_source_label, render_chat_history
from ui.header import render_header
from ui.hero import render_hero
from ui.sidebar import render_sidebar
from ui.styles import inject_custom_css

st.set_page_config(
    page_title="Sheharyar Nadeem // AI CV Portfolio",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_sidebar()
render_header()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Recreate engine if absent or if LLM configuration changed dynamically
if (
    "engine" not in st.session_state
    or st.session_state.get("current_llm_model") != Config.LLM_MODEL
):
    try:
        with st.spinner("Initializing AI engine..."):
            st.session_state.engine = CVChatEngine(auto_index=True)
            st.session_state.current_llm_model = Config.LLM_MODEL
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

engine: CVChatEngine = st.session_state.engine

# Show interactive suggestions when conversation is empty; otherwise render conversation feed
selected_prompt = None
if len(st.session_state.chat_history) == 0:
    selected_prompt = render_hero()
else:
    render_chat_history(st.session_state.chat_history)

user_input = st.chat_input("Ask about my experience, skills, projects, or education...")
active_query = user_input or selected_prompt

if active_query:
    # Render user prompt immediately for smooth visual feedback before streaming begins
    if len(st.session_state.chat_history) == 0:
        with st.chat_message("user"):
            st.write(active_query)

    with st.chat_message("assistant"):
        try:
            # Bound conversational turns passed to LLM to prevent context overflow
            history_tuples = [(h[0], h[1]) for h in st.session_state.chat_history[-4:]]

            with st.spinner("Searching CV..."):
                stream_gen, retrieved_docs = engine.query_stream(
                    question=active_query, chat_history=history_tuples
                )
                answer = st.write_stream(stream_gen)
                sources = sorted(
                    {
                        Path(str(doc.metadata.get("source", "CV"))).name
                        for doc in retrieved_docs
                    }
                )

            if sources:
                label = get_human_source_label(sources)
                with st.expander(label):
                    for i, doc in enumerate(retrieved_docs, 1):
                        st.markdown(
                            f"<div class='source-snippet'><div class='source-snippet-header'>Reference Snippet #{i}</div>{doc.page_content}</div>",
                            unsafe_allow_html=True,
                        )
        except Exception:
            answer = "Couldn't generate a response right now. Please try asking your question again."
            st.markdown(answer)
            sources, retrieved_docs = [], []

    st.session_state.chat_history.append(
        (active_query, answer, sources, retrieved_docs)
    )
    st.rerun()

