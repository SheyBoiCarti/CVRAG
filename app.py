import logging
import sys
from pathlib import Path
import streamlit as st

# Ensure repository root is on sys.path for direct 'streamlit run app.py' invocation
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import Config
from rag.engine import CVChatEngine, NO_RESULTS_ANSWER
from rag.presentation import query_error_message
from rag.runtime import get_engine
from rag.vector_store import provider_error_details
from ui.chat_feed import get_human_source_label, render_chat_history
from ui.header import render_header
from ui.hero import render_hero
from ui.sidebar import render_sidebar
from ui.styles import inject_custom_css


LOGGER = logging.getLogger(__name__)

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

try:
    with st.spinner("Initializing AI engine..."):
        engine: CVChatEngine = get_engine(
            Config.LLM_MODEL or "",
            Config.EMBEDDING_MODEL or "",
            Config.COLLECTION_NAME,
        )
except Exception as exc:
    status, detail = provider_error_details(exc)
    LOGGER.error(
        "AI engine initialization failed (embedding_model=%s, status=%s, "
        "type=%s, detail=%s)",
        Config.EMBEDDING_MODEL,
        status if status is not None else "unknown",
        type(exc).__name__,
        detail,
    )
    st.error(
        "The AI search index could not be initialized. "
        "Please try again later; the deployment owner can check the server logs."
    )
    st.stop()

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
        stage = "retrieval"
        try:
            # Bound conversational turns passed to LLM to prevent context overflow
            history_tuples = [(h[0], h[1]) for h in st.session_state.chat_history[-4:]]

            with st.spinner("Searching CV..."):
                prepared = engine.prepare_query(
                    question=active_query, chat_history=history_tuples
                )

            retrieved_docs = prepared.documents
            if retrieved_docs:
                stage = "generation"
                with st.spinner("Generating answer..."):
                    answer = st.write_stream(engine.stream_prepared(prepared))
            else:
                answer = NO_RESULTS_ANSWER
                st.markdown(answer)

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
        except Exception as exc:
            LOGGER.warning(
                "CV query failed (stage=%s, type=%s)",
                stage,
                type(exc).__name__,
            )
            answer = query_error_message(exc)
            st.markdown(answer)
            sources, retrieved_docs = [], []

    st.session_state.chat_history.append(
        (active_query, answer, sources, retrieved_docs)
    )
    st.rerun()
