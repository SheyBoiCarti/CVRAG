import logging
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

from rag.engine import CVChatEngine
from rag.errors import ProviderUnavailableError
from rag.presentation import citation_view


LOGGER = logging.getLogger(__name__)
SUGGESTED_QUESTIONS = {
    "Summary": "Give me a concise summary of your professional background and strengths.",
    "Skills": "What technical skills, programming languages, and tools are listed?",
    "Experience": "What are your key employment roles and achievements?",
    "Education": "What education, degrees, and certifications are listed?",
}

st.set_page_config(
    page_title="Chat with my CV",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stMainBlockContainer { max-width: 860px; padding-top: 3rem; padding-bottom: 7rem; }
      h1 { letter-spacing: -0.035em; }
      .stButton button, .stPills button { min-height: 42px; }
      button:focus-visible, input:focus-visible, textarea:focus-visible {
        outline: 3px solid #2563EB !important;
        outline-offset: 2px !important;
      }
      @media (max-width: 640px) {
        .stMainBlockContainer { padding: 1.25rem 1rem 6rem; }
        h1 { font-size: 2rem !important; }
      }
      @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
          animation-duration: 0.01ms !important;
          animation-iteration-count: 1 !important;
          transition-duration: 0.01ms !important;
          scroll-behavior: auto !important;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_engine() -> CVChatEngine:
    return CVChatEngine(auto_index=True)


def serialize_citations(documents: list[Document]) -> list[dict[str, str]]:
    serialized = []
    for document in documents:
        view = citation_view(document)
        serialized.append({"label": view.label, "excerpt": view.excerpt})
    return serialized


def render_citations(citations: list[dict[str, str]]) -> None:
    if not citations:
        return
    with st.expander("Sources"):
        for citation in citations:
            st.caption(citation["label"])
            st.text(citation["excerpt"])


def conversation_history(messages: list[dict]) -> list[tuple[str, str]]:
    pairs = []
    pending_user = None
    for message in messages:
        if message["role"] == "user":
            pending_user = message["content"]
        elif message["role"] == "assistant" and pending_user is not None:
            pairs.append((pending_user, message["content"]))
            pending_user = None
    return pairs[-4:]


if "messages" not in st.session_state:
    st.session_state.messages = []

header_left, header_right = st.columns([5, 1], vertical_alignment="center")
with header_left:
    st.title("Chat with my CV")
with header_right:
    if st.button("Clear", type="tertiary", disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

st.write(
    "Ask about my experience, skills, education, and projects. "
    "Answers are grounded in my CV."
)

try:
    with st.spinner("Preparing the CV assistant…"):
        engine = get_engine()
except Exception as exc:
    LOGGER.error("CV assistant initialization failed (type=%s)", type(exc).__name__)
    engine = None
    st.error("The CV assistant could not start. Check the app configuration and try again.")

selected_question = None
if not st.session_state.messages and engine is not None:
    st.write("Try a question")
    selected_label = st.pills(
        "Suggested questions",
        list(SUGGESTED_QUESTIONS),
        label_visibility="collapsed",
    )
    if selected_label:
        selected_question = SUGGESTED_QUESTIONS[selected_label]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_citations(message.get("citations", []))

typed_question = st.chat_input(
    "Ask a question about my CV",
    disabled=engine is None,
    max_chars=1000,
)
active_question = typed_question or selected_question

if active_question and engine is not None:
    request_id = uuid.uuid4().hex[:12]
    history = conversation_history(st.session_state.messages)
    st.session_state.messages.append(
        {"role": "user", "content": active_question, "citations": []}
    )
    with st.chat_message("user"):
        st.markdown(active_question)

    answer = ""
    cited_documents: list[Document] = []
    with st.chat_message("assistant"):
        try:
            stream, retrieved_documents = engine.query_stream(
                active_question, chat_history=history
            )
            answer = st.write_stream(stream)
            cited_documents = engine.cited_documents(answer, retrieved_documents)
            render_citations(serialize_citations(cited_documents))
        except (ProviderUnavailableError, TimeoutError, ConnectionError) as exc:
            LOGGER.warning(
                "CV answer unavailable (request_id=%s, type=%s)",
                request_id,
                type(exc).__name__,
            )
            answer = (
                "The AI service is temporarily unavailable. Please try again. "
                f"Reference: {request_id}"
            )
            st.error(answer)
        except ValueError as exc:
            answer = str(exc)
            st.warning(answer)
        except Exception as exc:
            LOGGER.error(
                "CV answer failed (request_id=%s, type=%s)",
                request_id,
                type(exc).__name__,
            )
            answer = f"Something went wrong. Please try again. Reference: {request_id}"
            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": serialize_citations(cited_documents),
        }
    )
    st.session_state.messages = st.session_state.messages[-40:]
