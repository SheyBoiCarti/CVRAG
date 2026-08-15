import streamlit as st

from rag.engine import CVChatEngine


@st.cache_resource(show_spinner=False)
def get_engine(
    llm_model: str,
    embedding_model: str,
    collection_name: str,
) -> CVChatEngine:
    """Builds one shared, stateless RAG engine per effective configuration."""
    del llm_model, embedding_model, collection_name
    return CVChatEngine(auto_index=True)
