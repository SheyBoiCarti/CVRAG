from __future__ import annotations

import dataclasses
import os
from typing import Any
import streamlit as st

from rag.engine import CVChatEngine
from rag.pipeline import RAGPipeline
from rag.settings import Settings, get_settings


def load_streamlit_secrets_into_environ() -> None:
    """Loads secrets from streamlit.secrets into os.environ if present."""
    try:
        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            for k, v in secrets.items():
                if isinstance(v, (str, int, float, bool)):
                    if not os.environ.get(k):
                        os.environ[k] = str(v)
    except Exception:
        # Secrets might not be configured in local dev environment
        pass


@st.cache_resource(show_spinner=False)
def get_pipeline(
    llm_model: str | None = None,
    embedding_model: str | None = None,
    collection_name: str | None = None,
) -> RAGPipeline:
    """Builds one shared, stateless RAG pipeline per effective configuration."""
    load_streamlit_secrets_into_environ()
    settings = get_settings()
    overrides: dict[str, Any] = {}
    if llm_model:
        overrides["LLM_MODEL"] = llm_model
    if embedding_model:
        overrides["EMBEDDING_MODEL"] = embedding_model
    if overrides:
        settings = dataclasses.replace(settings, **overrides)
    del collection_name
    return RAGPipeline(settings=settings)


@st.cache_resource(show_spinner=False)
def get_engine(
    llm_model: str | None = None,
    embedding_model: str | None = None,
    collection_name: str | None = None,
) -> CVChatEngine:
    """Builds one shared, stateless RAG engine per effective configuration (backward compatibility)."""
    del llm_model, embedding_model, collection_name
    return CVChatEngine(auto_index=True)


__all__ = [
    "load_streamlit_secrets_into_environ",
    "get_pipeline",
    "get_engine",
]
