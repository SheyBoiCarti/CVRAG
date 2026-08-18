from __future__ import annotations

import os
from unittest.mock import MagicMock
from langchain_core.documents import Document
import pytest
import streamlit as st

from rag.models import Citation
from rag.presentation import citation_view, query_error_message
from rag.runtime import get_pipeline, load_streamlit_secrets_into_environ


def test_get_pipeline_caches_resource(monkeypatch, tmp_path):
    created = []

    class FakePipeline:
        def __init__(self, settings=None):
            self.settings = settings
            created.append(self)

    monkeypatch.setattr("rag.runtime.RAGPipeline", FakePipeline)
    get_pipeline.clear()
    try:
        p1 = get_pipeline(llm_model="test-llm", embedding_model="test-embed")
        p2 = get_pipeline(llm_model="test-llm", embedding_model="test-embed")

        assert p1 is p2
        assert len(created) == 1
        assert p1.settings.LLM_MODEL == "test-llm"
        assert p1.settings.EMBEDDING_MODEL == "test-embed"
    finally:
        get_pipeline.clear()


def test_load_streamlit_secrets_into_environ(monkeypatch):
    monkeypatch.delenv("TEST_STREAMLIT_KEY", raising=False)
    monkeypatch.delenv("EXISTING_KEY", raising=False)
    monkeypatch.setenv("EXISTING_KEY", "existing_value")

    fake_secrets = {
        "TEST_STREAMLIT_KEY": "secret_abc_123",
        "EXISTING_KEY": "should_not_overwrite",
        "NUMERIC_SECRET": 42,
    }

    monkeypatch.setattr(st, "secrets", fake_secrets, raising=False)
    load_streamlit_secrets_into_environ()

    assert os.environ.get("TEST_STREAMLIT_KEY") == "secret_abc_123"
    assert os.environ.get("EXISTING_KEY") == "existing_value"
    assert os.environ.get("NUMERIC_SECRET") == "42"


def test_citation_view_with_domain_citation():
    citation = Citation(
        evidence_id="S1",
        chunk_id="chk-001",
        source="Sheharyar_Nadeem_CV.pdf",
        page=2,
        excerpt="Led development of cloud RAG systems.",
        support_status="supported",
    )

    view = citation_view(citation)
    assert view.label == "S1 · Sheharyar_Nadeem_CV.pdf · page 2"
    assert view.excerpt == "Led development of cloud RAG systems."


def test_citation_view_with_langchain_document():
    doc = Document(
        page_content="Senior AI Engineer with Python expertise.",
        metadata={"source": "C:/docs/resume.pdf", "page": 0, "evidence_id": "S3"},
    )

    view = citation_view(doc)
    assert view.label == "S3 · resume.pdf · page 1"
    assert view.excerpt == "Senior AI Engineer with Python expertise."
