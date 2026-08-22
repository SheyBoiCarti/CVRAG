"""Public package interface for the fixed-CV RAG demo."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "Config":
        from rag.config import Config

        return Config
    if name == "CVDocumentLoader":
        from rag.document_loader import CVDocumentLoader

        return CVDocumentLoader
    if name == "VectorStoreManager":
        from rag.vector_store import VectorStoreManager

        return VectorStoreManager
    if name == "CVChatEngine":
        from rag.engine import CVChatEngine

        return CVChatEngine
    raise AttributeError(f"module 'rag' has no attribute {name!r}")


__all__ = ["CVChatEngine", "CVDocumentLoader", "Config", "VectorStoreManager"]

