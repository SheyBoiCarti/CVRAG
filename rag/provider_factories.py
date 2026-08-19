from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from rag.settings import Settings


@runtime_checkable
class ProviderFactories(Protocol):
    """Protocol defining swappable factory constructors for embeddings and chat models."""

    def create_embeddings(self, settings: Settings) -> Embeddings:
        """Creates an embedding model client configured with settings."""
        ...

    def create_chat_model(self, settings: Settings) -> BaseChatModel:
        """Creates a chat model client configured with settings."""
        ...

    def create_reranker(self, settings: Settings) -> Any:
        """Creates a reranker client configured with settings."""
        ...

    def create_support_verifier(self, settings: Settings) -> Any:
        """Creates a claim support verifier client configured with settings."""
        ...


class LiveProviderFactories:
    """Production provider factory instantiating live NVIDIA endpoints.

    No network requests or live client instances are constructed at module import time.
    """

    def create_embeddings(self, settings: Settings) -> Embeddings:
        """Creates a robust NVIDIA embeddings client requiring configured API key."""
        if not settings.NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY is not configured.")
        from rag.vector_store import RobustNVIDIAEmbeddings

        return RobustNVIDIAEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.NVIDIA_API_KEY,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            max_retries=settings.MAX_RETRIES,
            timeout=settings.REQUEST_TIMEOUT,
        )

    def create_chat_model(self, settings: Settings) -> BaseChatModel:
        """Creates a ChatNVIDIA language model client requiring configured API key."""
        if not settings.NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY is not configured.")
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        return ChatNVIDIA(
            model=settings.LLM_MODEL,
            api_key=settings.NVIDIA_API_KEY,
            temperature=0.2,
            top_p=0.7,
            max_completion_tokens=800,
            timeout=settings.REQUEST_TIMEOUT,
        )

    def create_reranker(self, settings: Settings) -> Any:
        """Creates an NVIDIA reranker client requiring configured API key."""
        if not settings.NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY is not configured.")
        from langchain_nvidia_ai_endpoints import NVIDIARerank

        return NVIDIARerank(
            model=settings.RERANK_MODEL,
            api_key=settings.NVIDIA_API_KEY,
            top_n=settings.RERANK_CANDIDATE_K,
        )

    def create_support_verifier(self, settings: Settings) -> Any:
        """Creates an NVIDIA claim support verifier requiring configured API key."""
        if not settings.NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY is not configured.")
        from rag.verification.support import NvidiaSupportVerifier

        return NvidiaSupportVerifier(settings=settings)


__all__ = ["LiveProviderFactories", "ProviderFactories"]
