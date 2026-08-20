from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Protocol, runtime_checkable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

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


class FakeEmbeddings(Embeddings):
    """Deterministic in-memory embedding model producing stable float vectors for tests."""

    def __init__(self, dimension: int = 4096) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        import hashlib

        import numpy as np

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dimension).astype(float)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeChatModel(BaseChatModel):
    """Deterministic in-memory chat model supporting invoke, stream, and call recording."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    responses: list[str] = Field(default_factory=list)
    default_response: str = "Fake answer grounded in evidence [S1]."
    response_fn: Callable[[list[BaseMessage]], str] | None = None
    error: Exception | None = None
    error_after_chunks: int | None = None
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    _index: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _get_next_response(self, messages: list[BaseMessage]) -> str:
        if self.error is not None:
            raise self.error
        if self.response_fn is not None:
            return self.response_fn(messages)
        if self.responses:
            if self._index < len(self.responses):
                resp = self.responses[self._index]
                self._index += 1
                return resp
            return self.responses[-1]
        return self.default_response

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:

        self.calls.append(messages)
        content = self._get_next_response(messages)
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:

        self.calls.append(messages)
        content = self._get_next_response(messages)
        words = content.split(" ")
        for i, word in enumerate(words):
            if self.error_after_chunks is not None and i >= self.error_after_chunks:
                raise RuntimeError("Simulated mid-stream provider error")
            token = word if i == len(words) - 1 else word + " "
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk


class FakeReranker:
    """Deterministic offline reranker stub."""

    def rerank(self, query: str, documents: list[Any]) -> list[Any]:
        return documents


class FakeProviderFactories:
    """Offline provider factory producing in-memory fakes without network calls."""

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        chat_model: BaseChatModel | None = None,
        reranker: Any | None = None,
        support_verifier: Any | None = None,
        responses: list[str] | None = None,
        default_response: str = "Fake answer grounded in evidence [S1].",
        response_fn: Any | None = None,
        embedding_dimension: int = 4096,
    ) -> None:
        from rag.verification.support import FakeSupportVerifier

        self.embeddings = embeddings or FakeEmbeddings(dimension=embedding_dimension)
        self.chat_model = chat_model or FakeChatModel(
            responses=list(responses) if responses is not None else [],
            default_response=default_response,
            response_fn=response_fn,
        )
        self.reranker = reranker or FakeReranker()
        self.support_verifier = support_verifier or FakeSupportVerifier()

    def create_embeddings(self, settings: Settings) -> Embeddings:
        return self.embeddings

    def create_chat_model(self, settings: Settings) -> BaseChatModel:
        return self.chat_model

    def create_reranker(self, settings: Settings) -> Any:
        return self.reranker

    def create_support_verifier(self, settings: Settings) -> Any:
        return self.support_verifier


def get_provider_factories(settings: Settings | None = None) -> ProviderFactories:
    """Returns the appropriate provider factory depending on configuration mode."""
    if settings is not None and getattr(settings, "PROVIDER_MODE", "live") == "fake":
        return FakeProviderFactories()
    return LiveProviderFactories()


__all__ = [
    "FakeChatModel",
    "FakeEmbeddings",
    "FakeProviderFactories",
    "FakeReranker",
    "LiveProviderFactories",
    "ProviderFactories",
    "get_provider_factories",
]
