from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterator, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from rag.provider_factories import ProviderFactories
from rag.settings import Settings


class FakeEmbeddings(Embeddings):
    """Deterministic in-memory embedding model producing stable float vectors for tests."""

    def __init__(self, dimension: int = 4096) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
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
    response_fn: Optional[Callable[[list[BaseMessage]], str]] = None
    error: Optional[Exception] = None
    error_after_chunks: Optional[int] = None
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
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        content = self._get_next_response(messages)
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
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


class FakeSupportVerifier:
    """Deterministic offline claim support verifier stub."""

    def verify(self, claim: str, evidence: list[Any]) -> dict[str, Any]:
        return {"status": "supported", "reason": "fake_verified"}


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
        response_fn: Optional[Callable[[list[BaseMessage]], str]] = None,
        embedding_dimension: int = 4096,
    ) -> None:
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


__all__ = [
    "FakeChatModel",
    "FakeEmbeddings",
    "FakeProviderFactories",
    "FakeReranker",
    "FakeSupportVerifier",
]
