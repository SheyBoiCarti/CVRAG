from __future__ import annotations

import logging
import random
import re
import time
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from rag.config import Config


LOGGER = logging.getLogger(__name__)
STATUS_PATTERN = re.compile(r"\[(\d{3})\]")
BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[^\s,;]+")
API_KEY_PATTERN = re.compile(
    r"(?i)((?:nvidia_)?api[_ -]?key\s*[:=]\s*)[^\s,;]+"
)
NVAPI_PATTERN = re.compile(r"\bnvapi-[A-Za-z0-9_-]+\b")
MAX_PROVIDER_DETAIL_CHARS = 300


def provider_error_details(exc: Exception) -> tuple[int | None, str]:
    """Returns a bounded provider error summary with credential shapes redacted."""
    raw_status = getattr(exc, "status_code", None)
    status = raw_status if isinstance(raw_status, int) else None
    detail = next(
        (line.strip() for line in str(exc).splitlines() if line.strip()),
        "",
    )
    if status is None:
        match = STATUS_PATTERN.search(detail)
        status = int(match.group(1)) if match else None
    detail = BEARER_PATTERN.sub(r"\1[REDACTED]", detail)
    detail = API_KEY_PATTERN.sub(r"\1[REDACTED]", detail)
    detail = NVAPI_PATTERN.sub("[REDACTED]", detail)
    return status, detail[:MAX_PROVIDER_DETAIL_CHARS] or "no provider detail"


def is_transient_error(exc: Exception) -> bool:
    """Classifies network drops, rate limits (HTTP 429), and upstream 5xx errors as retryable."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status_code, msg = provider_error_details(exc)
    if status_code == 429 or (
        isinstance(status_code, int) and 500 <= status_code < 600
    ):
        return True
    name = type(exc).__name__.lower()
    if any(k in name for k in ("timeout", "connection", "ratelimit")):
        return True
    msg = msg.lower()
    if any(
        phrase in msg
        for phrase in (
            "resourceexhausted",
            "rate limit",
            "too many requests",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "temporarily unavailable",
        )
    ):
        return True
    return False


def retry_transient(
    operation,
    max_attempts: int,
    *,
    operation_name: str = "provider_request",
    model: str = "unknown",
):
    """Executes a callable with exponential backoff and jitter for transient failures."""
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            transient = is_transient_error(exc)
            status, detail = provider_error_details(exc)
            status_label = status if status is not None else "unknown"
            if not transient or attempt == max_attempts:
                LOGGER.error(
                    "Provider request failed (operation=%s, model=%s, "
                    "attempt=%s/%s, type=%s, status=%s, detail=%s)",
                    operation_name,
                    model,
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                    status_label,
                    detail,
                )
                raise
            delay = min(4.0, 2 ** (attempt - 1)) + random.uniform(0, 0.25)
            LOGGER.warning(
                "Transient provider failure; retrying (operation=%s, model=%s, "
                "attempt=%s/%s, type=%s, status=%s, detail=%s)",
                operation_name,
                model,
                attempt,
                max_attempts,
                type(exc).__name__,
                status_label,
                detail,
            )
            time.sleep(delay)
    raise RuntimeError("Retry loop exited unexpectedly")


class RobustNVIDIAEmbeddings(Embeddings):
    """NVIDIA embedding client with bounded batching and transient retries."""

    def __init__(
        self,
        model: str,
        api_key: str,
        batch_size: int = 15,
        max_retries: int = 3,
        timeout: float = 45.0,
    ):
        self.client = NVIDIAEmbeddings(
            model=model,
            api_key=api_key,
            truncate="NONE",
        )
        # NVIDIAEmbeddings 1.4.3 accepts but ignores a top-level timeout keyword.
        # Apply the configured bound directly to both pinned transport clients.
        self.client._client.timeout = timeout
        self.client._async_client.timeout = timeout
        self.batch_size = batch_size
        self.max_retries = max_retries

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Splits document texts into bounded batches to satisfy upstream API limits."""
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            batch_embeddings = retry_transient(
                lambda batch=batch: self.client.embed_documents(batch),
                self.max_retries,
                operation_name="embed_documents",
                model=str(self.client.model),
            )
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query string with transient retry support."""
        return retry_transient(
            lambda: self.client.embed_query(text),
            self.max_retries,
            operation_name="embed_query",
            model=str(self.client.model),
        )


class VectorStoreManager:
    """Manages local Chroma vector storage with versioned collection names."""

    def __init__(
        self,
        persist_dir: Path | None = None,
        embeddings: Embeddings | None = None,
        collection_name: str | None = None,
        settings: Any | None = None,
    ):
        self.settings = settings
        if self.settings is not None:
            if embeddings is None:
                self.settings.validate_runtime()
            self.persist_dir = str(persist_dir or self.settings.VECTOR_DB_DIR)
            self.embeddings = embeddings or RobustNVIDIAEmbeddings(
                model=self.settings.EMBEDDING_MODEL,
                api_key=self.settings.NVIDIA_API_KEY or "",
                batch_size=self.settings.EMBEDDING_BATCH_SIZE,
                max_retries=self.settings.MAX_RETRIES,
                timeout=self.settings.REQUEST_TIMEOUT,
            )
            self.collection_name = collection_name or self.settings.COLLECTION_NAME
            self.retrieval_k = self.settings.RETRIEVAL_K
        else:
            if embeddings is None:
                Config.validate()
            self.persist_dir = str(persist_dir or Config.VECTOR_DB_DIR)
            self.embeddings = embeddings or RobustNVIDIAEmbeddings(
                model=Config.EMBEDDING_MODEL,
                api_key=Config.NVIDIA_API_KEY or "",
                batch_size=Config.EMBEDDING_BATCH_SIZE,
                max_retries=Config.MAX_RETRIES,
                timeout=Config.REQUEST_TIMEOUT,
            )
            self.collection_name = collection_name or Config.COLLECTION_NAME
            self.retrieval_k = Config.RETRIEVAL_K

        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

    def ensure_documents(self, chunks: list[Document]) -> int:
        """Idempotently adds document chunks using their deterministic chunk_id."""
        if not chunks:
            return self.get_document_count()
        ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
        self.vector_store.add_documents(documents=chunks, ids=ids)
        return self.get_document_count()

    def get_retriever(self, search_kwargs: dict | None = None):
        """Returns a LangChain retriever interface for vector similarity search."""
        kwargs = search_kwargs or {"k": self.retrieval_k}
        return self.vector_store.as_retriever(search_kwargs=kwargs)

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        """Performs vector similarity search returning top-k Document chunks."""
        return self.vector_store.similarity_search(
            query=query, k=k or self.retrieval_k
        )

    def similarity_search_with_scores(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """Performs vector similarity search returning Document chunks and relevance scores."""
        return self.vector_store.similarity_search_with_relevance_scores(
            query=query, k=k or self.retrieval_k
        )

    def get_document_count(self) -> int:
        """Returns the total number of indexed document chunks in the Chroma collection."""
        return int(self.vector_store._collection.count())

