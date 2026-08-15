import logging
import random
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from rag.config import Config


LOGGER = logging.getLogger(__name__)


def is_transient_error(exc: Exception) -> bool:
    """Classifies network drops, rate limits (HTTP 429), and upstream 5xx errors as retryable."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name or "ratelimit" in name


def retry_transient(operation, max_attempts: int):
    """Executes a callable with exponential backoff and jitter for transient failures."""
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if not is_transient_error(exc) or attempt == max_attempts:
                raise
            delay = min(4.0, 2 ** (attempt - 1)) + random.uniform(0, 0.25)
            LOGGER.warning(
                "Transient provider failure; retrying (attempt %s/%s, type=%s)",
                attempt,
                max_attempts,
                type(exc).__name__,
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
            timeout=timeout,
        )
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
            )
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query string with transient retry support."""
        return retry_transient(
            lambda: self.client.embed_query(text), self.max_retries
        )


class VectorStoreManager:
    """Manages local Chroma vector storage with versioned collection names."""

    def __init__(
        self,
        persist_dir: Path | None = None,
        embeddings: Embeddings | None = None,
    ):
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
        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=Config.COLLECTION_NAME,
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
        kwargs = search_kwargs or {"k": Config.RETRIEVAL_K}
        return self.vector_store.as_retriever(search_kwargs=kwargs)

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        """Performs vector similarity search returning top-k Document chunks."""
        return self.vector_store.similarity_search(
            query=query, k=k or Config.RETRIEVAL_K
        )

    def similarity_search_with_scores(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """Performs vector similarity search returning Document chunks and relevance scores."""
        return self.vector_store.similarity_search_with_relevance_scores(
            query=query, k=k or Config.RETRIEVAL_K
        )

    def get_document_count(self) -> int:
        """Returns the total number of indexed document chunks in the Chroma collection."""
        return int(self.vector_store._collection.count())

