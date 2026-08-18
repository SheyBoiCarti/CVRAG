from pathlib import Path
from typing import ClassVar

from rag.settings import BASE_DIR, Settings, get_settings, make_collection_name


class Config:
    """Backward-compatible configuration bridge delegating to Settings."""

    BASE_DIR: Path = BASE_DIR
    _settings: ClassVar[Settings] = get_settings()

    NVIDIA_API_KEY: str | None = _settings.NVIDIA_API_KEY
    LLM_MODEL: str = _settings.LLM_MODEL
    EMBEDDING_MODEL: str = _settings.EMBEDDING_MODEL
    DOCUMENTS_DIR: Path = _settings.DOCUMENTS_DIR
    VECTOR_DB_DIR: Path = _settings.VECTOR_DB_DIR
    CHUNK_SIZE: int = _settings.CHUNK_SIZE
    CHUNK_OVERLAP: int = _settings.CHUNK_OVERLAP
    RETRIEVAL_K: int = _settings.RETRIEVAL_K
    MAX_QUESTION_CHARS: int = _settings.MAX_QUESTION_CHARS
    MAX_HISTORY_TURNS: int = _settings.MAX_HISTORY_TURNS
    REQUEST_TIMEOUT: float = _settings.REQUEST_TIMEOUT
    MAX_RETRIES: int = _settings.MAX_RETRIES
    EMBEDDING_BATCH_SIZE: int = _settings.EMBEDDING_BATCH_SIZE
    INDEX_SCHEMA_VERSION: str = _settings.INDEX_SCHEMA_VERSION
    COLLECTION_NAME: str = ""

    @classmethod
    def make_collection_name(cls) -> str:
        """Derives a deterministic Chroma collection name based on schema version and embedding model."""
        return make_collection_name(cls.INDEX_SCHEMA_VERSION, cls.EMBEDDING_MODEL)

    @classmethod
    def validate_question(cls, question: str) -> str:
        """Sanitizes and enforces length boundaries on user inputs."""
        normalized = question.strip()
        if not normalized:
            raise ValueError("Question cannot be empty.")
        if len(normalized) > cls.MAX_QUESTION_CHARS:
            raise ValueError(
                f"Question must be at most {cls.MAX_QUESTION_CHARS} characters."
            )
        return normalized

    @classmethod
    def validate(cls) -> None:
        """Validates critical runtime configuration before executing retrieval or generation."""
        if not cls.NVIDIA_API_KEY:
            cls.NVIDIA_API_KEY = get_settings().NVIDIA_API_KEY
        if not cls.NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY is not configured.")
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if not 1 <= cls.RETRIEVAL_K <= 12:
            raise ValueError("RETRIEVAL_K must be between 1 and 12.")
        cls.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


Config.COLLECTION_NAME = Config.make_collection_name()
