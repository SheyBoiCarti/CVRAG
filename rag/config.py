from pathlib import Path
from typing import Any

from rag.settings import BASE_DIR, Settings, get_settings, make_collection_name


class _ConfigMeta(type):
    """Metaclass that provides lazy, environment-backed attributes for Config."""

    _overrides: dict[str, Any] = {}

    def _get_settings(cls) -> Settings:
        try:
            return get_settings()
        except Exception:
            return Settings()

    def __getattr__(cls, name: str) -> Any:
        if name in cls._overrides:
            return cls._overrides[name]
        raise AttributeError(f"type object 'Config' has no attribute '{name}'")

    def __setattr__(cls, name: str, value: Any) -> None:
        if name in ("_overrides", "__doc__", "__module__"):
            super().__setattr__(name, value)
        else:
            cls._overrides[name] = value

    def __delattr__(cls, name: str) -> None:
        if name in ("_overrides", "__doc__", "__module__"):
            super().__delattr__(name)
        else:
            cls._overrides.pop(name, None)

    @property
    def BASE_DIR(cls) -> Path:
        return cls._overrides.get("BASE_DIR", BASE_DIR)

    @property
    def NVIDIA_API_KEY(cls) -> str | None:
        return cls._overrides.get("NVIDIA_API_KEY", cls._get_settings().NVIDIA_API_KEY)

    @property
    def LLM_MODEL(cls) -> str:
        return cls._overrides.get("LLM_MODEL", cls._get_settings().LLM_MODEL)

    @property
    def EMBEDDING_MODEL(cls) -> str:
        return cls._overrides.get("EMBEDDING_MODEL", cls._get_settings().EMBEDDING_MODEL)

    @property
    def DOCUMENTS_DIR(cls) -> Path:
        return cls._overrides.get("DOCUMENTS_DIR", cls._get_settings().DOCUMENTS_DIR)

    @property
    def VECTOR_DB_DIR(cls) -> Path:
        return cls._overrides.get("VECTOR_DB_DIR", cls._get_settings().VECTOR_DB_DIR)

    @property
    def CHUNK_SIZE(cls) -> int:
        return cls._overrides.get("CHUNK_SIZE", cls._get_settings().CHUNK_SIZE)

    @property
    def CHUNK_OVERLAP(cls) -> int:
        return cls._overrides.get("CHUNK_OVERLAP", cls._get_settings().CHUNK_OVERLAP)

    @property
    def RETRIEVAL_K(cls) -> int:
        return cls._overrides.get("RETRIEVAL_K", cls._get_settings().RETRIEVAL_K)

    @property
    def MAX_QUESTION_CHARS(cls) -> int:
        return cls._overrides.get("MAX_QUESTION_CHARS", cls._get_settings().MAX_QUESTION_CHARS)

    @property
    def MAX_HISTORY_TURNS(cls) -> int:
        return cls._overrides.get("MAX_HISTORY_TURNS", cls._get_settings().MAX_HISTORY_TURNS)

    @property
    def REQUEST_TIMEOUT(cls) -> float:
        return cls._overrides.get("REQUEST_TIMEOUT", cls._get_settings().REQUEST_TIMEOUT)

    @property
    def MAX_RETRIES(cls) -> int:
        return cls._overrides.get("MAX_RETRIES", cls._get_settings().MAX_RETRIES)

    @property
    def EMBEDDING_BATCH_SIZE(cls) -> int:
        return cls._overrides.get("EMBEDDING_BATCH_SIZE", cls._get_settings().EMBEDDING_BATCH_SIZE)

    @property
    def INDEX_SCHEMA_VERSION(cls) -> str:
        return cls._overrides.get("INDEX_SCHEMA_VERSION", cls._get_settings().INDEX_SCHEMA_VERSION)

    @property
    def COLLECTION_NAME(cls) -> str:
        return cls._overrides.get("COLLECTION_NAME", cls.make_collection_name())


class Config(metaclass=_ConfigMeta):
    """Backward-compatible configuration bridge delegating lazily to Settings."""

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
        settings = get_settings()
        if not settings.NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY is not configured.")
        if settings.CHUNK_OVERLAP >= settings.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if not 1 <= settings.RETRIEVAL_K <= 12:
            raise ValueError("RETRIEVAL_K must be between 1 and 12.")
        settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
