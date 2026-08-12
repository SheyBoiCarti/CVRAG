import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _configured_path(name: str, default: Path) -> Path:
    configured = Path(os.getenv(name, str(default)))
    if not configured.is_absolute():
        configured = BASE_DIR / configured
    return configured.resolve()


class Config:
    """Validated configuration for the fixed-CV demo."""

    BASE_DIR = BASE_DIR
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedcode-7b-v1")
    DOCUMENTS_DIR = _configured_path("DOCUMENTS_DIR", BASE_DIR / "documents")
    VECTOR_DB_DIR = _configured_path("VECTOR_DB_DIR", BASE_DIR / "vector_db")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "140"))
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))
    MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "1000"))
    MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "4"))

    REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "45"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "15"))
    INDEX_SCHEMA_VERSION = "v2"

    @classmethod
    def make_collection_name(cls) -> str:
        fingerprint = hashlib.sha256(
            f"{cls.INDEX_SCHEMA_VERSION}:{cls.EMBEDDING_MODEL}".encode("utf-8")
        ).hexdigest()[:12]
        return f"cv_documents_{fingerprint}"

    COLLECTION_NAME = ""

    @classmethod
    def validate_question(cls, question: str) -> str:
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
        if not cls.NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY is not configured.")
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if not 1 <= cls.RETRIEVAL_K <= 12:
            raise ValueError("RETRIEVAL_K must be between 1 and 12.")
        cls.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


Config.COLLECTION_NAME = Config.make_collection_name()
