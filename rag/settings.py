from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent


def make_collection_name(schema_version: str, embedding_model: str) -> str:
    """Derives a deterministic Chroma collection name based on schema version and embedding model."""
    fingerprint = hashlib.sha256(
        f"{schema_version}:{embedding_model}".encode("utf-8")
    ).hexdigest()[:12]
    return f"cv_documents_{fingerprint}"


@dataclass(frozen=True)
class Settings:
    """Environment-first, immutable runtime configuration model."""

    NVIDIA_API_KEY: str | None = None
    LLM_MODEL: str = "meta/llama-3.3-70b-instruct"
    EMBEDDING_MODEL: str = "nvidia/nv-embedqa-e5-v5"
    DOCUMENTS_DIR: Path = field(default_factory=lambda: (BASE_DIR / "documents").resolve())
    VECTOR_DB_DIR: Path = field(default_factory=lambda: (BASE_DIR / "vector_db").resolve())
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 140
    RETRIEVAL_K: int = 6
    RETRIEVAL_MODE: str = "dense"
    DENSE_CANDIDATE_K: int = 20
    SPARSE_CANDIDATE_K: int = 20
    FUSION_CANDIDATE_K: int = 20
    CONTEXT_K: int | None = None
    RRF_CONSTANT: int = 60
    DENSE_RRF_WEIGHT: float = 1.0
    SPARSE_RRF_WEIGHT: float = 1.0
    RERANK_ENABLED: bool = False
    RERANK_MODEL: str = "nvidia/llama-nemotron-rerank-1b-v2"
    RERANK_CANDIDATE_K: int = 20
    MAX_QUESTION_CHARS: int = 1000
    MAX_HISTORY_TURNS: int = 4
    REQUEST_TIMEOUT: float = 45.0
    MAX_RETRIES: int = 3
    EMBEDDING_BATCH_SIZE: int = 15
    API_CORS_ORIGINS: tuple[str, ...] = ()
    API_CORS_ALLOW_CREDENTIALS: bool = False
    API_MAX_BODY_BYTES: int = 32768
    API_MAX_CONCURRENT_GENERATIONS: int = 4
    PIPELINE_VERSION: str = "dense-v1"
    INDEX_SCHEMA_VERSION: str = "v3"

    def __post_init__(self) -> None:
        # Normalize paths relative to BASE_DIR if not absolute
        doc_dir = Path(self.DOCUMENTS_DIR)
        if not doc_dir.is_absolute():
            doc_dir = (BASE_DIR / doc_dir).resolve()
        else:
            doc_dir = doc_dir.resolve()
        object.__setattr__(self, "DOCUMENTS_DIR", doc_dir)

        vec_dir = Path(self.VECTOR_DB_DIR)
        if not vec_dir.is_absolute():
            vec_dir = (BASE_DIR / vec_dir).resolve()
        else:
            vec_dir = vec_dir.resolve()
        object.__setattr__(self, "VECTOR_DB_DIR", vec_dir)
        if self.CONTEXT_K is None:
            object.__setattr__(self, "CONTEXT_K", self.RETRIEVAL_K)

        # Normalize and validate CORS origins
        if isinstance(self.API_CORS_ORIGINS, str):
            origins = tuple(o.strip() for o in self.API_CORS_ORIGINS.split(",") if o.strip())
        else:
            origins = tuple(str(o).strip() for o in self.API_CORS_ORIGINS if str(o).strip())
        for origin in origins:
            if "*" in origin:
                raise ValueError(f"Wildcard CORS origins are not allowed: '{origin}'")
        object.__setattr__(self, "API_CORS_ORIGINS", origins)

        # Numerical and parameter bounds validation
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive.")
        if self.CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative.")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if not (1 <= self.RETRIEVAL_K <= 12):
            raise ValueError("RETRIEVAL_K must be between 1 and 12.")
        if self.RETRIEVAL_MODE not in {"dense", "hybrid"}:
            raise ValueError("RETRIEVAL_MODE must be 'dense' or 'hybrid'.")
        if any(value <= 0 for value in (self.DENSE_CANDIDATE_K, self.SPARSE_CANDIDATE_K, self.FUSION_CANDIDATE_K, self.CONTEXT_K, self.RRF_CONSTANT, self.RERANK_CANDIDATE_K)):
            raise ValueError("Retrieval candidate limits and RRF_CONSTANT must be positive.")
        if self.DENSE_RRF_WEIGHT < 0 or self.SPARSE_RRF_WEIGHT < 0:
            raise ValueError("RRF weights must be nonnegative.")
        if self.RETRIEVAL_MODE == "dense" and self.CONTEXT_K > self.DENSE_CANDIDATE_K:
            raise ValueError("CONTEXT_K cannot exceed DENSE_CANDIDATE_K in dense mode.")
        if self.RETRIEVAL_MODE == "hybrid":
            if self.FUSION_CANDIDATE_K > self.DENSE_CANDIDATE_K + self.SPARSE_CANDIDATE_K:
                raise ValueError("FUSION_CANDIDATE_K cannot exceed combined candidate counts.")
            if self.RERANK_ENABLED and not (self.CONTEXT_K <= self.RERANK_CANDIDATE_K <= self.FUSION_CANDIDATE_K):
                raise ValueError("Hybrid reranking requires CONTEXT_K <= RERANK_CANDIDATE_K <= FUSION_CANDIDATE_K.")
        if self.RERANK_ENABLED and not self.RERANK_MODEL.strip():
            raise ValueError("RERANK_MODEL is required when reranking is enabled.")
        if self.MAX_QUESTION_CHARS <= 0:
            raise ValueError("MAX_QUESTION_CHARS must be positive.")
        if not (1 <= self.MAX_HISTORY_TURNS <= 20):
            raise ValueError("MAX_HISTORY_TURNS must be between 1 and 20.")
        if self.REQUEST_TIMEOUT <= 0:
            raise ValueError("REQUEST_TIMEOUT must be positive.")
        if self.MAX_RETRIES <= 0:
            raise ValueError("MAX_RETRIES must be positive.")
        if self.EMBEDDING_BATCH_SIZE <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive.")
        if self.API_MAX_BODY_BYTES <= 0:
            raise ValueError("API_MAX_BODY_BYTES must be positive.")
        if self.API_MAX_CONCURRENT_GENERATIONS <= 0:
            raise ValueError("API_MAX_CONCURRENT_GENERATIONS must be positive.")

    @property
    def COLLECTION_NAME(self) -> str:
        """Derives a deterministic Chroma collection name based on schema version and embedding model."""
        return make_collection_name(self.INDEX_SCHEMA_VERSION, self.EMBEDDING_MODEL)

    def validate_question(self, question: str) -> str:
        """Sanitizes and enforces length boundaries on user inputs."""
        normalized = question.strip()
        if not normalized:
            raise ValueError("Question cannot be empty.")
        if len(normalized) > self.MAX_QUESTION_CHARS:
            raise ValueError(
                f"Question must be at most {self.MAX_QUESTION_CHARS} characters."
            )
        return normalized

    def validate_runtime(self) -> None:
        """Validates critical runtime prerequisites (API key and directory existence)."""
        if not self.NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY is not configured.")
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


def get_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Loads and validates settings from environment variables with .env support."""
    load_dotenv(BASE_DIR / ".env")
    env = os.environ if environ is None else environ

    def _get_str(key: str, default: str | None = None) -> str | None:
        val = env.get(key)
        if val is not None and val.strip() != "":
            return val.strip()
        return default

    def _get_bool(key: str, default: bool = False) -> bool:
        val = env.get(key)
        if val is None:
            return default
        return val.strip().lower() in ("true", "1", "yes", "on")

    raw_cors = env.get("API_CORS_ORIGINS")
    if raw_cors:
        cors_origins = tuple(o.strip() for o in raw_cors.split(",") if o.strip())
    else:
        cors_origins = ()

    kwargs: dict[str, object] = {}

    api_key = _get_str("NVIDIA_API_KEY")
    if api_key is not None:
        kwargs["NVIDIA_API_KEY"] = api_key

    if "LLM_MODEL" in env:
        kwargs["LLM_MODEL"] = env["LLM_MODEL"]
    if "EMBEDDING_MODEL" in env:
        kwargs["EMBEDDING_MODEL"] = env["EMBEDDING_MODEL"]
    if "DOCUMENTS_DIR" in env:
        kwargs["DOCUMENTS_DIR"] = Path(env["DOCUMENTS_DIR"])
    if "VECTOR_DB_DIR" in env:
        kwargs["VECTOR_DB_DIR"] = Path(env["VECTOR_DB_DIR"])
    if "CHUNK_SIZE" in env:
        kwargs["CHUNK_SIZE"] = int(env["CHUNK_SIZE"])
    if "CHUNK_OVERLAP" in env:
        kwargs["CHUNK_OVERLAP"] = int(env["CHUNK_OVERLAP"])
    if "RETRIEVAL_K" in env:
        kwargs["RETRIEVAL_K"] = int(env["RETRIEVAL_K"])
    for key in ("RETRIEVAL_MODE", "RERANK_MODEL"):
        if key in env:
            kwargs[key] = env[key]
    for key in ("DENSE_CANDIDATE_K", "SPARSE_CANDIDATE_K", "FUSION_CANDIDATE_K", "CONTEXT_K", "RRF_CONSTANT", "RERANK_CANDIDATE_K"):
        if key in env:
            kwargs[key] = int(env[key])
    for key in ("DENSE_RRF_WEIGHT", "SPARSE_RRF_WEIGHT"):
        if key in env:
            kwargs[key] = float(env[key])
    if "RERANK_ENABLED" in env:
        kwargs["RERANK_ENABLED"] = _get_bool("RERANK_ENABLED")
    if "MAX_QUESTION_CHARS" in env:
        kwargs["MAX_QUESTION_CHARS"] = int(env["MAX_QUESTION_CHARS"])
    if "MAX_HISTORY_TURNS" in env:
        kwargs["MAX_HISTORY_TURNS"] = int(env["MAX_HISTORY_TURNS"])
    if "REQUEST_TIMEOUT" in env:
        kwargs["REQUEST_TIMEOUT"] = float(env["REQUEST_TIMEOUT"])
    if "MAX_RETRIES" in env:
        kwargs["MAX_RETRIES"] = int(env["MAX_RETRIES"])
    if "EMBEDDING_BATCH_SIZE" in env:
        kwargs["EMBEDDING_BATCH_SIZE"] = int(env["EMBEDDING_BATCH_SIZE"])
    if raw_cors is not None:
        kwargs["API_CORS_ORIGINS"] = cors_origins
    if "API_CORS_ALLOW_CREDENTIALS" in env:
        kwargs["API_CORS_ALLOW_CREDENTIALS"] = _get_bool("API_CORS_ALLOW_CREDENTIALS")
    if "API_MAX_BODY_BYTES" in env:
        kwargs["API_MAX_BODY_BYTES"] = int(env["API_MAX_BODY_BYTES"])
    if "API_MAX_CONCURRENT_GENERATIONS" in env:
        kwargs["API_MAX_CONCURRENT_GENERATIONS"] = int(env["API_MAX_CONCURRENT_GENERATIONS"])
    if "PIPELINE_VERSION" in env:
        kwargs["PIPELINE_VERSION"] = env["PIPELINE_VERSION"]
    if "INDEX_SCHEMA_VERSION" in env:
        kwargs["INDEX_SCHEMA_VERSION"] = env["INDEX_SCHEMA_VERSION"]

    return Settings(**kwargs)  # type: ignore[arg-type]
