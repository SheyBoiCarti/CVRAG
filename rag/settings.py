from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

ALLOWED_CHUNK_STRATEGIES: set[str] = {
    "recursive_bullet_v1",
    "fixed_character_v1",
    "structure_aware_v1",
    "semantic_v1",
}


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
    CHUNK_STRATEGY: str = "recursive_bullet_v1"
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 140
    SEMANTIC_BREAK_THRESHOLD: float = 0.35
    SEMANTIC_MIN_CHARS: int = 200
    SEMANTIC_MAX_CHARS: int = 900
    SEMANTIC_OVERLAP_UNITS: int = 1
    STRUCTURE_MAX_CHARS: int = 700
    STRUCTURE_OVERLAP_CHARS: int = 140
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
    VERIFIER_ENABLED: bool = True
    VERIFIER_FAIL_CLOSED: bool = True
    ANSWERABILITY_CALIBRATION_FILE: str | None = None
    MAX_CORRECTION_ATTEMPTS: int = 1

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

        # Chunking strategy and parameter validation
        if self.CHUNK_STRATEGY not in ALLOWED_CHUNK_STRATEGIES:
            raise ValueError(
                f"Invalid CHUNK_STRATEGY '{self.CHUNK_STRATEGY}'. Must be one of {sorted(ALLOWED_CHUNK_STRATEGIES)}"
            )
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive.")
        if self.CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative.")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if not (0.0 <= self.SEMANTIC_BREAK_THRESHOLD <= 2.0):
            raise ValueError("SEMANTIC_BREAK_THRESHOLD must be between 0.0 and 2.0.")
        if self.SEMANTIC_MIN_CHARS <= 0:
            raise ValueError("SEMANTIC_MIN_CHARS must be positive.")
        if self.SEMANTIC_MAX_CHARS <= 0:
            raise ValueError("SEMANTIC_MAX_CHARS must be positive.")
        if self.SEMANTIC_MIN_CHARS >= self.SEMANTIC_MAX_CHARS:
            raise ValueError("SEMANTIC_MIN_CHARS must be strictly less than SEMANTIC_MAX_CHARS.")
        if self.SEMANTIC_OVERLAP_UNITS not in (0, 1):
            raise ValueError("SEMANTIC_OVERLAP_UNITS must be 0 or 1.")
        if self.STRUCTURE_MAX_CHARS <= 0:
            raise ValueError("STRUCTURE_MAX_CHARS must be positive.")
        if self.STRUCTURE_OVERLAP_CHARS < 0:
            raise ValueError("STRUCTURE_OVERLAP_CHARS cannot be negative.")
        if self.STRUCTURE_OVERLAP_CHARS >= self.STRUCTURE_MAX_CHARS:
            raise ValueError("STRUCTURE_OVERLAP_CHARS must be smaller than STRUCTURE_MAX_CHARS.")

        # Numerical and parameter bounds validation
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

    def chunk_parameters(self) -> dict[str, Any]:
        """Returns the effective strategy-specific chunk parameters."""
        if self.CHUNK_STRATEGY in ("recursive_bullet_v1", "fixed_character_v1"):
            return {
                "chunk_overlap": self.CHUNK_OVERLAP,
                "chunk_size": self.CHUNK_SIZE,
            }
        if self.CHUNK_STRATEGY == "structure_aware_v1":
            return {
                "max_chars": self.STRUCTURE_MAX_CHARS,
                "overlap_chars": self.STRUCTURE_OVERLAP_CHARS,
            }
        if self.CHUNK_STRATEGY == "semantic_v1":
            return {
                "break_threshold": self.SEMANTIC_BREAK_THRESHOLD,
                "embedding_model": self.EMBEDDING_MODEL,
                "max_chars": self.SEMANTIC_MAX_CHARS,
                "min_chars": self.SEMANTIC_MIN_CHARS,
                "overlap_units": self.SEMANTIC_OVERLAP_UNITS,
            }
        return {}

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
    if "CHUNK_STRATEGY" in env:
        kwargs["CHUNK_STRATEGY"] = env["CHUNK_STRATEGY"]
    if "CHUNK_SIZE" in env:
        kwargs["CHUNK_SIZE"] = int(env["CHUNK_SIZE"])
    if "CHUNK_OVERLAP" in env:
        kwargs["CHUNK_OVERLAP"] = int(env["CHUNK_OVERLAP"])
    if "SEMANTIC_BREAK_THRESHOLD" in env:
        kwargs["SEMANTIC_BREAK_THRESHOLD"] = float(env["SEMANTIC_BREAK_THRESHOLD"])
    if "SEMANTIC_MIN_CHARS" in env:
        kwargs["SEMANTIC_MIN_CHARS"] = int(env["SEMANTIC_MIN_CHARS"])
    if "SEMANTIC_MAX_CHARS" in env:
        kwargs["SEMANTIC_MAX_CHARS"] = int(env["SEMANTIC_MAX_CHARS"])
    if "SEMANTIC_OVERLAP_UNITS" in env:
        kwargs["SEMANTIC_OVERLAP_UNITS"] = int(env["SEMANTIC_OVERLAP_UNITS"])
    if "STRUCTURE_MAX_CHARS" in env:
        kwargs["STRUCTURE_MAX_CHARS"] = int(env["STRUCTURE_MAX_CHARS"])
    if "STRUCTURE_OVERLAP_CHARS" in env:
        kwargs["STRUCTURE_OVERLAP_CHARS"] = int(env["STRUCTURE_OVERLAP_CHARS"])
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
    if "VERIFIER_ENABLED" in env:
        kwargs["VERIFIER_ENABLED"] = _get_bool("VERIFIER_ENABLED", default=True)
    if "VERIFIER_FAIL_CLOSED" in env:
        kwargs["VERIFIER_FAIL_CLOSED"] = _get_bool("VERIFIER_FAIL_CLOSED", default=True)
    if "ANSWERABILITY_CALIBRATION_FILE" in env:
        kwargs["ANSWERABILITY_CALIBRATION_FILE"] = _get_str("ANSWERABILITY_CALIBRATION_FILE")
    if "MAX_CORRECTION_ATTEMPTS" in env:
        kwargs["MAX_CORRECTION_ATTEMPTS"] = int(env["MAX_CORRECTION_ATTEMPTS"])

    return Settings(**kwargs)  # type: ignore[arg-type]
