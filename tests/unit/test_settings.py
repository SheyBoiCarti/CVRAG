import sys
from pathlib import Path

import pytest

from rag.settings import BASE_DIR, Settings, get_settings, make_collection_name


def test_default_settings():
    settings = Settings()
    assert settings.NVIDIA_API_KEY is None
    assert settings.LLM_MODEL == "meta/llama-3.3-70b-instruct"
    assert settings.EMBEDDING_MODEL == "nvidia/nv-embedqa-e5-v5"
    assert settings.DOCUMENTS_DIR == (BASE_DIR / "documents").resolve()
    assert settings.VECTOR_DB_DIR == (BASE_DIR / "vector_db").resolve()
    assert settings.CHUNK_SIZE == 700
    assert settings.CHUNK_OVERLAP == 140
    assert settings.RETRIEVAL_K == 6
    assert settings.MAX_QUESTION_CHARS == 1000
    assert settings.MAX_HISTORY_TURNS == 4
    assert settings.REQUEST_TIMEOUT == 45.0
    assert settings.MAX_RETRIES == 3
    assert settings.EMBEDDING_BATCH_SIZE == 15
    assert settings.API_CORS_ORIGINS == ()
    assert settings.API_CORS_ALLOW_CREDENTIALS is False
    assert settings.API_MAX_BODY_BYTES == 32768
    assert settings.API_MAX_CONCURRENT_GENERATIONS == 4
    assert settings.PIPELINE_VERSION == "dense-v1"
    assert settings.INDEX_SCHEMA_VERSION == "v3"
    assert settings.COLLECTION_NAME.startswith("cv_documents_")


def test_settings_frozen():
    settings = Settings()
    with pytest.raises(Exception):  # FrozenInstanceError / dataclass frozen
        settings.CHUNK_SIZE = 500  # type: ignore


def test_get_settings_env_overrides(monkeypatch):
    test_env = {
        "NVIDIA_API_KEY": "nvapi-testkey-12345",
        "LLM_MODEL": "custom/llm-model",
        "EMBEDDING_MODEL": "custom/embed-model",
        "DOCUMENTS_DIR": "./custom_docs",
        "VECTOR_DB_DIR": "./custom_vec",
        "CHUNK_SIZE": "800",
        "CHUNK_OVERLAP": "100",
        "RETRIEVAL_K": "8",
        "MAX_QUESTION_CHARS": "500",
        "MAX_HISTORY_TURNS": "6",
        "REQUEST_TIMEOUT": "30.0",
        "MAX_RETRIES": "5",
        "EMBEDDING_BATCH_SIZE": "20",
        "API_CORS_ORIGINS": "https://example.com, http://localhost:3000",
        "API_CORS_ALLOW_CREDENTIALS": "true",
        "API_MAX_BODY_BYTES": "65536",
        "API_MAX_CONCURRENT_GENERATIONS": "8",
        "PIPELINE_VERSION": "hybrid-v2",
        "INDEX_SCHEMA_VERSION": "v4",
    }
    for k, v in test_env.items():
        monkeypatch.setenv(k, v)

    settings = get_settings()
    assert settings.NVIDIA_API_KEY == "nvapi-testkey-12345"
    assert settings.LLM_MODEL == "custom/llm-model"
    assert settings.EMBEDDING_MODEL == "custom/embed-model"
    assert settings.DOCUMENTS_DIR == (BASE_DIR / "custom_docs").resolve()
    assert settings.VECTOR_DB_DIR == (BASE_DIR / "custom_vec").resolve()
    assert settings.CHUNK_SIZE == 800
    assert settings.CHUNK_OVERLAP == 100
    assert settings.RETRIEVAL_K == 8
    assert settings.MAX_QUESTION_CHARS == 500
    assert settings.MAX_HISTORY_TURNS == 6
    assert settings.REQUEST_TIMEOUT == 30.0
    assert settings.MAX_RETRIES == 5
    assert settings.EMBEDDING_BATCH_SIZE == 20
    assert settings.API_CORS_ORIGINS == ("https://example.com", "http://localhost:3000")
    assert settings.API_CORS_ALLOW_CREDENTIALS is True
    assert settings.API_MAX_BODY_BYTES == 65536
    assert settings.API_MAX_CONCURRENT_GENERATIONS == 8
    assert settings.PIPELINE_VERSION == "hybrid-v2"
    assert settings.INDEX_SCHEMA_VERSION == "v4"


def test_path_normalization_relative_and_absolute(tmp_path):
    rel_settings = Settings(DOCUMENTS_DIR=Path("relative/docs"), VECTOR_DB_DIR=Path("./relative/vec"))
    assert rel_settings.DOCUMENTS_DIR == (BASE_DIR / "relative/docs").resolve()
    assert rel_settings.VECTOR_DB_DIR == (BASE_DIR / "relative/vec").resolve()

    abs_docs = (tmp_path / "abs_docs").resolve()
    abs_vec = (tmp_path / "abs_vec").resolve()
    abs_settings = Settings(DOCUMENTS_DIR=abs_docs, VECTOR_DB_DIR=abs_vec)
    assert abs_settings.DOCUMENTS_DIR == abs_docs
    assert abs_settings.VECTOR_DB_DIR == abs_vec


@pytest.mark.parametrize(
    "kwargs, expected_error",
    [
        ({"CHUNK_OVERLAP": 700, "CHUNK_SIZE": 700}, "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"),
        ({"CHUNK_OVERLAP": 800, "CHUNK_SIZE": 700}, "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"),
        ({"CHUNK_SIZE": 0}, "CHUNK_SIZE must be positive"),
        ({"CHUNK_SIZE": -10}, "CHUNK_SIZE must be positive"),
        ({"CHUNK_OVERLAP": -1}, "CHUNK_OVERLAP cannot be negative"),
        ({"RETRIEVAL_K": 0}, "RETRIEVAL_K must be between 1 and 12"),
        ({"RETRIEVAL_K": 13}, "RETRIEVAL_K must be between 1 and 12"),
        ({"MAX_QUESTION_CHARS": 0}, "MAX_QUESTION_CHARS must be positive"),
        ({"MAX_QUESTION_CHARS": -5}, "MAX_QUESTION_CHARS must be positive"),
        ({"MAX_HISTORY_TURNS": 0}, "MAX_HISTORY_TURNS must be between 1 and 20"),
        ({"MAX_HISTORY_TURNS": 21}, "MAX_HISTORY_TURNS must be between 1 and 20"),
        ({"REQUEST_TIMEOUT": 0.0}, "REQUEST_TIMEOUT must be positive"),
        ({"REQUEST_TIMEOUT": -1.0}, "REQUEST_TIMEOUT must be positive"),
        ({"MAX_RETRIES": 0}, "MAX_RETRIES must be positive"),
        ({"MAX_RETRIES": -1}, "MAX_RETRIES must be positive"),
        ({"EMBEDDING_BATCH_SIZE": 0}, "EMBEDDING_BATCH_SIZE must be positive"),
        ({"API_MAX_BODY_BYTES": 0}, "API_MAX_BODY_BYTES must be positive"),
        ({"API_MAX_BODY_BYTES": -100}, "API_MAX_BODY_BYTES must be positive"),
        ({"API_MAX_CONCURRENT_GENERATIONS": 0}, "API_MAX_CONCURRENT_GENERATIONS must be positive"),
        ({"API_MAX_CONCURRENT_GENERATIONS": -1}, "API_MAX_CONCURRENT_GENERATIONS must be positive"),
    ],
)
def test_validation_errors(kwargs, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        Settings(**kwargs)


@pytest.mark.parametrize(
    "invalid_cors",
    [
        ("*",),
        ("http://*",),
        ("https://*.example.com",),
        ("https://example.com", "*"),
    ],
)
def test_cors_origins_rejects_wildcards(invalid_cors):
    with pytest.raises(ValueError, match="Wildcard CORS origins are not allowed"):
        Settings(API_CORS_ORIGINS=invalid_cors)


def test_cors_origins_parsing_from_env(monkeypatch):
    monkeypatch.setenv("API_CORS_ORIGINS", "  https://foo.com , https://bar.com  ,  ")
    settings = get_settings()
    assert settings.API_CORS_ORIGINS == ("https://foo.com", "https://bar.com")


def test_make_collection_name_deterministic():
    name1 = make_collection_name("v3", "nvidia/nv-embedqa-e5-v5")
    name2 = make_collection_name("v3", "nvidia/nv-embedqa-e5-v5")
    name3 = make_collection_name("v4", "nvidia/nv-embedqa-e5-v5")
    name4 = make_collection_name("v3", "other/model")

    assert name1 == name2
    assert name1.startswith("cv_documents_")
    assert name1 != name3
    assert name1 != name4


def test_validate_question_method():
    settings = Settings(MAX_QUESTION_CHARS=50)
    assert settings.validate_question("  Hello world  ") == "Hello world"

    with pytest.raises(ValueError, match="Question cannot be empty"):
        settings.validate_question("")

    with pytest.raises(ValueError, match="Question cannot be empty"):
        settings.validate_question("   ")

    with pytest.raises(ValueError, match="at most 50 characters"):
        settings.validate_question("a" * 51)


def test_validate_runtime(tmp_path):
    settings_no_key = Settings(
        NVIDIA_API_KEY=None,
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vec",
    )
    with pytest.raises(ValueError, match="NVIDIA_API_KEY is not configured"):
        settings_no_key.validate_runtime()

    settings_with_key = Settings(
        NVIDIA_API_KEY="test-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vec",
    )
    settings_with_key.validate_runtime()
    assert (tmp_path / "docs").exists()
    assert (tmp_path / "vec").exists()


def test_no_streamlit_imported_when_importing_settings_and_config():
    # Ensure streamlit is not in sys.modules from rag.settings or rag.config
    sys.modules.pop("streamlit", None)
    sys.modules.pop("rag.settings", None)
    sys.modules.pop("rag.config", None)


    assert "streamlit" not in sys.modules
