import pytest

from rag.config import Config


def test_validate_question_strips_valid_input():
    assert Config.validate_question("  What skills are listed?  ") == "What skills are listed?"


@pytest.mark.parametrize("question", ["", "   "])
def test_validate_question_rejects_blank_input(question):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        Config.validate_question(question)


def test_validate_question_rejects_oversized_input(monkeypatch):
    monkeypatch.setattr(Config, "MAX_QUESTION_CHARS", 10)
    with pytest.raises(ValueError, match="10 characters"):
        Config.validate_question("x" * 11)


def test_collection_name_changes_with_schema_or_embedding(monkeypatch):
    original = Config.COLLECTION_NAME
    monkeypatch.setattr(Config, "INDEX_SCHEMA_VERSION", "v99")
    assert Config.make_collection_name() != original


def test_config_exposes_hybrid_retrieval_settings(monkeypatch):
    """Legacy Config callers can opt into the same retrieval mode as Settings callers."""
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    assert Config.RETRIEVAL_MODE == "hybrid"
