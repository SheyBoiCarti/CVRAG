from __future__ import annotations

import pytest

from rag.ingestion.chunking.factory import create_chunker
from rag.ingestion.chunking.fixed import FixedCharacterChunker
from rag.ingestion.chunking.recursive_baseline import RecursiveBulletChunker
from rag.ingestion.chunking.semantic import SemanticChunker
from rag.ingestion.chunking.structure_aware import StructureAwareChunker
from rag.settings import Settings
from tests.fakes import FakeEmbeddings


def test_factory_creates_recursive_bullet_chunker() -> None:
    settings = Settings(CHUNK_STRATEGY="recursive_bullet_v1", CHUNK_SIZE=600, CHUNK_OVERLAP=120)
    chunker = create_chunker(settings)
    assert isinstance(chunker, RecursiveBulletChunker)
    assert chunker.strategy_name == "recursive_bullet_v1"
    assert chunker.parameters() == {"chunk_size": 600, "chunk_overlap": 120}


def test_factory_creates_fixed_character_chunker() -> None:
    settings = Settings(CHUNK_STRATEGY="fixed_character_v1", CHUNK_SIZE=500, CHUNK_OVERLAP=100)
    chunker = create_chunker(settings)
    assert isinstance(chunker, FixedCharacterChunker)
    assert chunker.strategy_name == "fixed_character_v1"
    assert chunker.parameters() == {"chunk_size": 500, "chunk_overlap": 100}


def test_factory_creates_structure_aware_chunker() -> None:
    settings = Settings(CHUNK_STRATEGY="structure_aware_v1", STRUCTURE_MAX_CHARS=800, STRUCTURE_OVERLAP_CHARS=160)
    chunker = create_chunker(settings)
    assert isinstance(chunker, StructureAwareChunker)
    assert chunker.strategy_name == "structure_aware_v1"
    assert chunker.parameters() == {"max_chars": 800, "overlap_chars": 160}


def test_factory_creates_semantic_chunker_with_embeddings() -> None:
    embeddings = FakeEmbeddings()
    settings = Settings(
        CHUNK_STRATEGY="semantic_v1",
        SEMANTIC_BREAK_THRESHOLD=0.40,
        SEMANTIC_MIN_CHARS=150,
        SEMANTIC_MAX_CHARS=800,
        SEMANTIC_OVERLAP_UNITS=1,
    )
    chunker = create_chunker(settings, embeddings=embeddings)
    assert isinstance(chunker, SemanticChunker)
    assert chunker.strategy_name == "semantic_v1"
    assert chunker.parameters() == {
        "break_threshold": 0.40,
        "embedding_model": "nvidia/nv-embedqa-e5-v5",
        "max_chars": 800,
        "min_chars": 150,
        "overlap_units": 1,
    }


def test_factory_semantic_chunker_requires_embeddings() -> None:
    settings = Settings(CHUNK_STRATEGY="semantic_v1")
    with pytest.raises(ValueError, match="Embeddings instance required"):
        create_chunker(settings, embeddings=None)
