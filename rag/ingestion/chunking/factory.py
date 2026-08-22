"""Factory for instantiating document chunkers from application settings."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from rag.ingestion.chunking.base import Chunker
from rag.ingestion.chunking.fixed import FixedCharacterChunker
from rag.ingestion.chunking.recursive_baseline import RecursiveBulletChunker
from rag.ingestion.chunking.semantic import SemanticChunker
from rag.ingestion.chunking.structure_aware import StructureAwareChunker
from rag.settings import Settings


def create_chunker(settings: Settings, embeddings: Embeddings | None = None) -> Chunker:
    """Creates a configured Chunker instance based on settings and optional embeddings."""
    strategy = settings.CHUNK_STRATEGY

    if strategy == "recursive_bullet_v1":
        return RecursiveBulletChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    if strategy == "fixed_character_v1":
        return FixedCharacterChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    if strategy == "structure_aware_v1":
        return StructureAwareChunker(
            max_chars=settings.STRUCTURE_MAX_CHARS,
            overlap_chars=settings.STRUCTURE_OVERLAP_CHARS,
        )

    if strategy == "semantic_v1":
        if embeddings is None:
            raise ValueError("Embeddings instance required for semantic_v1 chunking strategy.")
        return SemanticChunker(
            embeddings=embeddings,
            break_threshold=settings.SEMANTIC_BREAK_THRESHOLD,
            min_chars=settings.SEMANTIC_MIN_CHARS,
            max_chars=settings.SEMANTIC_MAX_CHARS,
            overlap_units=settings.SEMANTIC_OVERLAP_UNITS,
            embedding_model=settings.EMBEDDING_MODEL,
        )

    raise ValueError(f"Unknown chunk strategy: '{strategy}'")
