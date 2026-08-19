"""Chunking strategies and interfaces for document ingestion."""

from __future__ import annotations

from rag.ingestion.chunking.base import Chunker, compute_chunk_id

__all__ = ["Chunker", "compute_chunk_id"]
