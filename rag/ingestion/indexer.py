"""Failure-safe construction and activation of immutable retrieval indexes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Sequence

from langchain_core.embeddings import Embeddings

from rag.ingestion.chunking.base import Chunker
from rag.ingestion.chunking.factory import create_chunker
from rag.ingestion.loader import CanonicalDocumentLoader
from rag.ingestion.manifest import ActiveIndexPointer, IndexManifest, compute_corpus_fingerprint, compute_index_fingerprint, dump_json, manifest_sha256
from rag.ingestion.snapshots import read_chunks_snapshot, validate_snapshot, write_chunks_snapshot
from rag.persistence.chroma_store import ChromaStoreAdapter
from rag.retrieval.base import ChunkRecord
from rag.settings import Settings


@dataclass(frozen=True)
class ActiveIndex:
    index_id: str
    collection_name: str
    index_fingerprint: str
    chunks: tuple[ChunkRecord, ...]


class IndexSynchronizer:
    """Synchronizes one canonical corpus snapshot to Chroma and an active pointer."""

    def __init__(self, *, settings: Settings, embeddings: Embeddings, chunker: Chunker | None = None) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.chunker = chunker or create_chunker(settings, embeddings=embeddings)
        self.loader = CanonicalDocumentLoader(settings.DOCUMENTS_DIR, chunker=self.chunker)
        self.indexes_dir = settings.VECTOR_DB_DIR / "indexes"
        self.pointer_path = settings.VECTOR_DB_DIR / "active-index.json"

    def ensure_active_index(self) -> ActiveIndex:
        """Reuses a validated matching build or atomically activates a replacement."""
        sources = self.loader.source_entries()
        corpus_fingerprint = compute_corpus_fingerprint(sources)
        chunk_strategy = self.chunker.strategy_name
        chunk_parameters = self.chunker.parameters()

        index_fingerprint = compute_index_fingerprint(
            corpus_fingerprint=corpus_fingerprint,
            chunk_strategy=chunk_strategy,
            chunk_parameters=chunk_parameters,
            embedding_model=self.settings.EMBEDDING_MODEL,
            schema_version=self.settings.INDEX_SCHEMA_VERSION,
        )
        current = self._validated_active_index(index_fingerprint)
        if current is not None:
            return current

        chunks = self.loader.load_and_chunk()
        index_id = f"idx_{index_fingerprint[:16]}"
        collection_name = f"cv_documents_{index_fingerprint[:16]}"
        build_dir = self.indexes_dir / index_id
        build_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = build_dir / "chunks.jsonl"
        snapshot_sha, ids_sha = write_chunks_snapshot(chunks, snapshot_path)
        building = IndexManifest(
            schema_version=self.settings.INDEX_SCHEMA_VERSION,
            index_id=index_id,
            collection_name=collection_name,
            build_state="building",
            corpus_fingerprint=corpus_fingerprint,
            index_fingerprint=index_fingerprint,
            embedding_model=self.settings.EMBEDDING_MODEL,
            chunk_strategy=chunk_strategy,
            chunk_parameters=chunk_parameters,
            sources=sources,
            chunk_count=len(chunks),
            chunk_ids_sha256=ids_sha,
            chunks_snapshot_sha256=snapshot_sha,
        )
        manifest_path = build_dir / "manifest.json"
        self._write_json(manifest_path, building.to_dict())
        store = ChromaStoreAdapter(persist_dir=self.settings.VECTOR_DB_DIR, collection_name=collection_name, embeddings=self.embeddings)
        store.add_chunks(chunks)
        if store.count() != len(chunks) or store.get_existing_ids([chunk.chunk_id for chunk in chunks]) != {chunk.chunk_id for chunk in chunks}:
            raise RuntimeError("Chroma collection validation failed; active index was not replaced.")
        complete = IndexManifest(**{**building.__dict__, "build_state": "complete"})
        self._write_json(manifest_path, complete.to_dict())
        pointer = ActiveIndexPointer(index_id, collection_name, manifest_sha256(complete), datetime.now(UTC).isoformat())
        self._write_json_atomically(self.pointer_path, pointer.to_dict())
        return ActiveIndex(index_id, collection_name, index_fingerprint, tuple(chunks))

    def _validated_active_index(self, expected_fingerprint: str) -> ActiveIndex | None:
        try:
            pointer = ActiveIndexPointer.from_dict(json.loads(self.pointer_path.read_text(encoding="utf-8")))
            manifest_path = self.indexes_dir / pointer.index_id / "manifest.json"
            manifest = IndexManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
            snapshot_path = manifest_path.parent / "chunks.jsonl"
            if pointer.manifest_sha256 != manifest_sha256(manifest) or manifest.build_state != "complete" or manifest.index_fingerprint != expected_fingerprint:
                return None
            if not validate_snapshot(snapshot_path, manifest.chunks_snapshot_sha256, manifest.chunk_ids_sha256, expected_count=manifest.chunk_count):
                return None
            chunks = read_chunks_snapshot(snapshot_path)
            store = ChromaStoreAdapter(persist_dir=self.settings.VECTOR_DB_DIR, collection_name=pointer.collection_name, embeddings=self.embeddings)
            sample = [chunk.chunk_id for chunk in chunks[: min(10, len(chunks))]]
            if store.count() != len(chunks) or store.get_existing_ids(sample) != set(sample):
                return None
            return ActiveIndex(manifest.index_id, manifest.collection_name, manifest.index_fingerprint, tuple(chunks))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump_json(value), encoding="utf-8")

    @classmethod
    def _write_json_atomically(cls, path: Path, value: dict) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        cls._write_json(temporary, value)
        os.replace(temporary, path)
