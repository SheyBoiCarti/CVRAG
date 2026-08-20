"""Deterministic source and index identity models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class SourceFileEntry:
    relative_path: str
    sha256: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IndexManifest:
    schema_version: str
    index_id: str
    collection_name: str
    build_state: str
    corpus_fingerprint: str
    index_fingerprint: str
    embedding_model: str
    chunk_strategy: str
    chunk_parameters: Mapping[str, Any]
    sources: tuple[SourceFileEntry, ...]
    chunk_count: int
    chunk_ids_sha256: str
    chunks_snapshot_sha256: str
    created_by_version: str = "hybrid-v1"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = [source.to_dict() for source in self.sources]
        data["chunk_parameters"] = dict(self.chunk_parameters)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IndexManifest:
        return cls(
            schema_version=str(data["schema_version"]),
            index_id=str(data["index_id"]),
            collection_name=str(data["collection_name"]),
            build_state=str(data["build_state"]),
            corpus_fingerprint=str(data["corpus_fingerprint"]),
            index_fingerprint=str(data["index_fingerprint"]),
            embedding_model=str(data["embedding_model"]),
            chunk_strategy=str(data["chunk_strategy"]),
            chunk_parameters=dict(data["chunk_parameters"]),
            sources=tuple(SourceFileEntry(**entry) for entry in data["sources"]),
            chunk_count=int(data["chunk_count"]),
            chunk_ids_sha256=str(data["chunk_ids_sha256"]),
            chunks_snapshot_sha256=str(data["chunks_snapshot_sha256"]),
            created_by_version=str(data.get("created_by_version", "hybrid-v1")),
        )


@dataclass(frozen=True)
class ActiveIndexPointer:
    index_id: str
    collection_name: str
    manifest_sha256: str
    activated_at: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActiveIndexPointer:
        return cls(**{key: str(data[key]) for key in ("index_id", "collection_name", "manifest_sha256", "activated_at")})


def compute_corpus_fingerprint(sources: Iterable[SourceFileEntry]) -> str:
    """Hashes source-relative identities and content digests without absolute paths."""
    payload = [source.to_dict() for source in sorted(sources, key=lambda item: item.relative_path.casefold())]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def compute_index_fingerprint(
    *,
    corpus_fingerprint: str,
    chunk_strategy: str,
    chunk_parameters: Mapping[str, Any],
    embedding_model: str,
    schema_version: str,
) -> str:
    """Hashes corpus identity plus settings that affect index contents."""
    return hashlib.sha256(_canonical_json({
        "corpus_fingerprint": corpus_fingerprint,
        "chunk_strategy": chunk_strategy,
        "chunk_parameters": dict(chunk_parameters),
        "embedding_model": embedding_model,
        "schema_version": schema_version,
    })).hexdigest()


def manifest_sha256(manifest: IndexManifest) -> str:
    return hashlib.sha256(_canonical_json(manifest.to_dict())).hexdigest()


def dump_json(value: Mapping[str, Any]) -> str:
    return _canonical_json(value).decode("utf-8")
