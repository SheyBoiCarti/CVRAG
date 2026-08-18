"""Canonical chunk snapshot serialization and integrity validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from rag.retrieval.base import ChunkRecord


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ordered(chunks: Sequence[ChunkRecord]) -> list[ChunkRecord]:
    return sorted(chunks, key=lambda chunk: (chunk.source, chunk.page, chunk.chunk_index, chunk.chunk_id))


def write_chunks_snapshot(chunks: Sequence[ChunkRecord], path: Path) -> tuple[str, str]:
    """Writes a sorted-key UTF-8 JSONL canonical snapshot and its digests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = _ordered(chunks)
    data = b"".join(
        json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for chunk in ordered
    )
    path.write_bytes(data)
    ids_digest = _digest("\n".join(sorted(chunk.chunk_id for chunk in ordered)).encode("utf-8"))
    return _digest(data), ids_digest


def read_chunks_snapshot(path: Path) -> list[ChunkRecord]:
    """Reads a canonical snapshot, rejecting malformed JSON or missing fields."""
    records: list[ChunkRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        item = json.loads(line)
        records.append(ChunkRecord(
            chunk_id=str(item["chunk_id"]), content=str(item["content"]), source=str(item["source"]),
            page=int(item["page"]), chunk_index=int(item["chunk_index"]), metadata=item.get("metadata", {}),
        ))
    return records


def validate_snapshot(path: Path, snapshot_sha256: str, chunk_ids_sha256: str, *, expected_count: int) -> bool:
    """Confirms a snapshot's bytes, IDs, ordering, and expected record count."""
    try:
        data = path.read_bytes()
        records = read_chunks_snapshot(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    if _digest(data) != snapshot_sha256 or len(records) != expected_count or records != _ordered(records):
        return False
    ids_digest = _digest("\n".join(sorted(record.chunk_id for record in records)).encode("utf-8"))
    return ids_digest == chunk_ids_sha256 and len({record.chunk_id for record in records}) == len(records)
