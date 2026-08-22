from __future__ import annotations

from rag.ingestion.snapshots import (
    read_chunks_snapshot,
    validate_snapshot,
    write_chunks_snapshot,
)
from rag.retrieval.base import ChunkRecord


def test_snapshot_round_trip_detects_tampered_content(tmp_path) -> None:
    """Activation must reject a snapshot whose verified bytes have changed."""
    snapshot = tmp_path / "chunks.jsonl"
    chunks = [
        ChunkRecord("c2", "Second", "cv.md", 0, 1, {}),
        ChunkRecord("c1", "First", "cv.md", 0, 0, {"section": "skills"}),
    ]

    snapshot_sha, ids_sha = write_chunks_snapshot(chunks, snapshot)

    assert read_chunks_snapshot(snapshot) == [chunks[1], chunks[0]]
    assert validate_snapshot(snapshot, snapshot_sha, ids_sha, expected_count=2)

    snapshot.write_text(snapshot.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert not validate_snapshot(snapshot, snapshot_sha, ids_sha, expected_count=2)
