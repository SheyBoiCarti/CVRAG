from __future__ import annotations

import warnings

from rag.persistence.chroma_store import ChromaStoreAdapter
from rag.retrieval.base import ChunkRecord
from tests.fakes import FakeEmbeddings


def test_chroma_adapter_indexes_canonical_chunks_idempotently(tmp_path) -> None:
    """Replaying a snapshot keeps one document per canonical chunk ID."""
    store = ChromaStoreAdapter(
        persist_dir=tmp_path / "chroma",
        collection_name="phase_two_test",
        embeddings=FakeEmbeddings(dimension=8),
    )
    chunk = ChunkRecord(
        chunk_id="chunk-python",
        content="Python backend engineer",
        source="Custom_CV.md",
        page=0,
        chunk_index=0,
        metadata={"section": "skills"},
    )

    assert store.add_chunks([chunk]) == 1
    assert store.add_chunks([chunk]) == 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        found, score = store.similarity_search_with_scores("Python backend", k=1)[0]
    assert found == chunk
    assert isinstance(score, float)
    assert store.get_existing_ids(["missing", "chunk-python"]) == {"chunk-python"}
