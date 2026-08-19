from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pytest

from rag.ingestion.indexer import IndexSynchronizer
from rag.ingestion.manifest import compute_index_fingerprint
from rag.settings import Settings
from tests.fakes import FakeEmbeddings


class FailingEmbeddings(FakeEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Simulated embedding provider failure during indexing")


def test_different_strategies_produce_different_index_fingerprints() -> None:
    corpus_fp = "corpus_abc123"
    schema_v = "v3"
    emb_model = "nvidia/nv-embedqa-e5-v5"

    fp_recursive = compute_index_fingerprint(
        corpus_fingerprint=corpus_fp,
        chunk_strategy="recursive_bullet_v1",
        chunk_parameters={"chunk_size": 700, "chunk_overlap": 140},
        embedding_model=emb_model,
        schema_version=schema_v,
    )

    fp_fixed = compute_index_fingerprint(
        corpus_fingerprint=corpus_fp,
        chunk_strategy="fixed_character_v1",
        chunk_parameters={"chunk_size": 700, "chunk_overlap": 140},
        embedding_model=emb_model,
        schema_version=schema_v,
    )

    fp_fixed_500 = compute_index_fingerprint(
        corpus_fingerprint=corpus_fp,
        chunk_strategy="fixed_character_v1",
        chunk_parameters={"chunk_size": 500, "chunk_overlap": 100},
        embedding_model=emb_model,
        schema_version=schema_v,
    )

    fp_structure = compute_index_fingerprint(
        corpus_fingerprint=corpus_fp,
        chunk_strategy="structure_aware_v1",
        chunk_parameters={"max_chars": 700, "overlap_chars": 140},
        embedding_model=emb_model,
        schema_version=schema_v,
    )

    fp_semantic = compute_index_fingerprint(
        corpus_fingerprint=corpus_fp,
        chunk_strategy="semantic_v1",
        chunk_parameters={
            "break_threshold": 0.35,
            "embedding_model": emb_model,
            "max_chars": 900,
            "min_chars": 200,
            "overlap_units": 1,
        },
        embedding_model=emb_model,
        schema_version=schema_v,
    )

    fingerprints = {fp_recursive, fp_fixed, fp_fixed_500, fp_structure, fp_semantic}
    assert len(fingerprints) == 5, "All strategies and configurations must yield distinct fingerprints"


def test_failed_semantic_build_preserves_active_index(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "sample.md").write_text("# Title\n\nSome paragraph text for testing.\n", encoding="utf-8")

    vector_db_dir = tmp_path / "vector_db"
    vector_db_dir.mkdir(parents=True)

    # 1. Build initial successful index with baseline recursive chunker
    base_settings = Settings(
        CHUNK_STRATEGY="recursive_bullet_v1",
        DOCUMENTS_DIR=docs_dir,
        VECTOR_DB_DIR=vector_db_dir,
    )
    embeddings = FakeEmbeddings()
    synchronizer = IndexSynchronizer(settings=base_settings, embeddings=embeddings)
    initial_index = synchronizer.ensure_active_index()
    assert initial_index is not None
    assert (vector_db_dir / "active-index.json").exists()

    initial_pointer_content = (vector_db_dir / "active-index.json").read_text(encoding="utf-8")

    # 2. Attempt to build semantic index with failing embeddings
    failing_settings = Settings(
        CHUNK_STRATEGY="semantic_v1",
        DOCUMENTS_DIR=docs_dir,
        VECTOR_DB_DIR=vector_db_dir,
    )
    failing_synchronizer = IndexSynchronizer(settings=failing_settings, embeddings=FailingEmbeddings())

    with pytest.raises(RuntimeError, match="Simulated embedding provider failure"):
        failing_synchronizer.ensure_active_index()

    # 3. Active index pointer must remain intact and valid
    current_pointer_content = (vector_db_dir / "active-index.json").read_text(encoding="utf-8")
    assert current_pointer_content == initial_pointer_content
