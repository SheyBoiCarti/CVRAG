from __future__ import annotations

from rag.ingestion.manifest import SourceFileEntry, compute_corpus_fingerprint, compute_index_fingerprint


def test_fingerprints_depend_on_normalized_source_identity_and_index_settings() -> None:
    """Moving a corpus does not rebuild it, while changing its bytes or settings does."""
    source = SourceFileEntry(relative_path="CV.md", sha256="a" * 64, size=42)
    corpus = compute_corpus_fingerprint([source])

    assert corpus == compute_corpus_fingerprint([source])
    assert corpus != compute_corpus_fingerprint(
        [SourceFileEntry(relative_path="CV.md", sha256="b" * 64, size=42)]
    )
    assert compute_index_fingerprint(
        corpus_fingerprint=corpus,
        chunk_strategy="recursive_bullet_v1",
        chunk_parameters={"chunk_overlap": 140, "chunk_size": 700},
        embedding_model="nvidia/embed",
        schema_version="v4",
    ) != compute_index_fingerprint(
        corpus_fingerprint=corpus,
        chunk_strategy="recursive_bullet_v1",
        chunk_parameters={"chunk_overlap": 140, "chunk_size": 800},
        embedding_model="nvidia/embed",
        schema_version="v4",
    )
