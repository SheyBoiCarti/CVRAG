from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.ingestion.chunking.recursive_baseline import RecursiveBulletChunker


def test_recursive_bullet_chunker_initialization_and_validation() -> None:
    chunker = RecursiveBulletChunker(chunk_size=700, chunk_overlap=140)
    assert chunker.strategy_name == "recursive_bullet_v1"
    assert chunker.parameters() == {"chunk_size": 700, "chunk_overlap": 140}

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        RecursiveBulletChunker(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        RecursiveBulletChunker(chunk_size=500, chunk_overlap=-10)

    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
        RecursiveBulletChunker(chunk_size=500, chunk_overlap=500)


def test_recursive_bullet_chunker_split_and_metadata() -> None:
    chunker = RecursiveBulletChunker(chunk_size=700, chunk_overlap=140)
    doc1 = Document(
        page_content="# Experience\n\nSoftware Engineer at TechCorp.\n\n• Built RAG systems\n• Scaled backend",
        metadata={"source": "cv.md", "page": 0},
    )
    doc2 = Document(
        page_content="# Education\n\nBSc Computer Science",
        metadata={"source": "cv.md", "page": 1},
    )

    records = chunker.split([doc1, doc2])
    assert len(records) == 2

    r0 = records[0]
    assert r0.source == "cv.md"
    assert r0.page == 0
    assert r0.chunk_index == 0
    assert r0.metadata["chunk_strategy"] == "recursive_bullet_v1"
    assert r0.metadata["section_path"] == ""
    assert "Software Engineer at TechCorp" in r0.content

    r1 = records[1]
    assert r1.source == "cv.md"
    assert r1.page == 1
    assert r1.chunk_index == 0
    assert r1.metadata["chunk_strategy"] == "recursive_bullet_v1"
    assert r1.metadata["section_path"] == ""
    assert "BSc Computer Science" in r1.content


def test_recursive_bullet_chunker_deterministic_ids() -> None:
    chunker = RecursiveBulletChunker()
    doc = Document(
        page_content="A " * 500,
        metadata={"source": "doc.txt", "page": 0},
    )
    run1 = chunker.split([doc])
    run2 = chunker.split([doc])

    assert len(run1) > 1
    assert [c.chunk_id for c in run1] == [c.chunk_id for c in run2]
    assert [c.content for c in run1] == [c.content for c in run2]
