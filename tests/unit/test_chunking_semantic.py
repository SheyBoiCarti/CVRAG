from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.ingestion.chunking.semantic import SemanticChunker


class FakeDeterministicEmbeddings(Embeddings):
    """Fake embeddings returning predetermined vectors based on text content."""

    def __init__(self, mapping: dict[str, list[float]] | None = None, dim: int = 4) -> None:
        self.mapping = mapping or {}
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for t in texts:
            if t in self.mapping:
                results.append(self.mapping[t])
            else:
                # Deterministic pseudo-vector from text hash
                val = sum(ord(c) for c in t) % 100
                vec = [float((val + i) % 10) for i in range(self.dim)]
                results.append(vec)
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def test_semantic_chunker_validation() -> None:
    embeddings = FakeDeterministicEmbeddings()
    chunker = SemanticChunker(
        embeddings=embeddings,
        break_threshold=0.35,
        min_chars=200,
        max_chars=900,
        overlap_units=1,
    )
    assert chunker.strategy_name == "semantic_v1"
    assert chunker.parameters() == {
        "break_threshold": 0.35,
        "embedding_model": "nvidia/nv-embedqa-e5-v5",
        "max_chars": 900,
        "min_chars": 200,
        "overlap_units": 1,
    }

    with pytest.raises(ValueError, match="break_threshold"):
        SemanticChunker(embeddings=embeddings, break_threshold=-0.1)

    with pytest.raises(ValueError, match="break_threshold"):
        SemanticChunker(embeddings=embeddings, break_threshold=2.5)

    with pytest.raises(ValueError, match="min_chars must be positive"):
        SemanticChunker(embeddings=embeddings, min_chars=0)

    with pytest.raises(ValueError, match="min_chars must be strictly less than max_chars"):
        SemanticChunker(embeddings=embeddings, min_chars=900, max_chars=900)

    with pytest.raises(ValueError, match="overlap_units must be 0 or 1"):
        SemanticChunker(embeddings=embeddings, overlap_units=2)


def test_semantic_chunker_breaks_on_cosine_distance_threshold() -> None:
    # 3 paragraphs, each 250 chars (>= min_chars=200)
    # Unit 0 and Unit 1 are semantically similar (distance ~ 0)
    # Unit 1 and Unit 2 are semantically dissimilar (distance ~ 1.0 >= 0.35)
    u0 = "Unit zero content about machine learning and python engineering. " * 4  # 260 chars
    u1 = "Unit one content about machine learning and deep learning models. " * 4  # 264 chars
    u2 = "Unit two content about cooking Italian pasta and baking sourdough bread. " * 4  # 292 chars

    # Vectors: u0 and u1 parallel [1, 0], u2 orthogonal [0, 1]
    # Cosine dist(u0, u1) = 0.0 < 0.35
    # Cosine dist(u1, u2) = 1.0 >= 0.35
    fake_embeddings = FakeDeterministicEmbeddings(
        mapping={
            u0.strip(): [1.0, 0.0, 0.0, 0.0],
            u1.strip(): [1.0, 0.0, 0.0, 0.0],
            u2.strip(): [0.0, 1.0, 0.0, 0.0],
        }
    )

    chunker = SemanticChunker(
        embeddings=fake_embeddings,
        break_threshold=0.35,
        min_chars=200,
        max_chars=900,
        overlap_units=0,
    )

    doc = Document(page_content=f"{u0}\n\n{u1}\n\n{u2}", metadata={"source": "test.txt", "page": 0})
    chunks = chunker.split([doc])

    # Expect 2 chunks: Chunk 0 = [u0, u1], Chunk 1 = [u2]
    assert len(chunks) == 2
    assert u0.strip() in chunks[0].content and u1.strip() in chunks[0].content
    assert u2.strip() in chunks[1].content
    assert u0.strip() not in chunks[1].content
    assert chunks[0].metadata["chunk_strategy"] == "semantic_v1"
    assert chunks[0].metadata["section_path"] == ""


def test_semantic_chunker_with_overlap_unit_1() -> None:
    u0 = "A " * 110  # 220 chars
    u1 = "B " * 110  # 220 chars
    fake_embeddings = FakeDeterministicEmbeddings(
        mapping={
            u0.strip(): [1.0, 0.0],
            u1.strip(): [0.0, 1.0],
        },
        dim=2,
    )
    chunker = SemanticChunker(
        embeddings=fake_embeddings,
        break_threshold=0.35,
        min_chars=200,
        max_chars=900,
        overlap_units=1,
    )
    doc = Document(page_content=f"{u0}\n\n{u1}", metadata={"source": "test.txt", "page": 0})
    chunks = chunker.split([doc])

    # With distance >= 0.35 and overlap_units=1, chunk 0 is u0, chunk 1 carries u0 + u1
    assert len(chunks) == 2
    assert chunks[0].content == u0.strip()
    assert u0.strip() in chunks[1].content
    assert u1.strip() in chunks[1].content


def test_semantic_chunker_oversized_paragraph_sentence_split() -> None:
    # One long paragraph with sentences, total > max_chars 900
    s1 = "Sentence one is informative and detailed about systems. " * 7  # ~392 chars
    s2 = "Sentence two discusses database architecture and caching. " * 7  # ~406 chars
    s3 = "Sentence three covers deployment on Kubernetes clusters. " * 7  # ~399 chars
    long_para = f"{s1}{s2}{s3}"
    assert len(long_para) > 900

    fake_embeddings = FakeDeterministicEmbeddings()
    chunker = SemanticChunker(
        embeddings=fake_embeddings,
        break_threshold=0.35,
        min_chars=200,
        max_chars=900,
        overlap_units=0,
    )
    doc = Document(page_content=long_para, metadata={"source": "long.txt", "page": 0})
    chunks = chunker.split([doc])

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.content) <= 900
