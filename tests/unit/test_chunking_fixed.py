from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.ingestion.chunking.fixed import FixedCharacterChunker


def test_fixed_character_chunker_validation() -> None:
    chunker = FixedCharacterChunker(chunk_size=500, chunk_overlap=100)
    assert chunker.strategy_name == "fixed_character_v1"
    assert chunker.parameters() == {"chunk_overlap": 100, "chunk_size": 500}

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        FixedCharacterChunker(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        FixedCharacterChunker(chunk_size=500, chunk_overlap=-1)

    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
        FixedCharacterChunker(chunk_size=500, chunk_overlap=500)


def test_fixed_character_chunker_exact_windows() -> None:
    chunker = FixedCharacterChunker(chunk_size=10, chunk_overlap=3)
    text = "0123456789abcdef"  # len 16
    doc = Document(page_content=text, metadata={"source": "digits.txt", "page": 0})

    # step = 10 - 3 = 7
    # chunk 0: text[0:10] = "0123456789"
    # chunk 1: text[7:16] = "789abcdef" (len 9)
    chunks = chunker.split([doc])
    assert len(chunks) == 2
    assert chunks[0].content == "0123456789"
    assert chunks[1].content == "789abcdef"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].source == "digits.txt"
    assert chunks[0].page == 0
    assert chunks[0].metadata["chunk_strategy"] == "fixed_character_v1"
    assert chunks[0].metadata["section_path"] == ""


def test_fixed_character_preserves_whitespace_and_removes_empty() -> None:
    chunker = FixedCharacterChunker(chunk_size=10, chunk_overlap=2)
    doc_spaces = Document(page_content="   hello   ", metadata={"source": "spaces.txt", "page": 0})
    chunks = chunker.split([doc_spaces])
    assert len(chunks) == 2
    assert chunks[0].content == "   hello  "  # exact 10 chars with leading/trailing spaces
    assert chunks[1].content == "   "

    doc_empty = Document(page_content="", metadata={"source": "empty.txt", "page": 0})
    assert chunker.split([doc_empty]) == []


def test_fixed_character_preserves_page_boundaries() -> None:
    chunker = FixedCharacterChunker(chunk_size=10, chunk_overlap=2)
    doc1 = Document(page_content="Page1Content", metadata={"source": "multi.pdf", "page": 0})
    doc2 = Document(page_content="Page2Content", metadata={"source": "multi.pdf", "page": 1})

    chunks = chunker.split([doc1, doc2])
    assert len(chunks) == 4
    # page 0 chunks
    assert chunks[0].page == 0
    assert chunks[0].chunk_index == 0
    assert chunks[1].page == 0
    assert chunks[1].chunk_index == 1
    # page 1 chunks
    assert chunks[2].page == 1
    assert chunks[2].chunk_index == 0
    assert chunks[3].page == 1
    assert chunks[3].chunk_index == 1
