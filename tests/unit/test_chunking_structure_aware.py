from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.ingestion.chunking.structure_aware import StructureAwareChunker


def test_structure_aware_validation() -> None:
    chunker = StructureAwareChunker(max_chars=700, overlap_chars=140)
    assert chunker.strategy_name == "structure_aware_v1"
    assert chunker.parameters() == {"max_chars": 700, "overlap_chars": 140}

    with pytest.raises(ValueError, match="max_chars must be positive"):
        StructureAwareChunker(max_chars=0, overlap_chars=0)

    with pytest.raises(ValueError, match="overlap_chars cannot be negative"):
        StructureAwareChunker(max_chars=500, overlap_chars=-1)

    with pytest.raises(ValueError, match="overlap_chars must be smaller than max_chars"):
        StructureAwareChunker(max_chars=500, overlap_chars=500)


def test_structure_aware_heading_stack_and_section_paths() -> None:
    text = (
        "# John Doe\n\n"
        "Contact: john@example.com\n\n"
        "## Professional Experience\n\n"
        "### Wirk\n\n"
        "Lead Developer (2021-2023)\n\n"
        "• Led backend team\n"
        "• Implemented RAG pipeline\n\n"
        "### Acmo Corp\n\n"
        "Senior Engineer (2018-2021)\n\n"
        "## Education\n\n"
        "BSc Computer Science\n"
    )
    doc = Document(page_content=text, metadata={"source": "cv.md", "page": 0})
    chunker = StructureAwareChunker(max_chars=700, overlap_chars=140)
    chunks = chunker.split([doc])

    section_paths = [c.metadata["section_path"] for c in chunks]
    assert "John Doe" in section_paths
    assert "John Doe > Professional Experience > Wirk" in section_paths
    assert "John Doe > Professional Experience > Acmo Corp" in section_paths
    assert "John Doe > Education" in section_paths


def test_structure_aware_oversized_paragraph_split() -> None:
    long_para = "Word " * 200  # 1000 chars > 700
    text = f"# Header\n\n{long_para}"
    doc = Document(page_content=text, metadata={"source": "long.md", "page": 0})
    chunker = StructureAwareChunker(max_chars=700, overlap_chars=140)
    chunks = chunker.split([doc])

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.content) <= 700
        assert chunk.metadata["section_path"] == "Header"


def test_structure_aware_oversized_list_split_between_items() -> None:
    # 10 items of 100 chars each = 1000 chars
    items = [f"• Item {i:02d}: " + ("x" * 85) for i in range(10)]
    text = "# Skills\n\n" + "\n".join(items)
    doc = Document(page_content=text, metadata={"source": "skills.md", "page": 0})
    chunker = StructureAwareChunker(max_chars=700, overlap_chars=140)
    chunks = chunker.split([doc])

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.content) <= 700
        assert chunk.metadata["section_path"] == "Skills"


def test_structure_aware_overlap_and_no_heading_alone_carry() -> None:
    text = (
        "# Section 1\n\n"
        "Short paragraph 1.\n\n"
        "Short paragraph 2.\n\n"
        "# Section 2\n\n"
        "Paragraph in section 2.\n"
    )
    doc = Document(page_content=text, metadata={"source": "sections.md", "page": 0})
    chunker = StructureAwareChunker(max_chars=700, overlap_chars=140)
    chunks = chunker.split([doc])

    # Section 1 and Section 2 must be separate chunks because heading transition closes section
    assert len(chunks) == 2
    assert chunks[0].metadata["section_path"] == "Section 1"
    assert chunks[1].metadata["section_path"] == "Section 2"
    # Section 2 chunk should NOT have Section 1 heading or text carried over
    assert "Section 1" not in chunks[1].content


def test_structure_aware_plain_text_without_headings() -> None:
    text = "Paragraph 1 text.\n\nParagraph 2 text.\n\n• Bullet 1\n• Bullet 2"
    doc = Document(page_content=text, metadata={"source": "plain.txt", "page": 0})
    chunker = StructureAwareChunker(max_chars=700, overlap_chars=140)
    chunks = chunker.split([doc])

    assert len(chunks) == 1
    assert chunks[0].metadata["section_path"] == ""
    assert chunks[0].metadata["chunk_strategy"] == "structure_aware_v1"
