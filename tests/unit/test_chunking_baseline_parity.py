from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.document_loader import CVDocumentLoader
from rag.ingestion.loader import CanonicalDocumentLoader


def test_baseline_loader_splits_custom_cv_deterministically() -> None:
    docs_dir = Path(__file__).resolve().parent.parent.parent / "documents"
    loader = CanonicalDocumentLoader(docs_dir, chunk_size=700, chunk_overlap=140)
    chunks = loader.load_and_chunk()
    assert len(chunks) > 0

    # Ensure all chunks have content and valid metadata
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id
        assert chunk.content
        assert chunk.source == "Custom_CV.md"
        assert chunk.page == 0
        assert chunk.chunk_index == i

    # Re-running loader produces identical chunks
    chunks_again = loader.load_and_chunk()
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in chunks_again]
    assert [c.content for c in chunks] == [c.content for c in chunks_again]


def test_recursive_character_text_splitter_exact_separators() -> None:
    sample_text = (
        "# Profile\n\n"
        "Experienced software engineer with expertise in RAG pipelines.\n\n"
        "## Skills\n\n"
        "• Python\n"
        "• FastAPI\n"
        "- LangChain\n"
        "- ChromaDB\n"
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=140,
        separators=["\n\n", "\n• ", "\n- ", "\n", " ", ""],
    )
    docs = [Document(page_content=sample_text, metadata={"source": "test.md", "page": 0})]
    split_docs = splitter.split_documents(docs)
    assert len(split_docs) == 1
    assert split_docs[0].page_content == sample_text.strip()
