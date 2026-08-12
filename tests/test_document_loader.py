from pathlib import Path

from rag.document_loader import CVDocumentLoader


def test_loader_excludes_readme_hidden_and_nested_files(tmp_path: Path):
    (tmp_path / "README.md").write_text("Do not index me", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("Do not index me", encoding="utf-8")
    (tmp_path / "cv.txt").write_text(
        "EXPERIENCE\nEngineer at Example", encoding="utf-8"
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "other.txt").write_text("Do not index me", encoding="utf-8")

    docs = CVDocumentLoader(tmp_path).load_documents()

    assert [Path(doc.metadata["source"]).name for doc in docs] == ["cv.txt"]


def test_chunks_have_stable_source_page_index_and_id(tmp_path: Path):
    (tmp_path / "cv.txt").write_text("Skills\nPython and SQL", encoding="utf-8")
    loader = CVDocumentLoader(tmp_path)

    first = loader.load_and_split()
    second = loader.load_and_split()

    assert first[0].metadata["page"] == 0
    assert first[0].metadata["chunk_index"] == 0
    assert len(first[0].metadata["chunk_id"]) == 64
    assert first[0].metadata["chunk_id"] == second[0].metadata["chunk_id"]
