"""Loads source files into canonical chunk records without index-side effects."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import pypdf
from langchain_core.documents import Document

from rag.ingestion.chunking.base import Chunker
from rag.ingestion.chunking.recursive_baseline import RecursiveBulletChunker
from rag.ingestion.manifest import SourceFileEntry
from rag.retrieval.base import ChunkRecord

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
EXCLUDED_FILENAMES = {"readme.md"}


class CanonicalDocumentLoader:
    """Builds an ordered corpus snapshot from files below one document root."""

    def __init__(
        self,
        documents_dir: Path,
        *,
        chunker: Chunker | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.documents_dir = Path(documents_dir)
        if chunker is not None:
            self.chunker = chunker
        else:
            c_size = chunk_size if chunk_size is not None else 700
            c_overlap = chunk_overlap if chunk_overlap is not None else 140
            self.chunker = RecursiveBulletChunker(chunk_size=c_size, chunk_overlap=c_overlap)

    @property
    def chunk_size(self) -> int:
        return int(self.chunker.parameters().get("chunk_size", 700))

    @property
    def chunk_overlap(self) -> int:
        return int(self.chunker.parameters().get("chunk_overlap", 140))

    def source_entries(self) -> tuple[SourceFileEntry, ...]:
        return tuple(
            SourceFileEntry(
                relative_path=path.relative_to(self.documents_dir).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size=path.stat().st_size,
            )
            for path in self._paths()
        )

    def load_documents(self) -> list[Document]:
        documents: list[Document] = []
        for path in self._paths():
            source = path.relative_to(self.documents_dir).as_posix()
            if path.suffix.lower() == ".pdf":
                documents.extend(self._pdf_documents(path, source))
            else:
                documents.append(
                    Document(
                        page_content=self._read_text(path),
                        metadata={"source": source, "page": 0},
                    )
                )
        return documents

    def load_and_chunk(self) -> list[ChunkRecord]:
        documents = self.load_documents()
        return self.chunker.split(documents)

    def _paths(self) -> list[Path]:
        if not self.documents_dir.exists():
            return []
        return sorted(
            (path for path in self.documents_dir.rglob("*") if self._is_ingestible(path)),
            key=lambda path: path.relative_to(self.documents_dir).as_posix().casefold(),
        )

    @staticmethod
    def _is_ingestible(path: Path) -> bool:
        return (
            path.is_file()
            and not path.name.startswith(".")
            and path.name.casefold() not in EXCLUDED_FILENAMES
            and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        )

    @staticmethod
    def _read_text(path: Path) -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                pass
        return path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _pdf_documents(path: Path, source: str) -> Iterable[Document]:
        for page, raw_page in enumerate(pypdf.PdfReader(str(path)).pages):
            content = raw_page.extract_text() or ""
            if content.strip():
                yield Document(page_content=content, metadata={"source": source, "page": page})
