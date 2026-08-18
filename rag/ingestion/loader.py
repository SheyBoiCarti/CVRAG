"""Loads source files into canonical chunk records without index-side effects."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.ingestion.manifest import SourceFileEntry
from rag.retrieval.base import ChunkRecord


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
EXCLUDED_FILENAMES = {"readme.md"}


class CanonicalDocumentLoader:
    """Builds an ordered corpus snapshot from files below one document root."""

    def __init__(self, documents_dir: Path, *, chunk_size: int, chunk_overlap: int) -> None:
        self.documents_dir = Path(documents_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def source_entries(self) -> tuple[SourceFileEntry, ...]:
        return tuple(
            SourceFileEntry(
                relative_path=path.relative_to(self.documents_dir).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size=path.stat().st_size,
            )
            for path in self._paths()
        )

    def load_and_chunk(self) -> list[ChunkRecord]:
        documents: list[Document] = []
        for path in self._paths():
            source = path.relative_to(self.documents_dir).as_posix()
            if path.suffix.lower() == ".pdf":
                documents.extend(self._pdf_documents(path, source))
            else:
                documents.append(Document(page_content=self._read_text(path), metadata={"source": source, "page": 0}))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n• ", "\n- ", "\n", " ", ""],
        )
        page_counters: dict[tuple[str, int], int] = {}
        records: list[ChunkRecord] = []
        for chunk in splitter.split_documents(documents):
            source = str(chunk.metadata["source"])
            page = int(chunk.metadata["page"])
            key = (source, page)
            chunk_index = page_counters.get(key, 0)
            page_counters[key] = chunk_index + 1
            content = chunk.page_content
            chunk_id = hashlib.sha256(f"{source}:{page}:{chunk_index}:{content}".encode("utf-8")).hexdigest()
            records.append(ChunkRecord(chunk_id, content, source, page, chunk_index, {}))
        return sorted(records, key=lambda chunk: (chunk.source, chunk.page, chunk.chunk_index, chunk.chunk_id))

    def _paths(self) -> list[Path]:
        if not self.documents_dir.exists():
            return []
        return sorted(
            (path for path in self.documents_dir.rglob("*") if self._is_ingestible(path)),
            key=lambda path: path.relative_to(self.documents_dir).as_posix().casefold(),
        )

    @staticmethod
    def _is_ingestible(path: Path) -> bool:
        return path.is_file() and not path.name.startswith(".") and path.name.casefold() not in EXCLUDED_FILENAMES and path.suffix.casefold() in SUPPORTED_EXTENSIONS

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
