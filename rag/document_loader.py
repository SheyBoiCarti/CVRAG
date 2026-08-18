import hashlib
from pathlib import Path

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import Config


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
EXCLUDED_FILENAMES = {"readme.md"}


def _is_ingestible(path: Path) -> bool:
    """Filters candidate files to supported extensions while ignoring hidden and readme files."""
    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.name.lower() not in EXCLUDED_FILENAMES
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


class CVDocumentLoader:
    """Loads and chunks CV documents from the configured documents directory."""

    def __init__(
        self,
        documents_dir: Path | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        settings: Any | None = None,
    ):
        if settings is not None:
            self.documents_dir = Path(documents_dir or settings.DOCUMENTS_DIR)
            self.chunk_size = chunk_size or settings.CHUNK_SIZE
            self.chunk_overlap = (
                chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
            )
        else:
            self.documents_dir = Path(documents_dir or Config.DOCUMENTS_DIR)
            self.chunk_size = chunk_size or Config.CHUNK_SIZE
            self.chunk_overlap = (
                chunk_overlap if chunk_overlap is not None else Config.CHUNK_OVERLAP
            )

    def load_documents(self) -> list[Document]:
        """Reads all ingestible CV files from disk and returns Document objects."""
        if not self.documents_dir.exists():
            return []

        documents: list[Document] = []
        paths = sorted(self.documents_dir.iterdir(), key=lambda path: path.name.lower())
        for path in paths:
            if not _is_ingestible(path):
                continue
            if path.suffix.lower() == ".pdf":
                documents.extend(self._load_pdf(path))
            else:
                documents.append(self._load_text(path))
        return documents

    @staticmethod
    def _load_pdf(path: Path) -> list[Document]:
        """Extracts text page-by-page from a PDF document with 0-indexed page metadata."""
        reader = pypdf.PdfReader(str(path))
        documents = []
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "page": page_number},
                    )
                )
        return documents

    @staticmethod
    def _load_text(path: Path) -> Document:
        """Loads UTF-8 text or Markdown files with fallback encoding detection."""
        text = None
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                text = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = path.read_text(encoding="utf-8", errors="ignore")
        return Document(
            page_content=text,
            metadata={"source": str(path), "page": 0},
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Splits documents into overlapping chunks with bullet-aware separators and deterministic IDs."""
        # Include bullet markers in separators to avoid breaking resume lists mid-sentence
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n• ", "\n- ", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        for chunk_index, chunk in enumerate(chunks):
            source = str(chunk.metadata.get("source", ""))
            page = int(chunk.metadata.get("page", 0))
            # Generate deterministic hash for idempotent vector store indexing
            identity = (
                f"{source}:{page}:{chunk_index}:{chunk.page_content}".encode("utf-8")
            )
            chunk.metadata = {
                **chunk.metadata,
                "page": page,
                "chunk_index": chunk_index,
                "chunk_id": hashlib.sha256(identity).hexdigest(),
            }
        return chunks

    def load_and_split(self) -> list[Document]:
        """Loads and splits all available documents in a single step."""
        documents = self.load_documents()
        return self.split_documents(documents) if documents else []

