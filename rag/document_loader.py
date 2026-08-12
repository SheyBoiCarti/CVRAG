import hashlib
from pathlib import Path

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import Config


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
EXCLUDED_FILENAMES = {"readme.md"}


def _is_ingestible(path: Path) -> bool:
    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.name.lower() not in EXCLUDED_FILENAMES
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


class CVDocumentLoader:
    """Load and chunk the fixed CV corpus."""

    def __init__(self, documents_dir: Path | None = None):
        self.documents_dir = Path(documents_dir or Config.DOCUMENTS_DIR)

    def load_documents(self) -> list[Document]:
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
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n• ", "\n- ", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        for chunk_index, chunk in enumerate(chunks):
            source = str(chunk.metadata.get("source", ""))
            page = int(chunk.metadata.get("page", 0))
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
        documents = self.load_documents()
        return self.split_documents(documents) if documents else []
