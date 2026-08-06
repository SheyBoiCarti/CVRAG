import os
import re
import sys
import site
from pathlib import Path
from typing import List
from langchain_core.documents import Document

# Ensure user site packages are in sys.path
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    HAS_LANGCHAIN_COMMUNITY = True
except ImportError:
    HAS_LANGCHAIN_COMMUNITY = False

import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.config import Config

class CVDocumentLoader:
    """
    Handles loading, parsing, and tokenization/chunking of CV documents.
    Supports PDF, TXT, and Markdown (.md) formats.
    """

    def __init__(self, documents_dir: Path = None):
        """
        Initialize document loader with document directory path.
        """
        self.documents_dir = documents_dir or Config.DOCUMENTS_DIR

    def load_documents(self) -> List[Document]:
        """
        Scans the documents directory and parses all supported files into LangChain Documents.

        Returns:
            List[Document]: List of parsed raw documents with metadata.
        """
        documents: List[Document] = []

        if not self.documents_dir.exists():
            print(f"[Warning] Documents directory '{self.documents_dir}' does not exist.")
            return documents

        # Iterating through all files in the documents directory
        for file_path in self.documents_dir.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                try:
                    if ext == ".pdf":
                        if HAS_LANGCHAIN_COMMUNITY:
                            loader = PyPDFLoader(str(file_path))
                            docs = loader.load()
                            documents.extend(docs)
                            print(f"Loaded PDF (via PyPDFLoader): {file_path.name} ({len(docs)} pages)")
                        else:
                            # Fallback using native pypdf
                            reader = pypdf.PdfReader(str(file_path))
                            for page_num, page in enumerate(reader.pages):
                                text = page.extract_text()
                                if text:
                                    doc = Document(
                                        page_content=text,
                                        metadata={"source": str(file_path), "page": page_num}
                                    )
                                    documents.append(doc)
                            print(f"Loaded PDF (via native pypdf): {file_path.name} ({len(reader.pages)} pages)")

                    elif ext in [".txt", ".md"]:
                        if HAS_LANGCHAIN_COMMUNITY:
                            try:
                                loader = TextLoader(str(file_path), encoding="utf-8")
                                docs = loader.load()
                            except UnicodeDecodeError:
                                loader = TextLoader(str(file_path), encoding="latin-1")
                                docs = loader.load()
                            documents.extend(docs)
                            print(f"Loaded Text/MD: {file_path.name}")
                        else:
                            text = None
                            for enc in ["utf-8", "utf-8-sig", "latin-1"]:
                                try:
                                    with open(file_path, "r", encoding=enc) as f:
                                        text = f.read()
                                    break
                                except UnicodeDecodeError:
                                    continue
                            if text is None:
                                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                    text = f.read()
                            doc = Document(page_content=text, metadata={"source": str(file_path)})
                            documents.append(doc)
                            print(f"Loaded Text/MD (native): {file_path.name}")

                except Exception as e:
                    print(f"[Error] Failed to load file {file_path.name}: {e}")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Tokenization & Chunking Phase:
        Splits long document texts into smaller, semantically coherent chunks suitable for vector embedding.
        Uses generic regex patterns to detect section headers and role transitions in ANY resume layout.
        """
        processed_docs = []
        for doc in documents:
            content = doc.page_content
            
            # 1. Generic transition: Insert double newline when a bullet list ends and a title/header line begins
            content = re.sub(r'(\n[•\-\*\u2022]\s*.*?)\n([A-Z0-9][A-Za-z0-9\s,\-\|\(\)/]{2,60}(?:\n|\b))', r'\1\n\n\2', content)

            # 2. Generic section boundaries: Ensure major resume section headers have double newlines
            section_pattern = r'\n(?=(?:Experience|Work Experience|Employment|Education|Skills|Projects|Certifications|Summary|Professional Overview|Work History)\b)'
            content = re.sub(section_pattern, r'\n\n', content, flags=re.IGNORECASE)

            processed_docs.append(Document(page_content=content, metadata=doc.metadata))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n• ", "\n- ", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(processed_docs)
        print(f"Tokenization complete: Split {len(processed_docs)} raw document(s) into {len(chunks)} text chunks.")
        return chunks

    def load_and_split(self) -> List[Document]:
        """
        Convenience pipeline method: loads all documents and splits them into tokenized chunks.
        """
        raw_docs = self.load_documents()
        if not raw_docs:
            print("[Warning] No valid documents found in documents directory.")
            return []
        return self.split_documents(raw_docs)
