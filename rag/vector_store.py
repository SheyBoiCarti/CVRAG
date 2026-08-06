import os
import sys
import site
import time
from pathlib import Path
from typing import List, Tuple

user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from rag.config import Config


class RobustNVIDIAEmbeddings(Embeddings):
    """
    Wrapper around NVIDIAEmbeddings that adds:
    1. Automatic chunk batching to prevent API ReadTimeout on large document collections.
    2. Exponential backoff retry logic for transient network/API timeouts.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        batch_size: int = 15,
        max_retries: int = 5,
        timeout: float = 120.0
    ):
        self.client = NVIDIAEmbeddings(
            model=model,
            api_key=api_key,
            truncate="NONE",
            timeout=timeout
        )
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

    def _execute_with_retry(self, func, *args, **kwargs):
        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                delay = 2 ** attempt
                print(f"[NVIDIA Embeddings] Network attempt {attempt}/{self.max_retries} failed ({e}). Retrying in {delay}s...")
                time.sleep(delay)
        raise last_exception

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._execute_with_retry(self.client.embed_documents, batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._execute_with_retry(self.client.embed_query, text)


class VectorStoreManager:
    """
    Manages Vector Database operations using ChromaDB and NVIDIA Embeddings.
    Handles embedding generation, persistent vector storage, indexing, and similarity search.
    """

    def __init__(self, persist_dir: Path = None):
        """
        Initializes NVIDIA Embeddings client and persistent Chroma Vector DB.
        """
        Config.validate()
        self.persist_dir = str(persist_dir or Config.VECTOR_DB_DIR)

        # Initialize Resilient NVIDIA Embeddings API Client with Batching & Network Retries
        self.embeddings = RobustNVIDIAEmbeddings(
            model=Config.EMBEDDING_MODEL,
            api_key=Config.NVIDIA_API_KEY,
            batch_size=Config.EMBEDDING_BATCH_SIZE,
            max_retries=Config.MAX_RETRIES,
            timeout=Config.REQUEST_TIMEOUT
        )

        # Initialize or load existing Chroma vector database instance
        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name="cv_documents"
        )

    def clear_vector_store(self):
        """
        Clears existing collection in ChromaDB to prevent duplicate vector accumulation.
        """
        try:
            self.vector_store.delete_collection()
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name="cv_documents"
            )
            print("[VectorStoreManager] Successfully cleared existing vector collection.")
        except Exception as e:
            print(f"[VectorStoreManager] Reset notice: {e}")

    def build_vector_store(self, chunks: List[Document], clear_existing: bool = True) -> Chroma:
        """
        Indexes document chunks into Chroma vector store.
        Persists vectors and metadata to disk.

        Args:
            chunks (List[Document]): Tokenized document text chunks.
            clear_existing (bool): If True, resets existing collection before indexing.

        Returns:
            Chroma: Updated vector store instance.
        """
        if not chunks:
            print("[Warning] No document chunks provided for indexing.")
            return self.vector_store

        # Deduplicate chunks based on trimmed page_content
        seen_texts = set()
        unique_chunks = []
        for chunk in chunks:
            text_strip = chunk.page_content.strip()
            if text_strip not in seen_texts:
                seen_texts.add(text_strip)
                unique_chunks.append(chunk)

        if clear_existing:
            self.clear_vector_store()

        print(f"Indexing {len(unique_chunks)} unique text chunks into ChromaDB at '{self.persist_dir}'...")
        
        # Adding document chunks to ChromaDB generates embeddings via NVIDIA API
        # and stores vector coordinates alongside original text and metadata.
        self.vector_store.add_documents(documents=unique_chunks)
        print("Vector database build & persistence complete.")
        return self.vector_store

    def get_retriever(self, search_kwargs: dict = None):
        """
        Returns a LangChain retriever interface for vector similarity queries.

        Args:
            search_kwargs (dict): Search parameters such as k (number of top results to retrieve).

        Returns:
            VectorStoreRetriever: LangChain retriever instance.
        """
        kwargs = search_kwargs or {"k": Config.RETRIEVAL_K}
        return self.vector_store.as_retriever(search_kwargs=kwargs)

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """
        Executes vector similarity search for a query string.

        Args:
            query (str): User question.
            k (int): Number of top matching snippets to retrieve.

        Returns:
            List[Document]: Top matching document snippets.
        """
        target_k = k or Config.RETRIEVAL_K
        return self.vector_store.similarity_search(query=query, k=target_k)

    def similarity_search_with_scores(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Executes similarity search and returns relevance distance scores for debugging/analysis.

        Args:
            query (str): User question.
            k (int): Number of top matching snippets.

        Returns:
            List[Tuple[Document, float]]: Pairs of (Document, relevance_score).
        """
        target_k = k or Config.RETRIEVAL_K
        return self.vector_store.similarity_search_with_relevance_scores(query=query, k=target_k)

    def get_document_count(self) -> int:
        """
        Returns the total count of indexed document vectors in ChromaDB.
        """
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0
