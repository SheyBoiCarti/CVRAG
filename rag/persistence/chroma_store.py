"""Public Chroma adapter used by ingestion and dense retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import warnings

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.retrieval.base import ChunkRecord


class ChromaStoreAdapter:
    """Owns all direct Chroma collection access for canonical chunk records."""

    def __init__(
        self,
        *,
        persist_dir: Path | str,
        collection_name: str,
        embeddings: Embeddings,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._embeddings = embeddings
        self._store = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=embeddings,
            collection_name=collection_name,
        )

    def clear(self) -> None:
        """Deletes the underlying collection to ensure a clean build slate."""
        try:
            self._store.delete_collection()
            self._store = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self._embeddings,
                collection_name=self.collection_name,
            )
        except Exception:
            pass

    def count(self) -> int:
        """Returns the number of records in this collection."""
        return int(self._store._collection.count())

    def add_chunks(self, chunks: Sequence[ChunkRecord]) -> int:
        """Upserts canonical chunks, using their stable IDs as collection IDs."""
        if not chunks:
            return self.count()
        documents = [self._to_document(chunk) for chunk in chunks]
        self._store.add_documents(documents=documents, ids=[chunk.chunk_id for chunk in chunks])
        return self.count()

    def similarity_search_with_scores(
        self, query: str, *, k: int
    ) -> list[tuple[ChunkRecord, float]]:
        """Returns relevance-scored canonical chunks from Chroma."""
        if k <= 0:
            return []
        # Chroma can return raw diagnostics outside its normalized relevance
        # range; they remain valid ranking diagnostics for this adapter.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Relevance scores must be between 0 and 1.*",
                category=UserWarning,
            )
            results = self._store.similarity_search_with_relevance_scores(
                query=query, k=k
            )
        return [
            (self._from_document(document), float(score))
            for document, score in results
        ]

    def get_existing_ids(self, chunk_ids: Sequence[str]) -> set[str]:
        """Returns supplied IDs that are currently present in the collection."""
        if not chunk_ids:
            return set()
        result = self._store._collection.get(ids=list(chunk_ids), include=[])
        return {str(chunk_id) for chunk_id in result.get("ids", [])}

    @staticmethod
    def _to_document(chunk: ChunkRecord) -> Document:
        return Document(
            page_content=chunk.content,
            metadata={
                **dict(chunk.metadata),
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
            },
        )

    @staticmethod
    def _from_document(document: Document) -> ChunkRecord:
        metadata = dict(document.metadata)
        return ChunkRecord(
            chunk_id=str(metadata.pop("chunk_id")),
            content=document.page_content,
            source=str(metadata.pop("source")),
            page=int(metadata.pop("page")),
            chunk_index=int(metadata.pop("chunk_index")),
            metadata=metadata,
        )


__all__ = ["ChromaStoreAdapter"]
