"""Baseline recursive character chunker with bullet-aware separators."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.ingestion.chunking.base import compute_chunk_id
from rag.retrieval.base import ChunkRecord


class RecursiveBulletChunker:
    """Baseline chunking strategy using RecursiveCharacterTextSplitter with bullet separators."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 140) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n• ", "\n- ", "\n", " ", ""]
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    @property
    def strategy_name(self) -> str:
        return "recursive_bullet_v1"

    def parameters(self) -> Mapping[str, Any]:
        return {
            "chunk_overlap": self.chunk_overlap,
            "chunk_size": self.chunk_size,
        }

    def split(self, documents: Sequence[Document]) -> list[ChunkRecord]:
        page_counters: dict[tuple[str, int], int] = {}
        records: list[ChunkRecord] = []
        strategy = self.strategy_name
        params = self.parameters()

        for doc in documents:
            source = str(doc.metadata.get("source", ""))
            page = int(doc.metadata.get("page", 0))
            split_docs = self._splitter.split_documents([doc])

            for chunk_doc in split_docs:
                content = chunk_doc.page_content
                if not content:
                    continue
                key = (source, page)
                chunk_index = page_counters.get(key, 0)
                page_counters[key] = chunk_index + 1

                chunk_id = compute_chunk_id(
                    source=source,
                    page=page,
                    strategy=strategy,
                    parameters=params,
                    chunk_index=chunk_index,
                    content=content,
                )
                metadata = {
                    "chunk_strategy": strategy,
                    "section_path": "",
                }
                records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        content=content,
                        source=source,
                        page=page,
                        chunk_index=chunk_index,
                        metadata=metadata,
                    )
                )

        return sorted(records, key=lambda c: (c.source, c.page, c.chunk_index, c.chunk_id))
