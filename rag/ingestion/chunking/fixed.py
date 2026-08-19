"""Fixed-size character window chunker."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_core.documents import Document

from rag.ingestion.chunking.base import compute_chunk_id
from rag.retrieval.base import ChunkRecord


class FixedCharacterChunker:
    """Chunks documents into deterministic fixed-character sliding windows."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 140) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def strategy_name(self) -> str:
        return "fixed_character_v1"

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
        step = self.chunk_size - self.chunk_overlap

        for doc in documents:
            source = str(doc.metadata.get("source", ""))
            page = int(doc.metadata.get("page", 0))
            text = doc.page_content
            if not text:
                continue

            text_len = len(text)
            start = 0

            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                chunk_text = text[start:end]

                if chunk_text:
                    key = (source, page)
                    chunk_index = page_counters.get(key, 0)
                    page_counters[key] = chunk_index + 1

                    chunk_id = compute_chunk_id(
                        source=source,
                        page=page,
                        strategy=strategy,
                        parameters=params,
                        chunk_index=chunk_index,
                        content=chunk_text,
                    )
                    metadata = {
                        "chunk_strategy": strategy,
                        "section_path": "",
                    }
                    records.append(
                        ChunkRecord(
                            chunk_id=chunk_id,
                            content=chunk_text,
                            source=source,
                            page=page,
                            chunk_index=chunk_index,
                            metadata=metadata,
                        )
                    )

                if end >= text_len:
                    break
                start += step

        return sorted(records, key=lambda c: (c.source, c.page, c.chunk_index, c.chunk_id))
