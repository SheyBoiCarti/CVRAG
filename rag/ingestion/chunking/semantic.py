"""Semantic chunking strategy partitioning on embedding cosine distance."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

from rag.ingestion.chunking.base import compute_chunk_id
from rag.retrieval.base import ChunkRecord


class SemanticChunker:
    """Experimental semantic chunker using embedding cosine distance thresholds."""

    def __init__(
        self,
        embeddings: Embeddings,
        break_threshold: float = 0.35,
        min_chars: int = 200,
        max_chars: int = 900,
        overlap_units: int = 1,
        embedding_model: str = "nvidia/nv-embedqa-e5-v5",
    ) -> None:
        if not (0.0 <= break_threshold <= 2.0):
            raise ValueError("break_threshold must be between 0.0 and 2.0.")
        if min_chars <= 0:
            raise ValueError("min_chars must be positive.")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive.")
        if min_chars >= max_chars:
            raise ValueError("min_chars must be strictly less than max_chars.")
        if overlap_units not in (0, 1):
            raise ValueError("overlap_units must be 0 or 1.")

        self.embeddings = embeddings
        self.break_threshold = float(break_threshold)
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)
        self.overlap_units = int(overlap_units)
        self.embedding_model = str(embedding_model)

        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chars,
            chunk_overlap=140,
            separators=["\n\n", "\n• ", "\n- ", "\n", " ", ""],
        )

    @property
    def strategy_name(self) -> str:
        return "semantic_v1"

    def parameters(self) -> Mapping[str, Any]:
        return {
            "break_threshold": self.break_threshold,
            "embedding_model": self.embedding_model,
            "max_chars": self.max_chars,
            "min_chars": self.min_chars,
            "overlap_units": self.overlap_units,
        }

    def split(self, documents: Sequence[Document]) -> list[ChunkRecord]:
        page_counters: dict[tuple[str, int], int] = {}
        records: list[ChunkRecord] = []
        strategy = self.strategy_name
        params = self.parameters()

        for doc in documents:
            source = str(doc.metadata.get("source", ""))
            page = int(doc.metadata.get("page", 0))
            raw_text = doc.page_content
            if not raw_text:
                continue

            text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
            units = self._extract_units(text)
            if not units:
                continue

            chunks_text = self._partition_units(units)

            for chunk_text in chunks_text:
                if not chunk_text:
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

        return sorted(records, key=lambda c: (c.source, c.page, c.chunk_index, c.chunk_id))

    def _extract_units(self, text: str) -> list[str]:
        """Splits source text into paragraph and sentence units respecting max_chars."""
        raw_paragraphs = re.split(r"\n{2,}", text)
        units: list[str] = []

        for p_raw in raw_paragraphs:
            p = p_raw.strip()
            if not p:
                continue

            if len(p) <= self.max_chars:
                units.append(p)
            else:
                # Split paragraph into sentences on fixed boundary regex
                sentences = re.split(r"(?<=[.!?])\s+", p)
                for s in sentences:
                    s_clean = s.strip()
                    if not s_clean:
                        continue
                    if len(s_clean) <= self.max_chars:
                        units.append(s_clean)
                    else:
                        # Fallback to recursive splitter for oversized sentences
                        sub_docs = self._fallback_splitter.split_documents([Document(page_content=s_clean)])
                        for sub in sub_docs:
                            sub_text = sub.page_content.strip()
                            if sub_text:
                                units.append(sub_text)

        return units

    def _partition_units(self, units: list[str]) -> list[str]:
        """Clusters units into chunks using adjacent embedding cosine distance."""
        if not units:
            return []
        if len(units) == 1:
            return [units[0]]

        # Embed all units and L2-normalize
        raw_vectors = self.embeddings.embed_documents(units)
        vectors = np.array(raw_vectors, dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normed = vectors / norms

        # Compute adjacent cosine distances: 1 - cosine_similarity(left, right)
        # With unit-normalized vectors, cosine_similarity is dot product
        distances: list[float] = []
        for i in range(len(units) - 1):
            sim = float(np.dot(normed[i], normed[i + 1]))
            # Bound numerical precision to [-1.0, 1.0]
            sim = max(-1.0, min(1.0, sim))
            distances.append(1.0 - sim)

        chunks: list[str] = []
        current: list[str] = [units[0]]

        for i in range(1, len(units)):
            next_unit = units[i]
            dist = distances[i - 1]

            curr_len = sum(len(u) for u in current) + 2 * (len(current) - 1)
            cand_len = curr_len + 2 + len(next_unit)

            should_break = False
            if curr_len >= self.min_chars:
                if dist >= self.break_threshold or cand_len > self.max_chars:
                    should_break = True
            else:
                if cand_len > self.max_chars:
                    should_break = True

            if should_break:
                chunks.append("\n\n".join(current))
                if self.overlap_units == 1 and current:
                    overlap_unit = current[-1]
                    cand_with_overlap = f"{overlap_unit}\n\n{next_unit}"
                    if len(cand_with_overlap) <= self.max_chars:
                        current = [overlap_unit, next_unit]
                    else:
                        current = [next_unit]
                else:
                    current = [next_unit]
            else:
                current.append(next_unit)

        if current:
            chunks.append("\n\n".join(current))

        return chunks
