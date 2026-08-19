"""Structure-aware Markdown chunker prioritizing headings, paragraphs, and list blocks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.ingestion.chunking.base import compute_chunk_id
from rag.retrieval.base import ChunkRecord


@dataclass
class _Block:
    block_type: str  # "heading", "paragraph", "list"
    text: str
    section_path: str
    items: list[str] | None = None


class StructureAwareChunker:
    """Chunks Markdown documents preserving ATX heading hierarchies, paragraphs, and lists."""

    def __init__(self, max_chars: int = 700, overlap_chars: int = 140) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive.")
        if overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative.")
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars.")

        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chars,
            chunk_overlap=self.overlap_chars,
            separators=["\n\n", "\n• ", "\n- ", "\n", " ", ""],
        )

    @property
    def strategy_name(self) -> str:
        return "structure_aware_v1"

    def parameters(self) -> Mapping[str, Any]:
        return {
            "max_chars": self.max_chars,
            "overlap_chars": self.overlap_chars,
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

            # 1. Normalize line endings only
            text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
            sections = self._parse_sections(text)

            for section_path, raw_blocks in sections:
                atomic_blocks = self._expand_oversized_blocks(raw_blocks, section_path)
                chunks_text = self._assemble_chunks(atomic_blocks)

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
                        "section_path": section_path,
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

    def _parse_sections(self, text: str) -> list[tuple[str, list[_Block]]]:
        """Parses text into sections delineated by ATX headings with stack-derived paths."""
        lines = text.split("\n")
        heading_stack: list[tuple[int, str]] = []
        sections: list[tuple[str, list[_Block]]] = []

        current_path = ""
        current_blocks: list[_Block] = []

        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)

            if heading_match:
                # Close prior section if it had blocks
                if current_blocks:
                    sections.append((current_path, current_blocks))
                    current_blocks = []

                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # Truncate stack when same-or-higher-level heading begins
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                current_path = " > ".join(t for _, t in heading_stack)

                current_blocks.append(_Block("heading", line, current_path))
                i += 1
                continue

            if not line.strip():
                i += 1
                continue

            # Check if line starts a list
            if re.match(r"^\s*([*•+-]|\d+[.)])\s+", line):
                list_lines: list[str] = [line]
                items: list[str] = [line]
                i += 1
                while i < n:
                    next_line = lines[i]
                    if not next_line.strip():
                        break
                    if re.match(r"^(#{1,6})\s+", next_line):
                        break
                    if re.match(r"^\s*([*•+-]|\d+[.)])\s+", next_line):
                        list_lines.append(next_line)
                        items.append(next_line)
                    else:
                        # Indented list item continuation
                        list_lines.append(next_line)
                        if items:
                            items[-1] = items[-1] + "\n" + next_line
                    i += 1
                current_blocks.append(_Block("list", "\n".join(list_lines), current_path, items))
                continue

            # Regular paragraph
            para_lines: list[str] = [line]
            i += 1
            while i < n:
                next_line = lines[i]
                if not next_line.strip() or re.match(r"^(#{1,6})\s+", next_line) or re.match(r"^\s*([*•+-]|\d+[.)])\s+", next_line):
                    break
                para_lines.append(next_line)
                i += 1
            current_blocks.append(_Block("paragraph", "\n".join(para_lines), current_path))

        if current_blocks:
            sections.append((current_path, current_blocks))

        return sections

    def _expand_oversized_blocks(self, blocks: list[_Block], section_path: str) -> list[_Block]:
        """Ensures all individual blocks respect max_chars using recursive fallback when needed."""
        atomic: list[_Block] = []

        for block in blocks:
            if len(block.text) <= self.max_chars:
                atomic.append(block)
                continue

            if block.block_type == "list" and block.items:
                # 1. Expand items if an individual item > max_chars
                expanded_items: list[str] = []
                for item in block.items:
                    if len(item) > self.max_chars:
                        sub_docs = self._fallback_splitter.split_documents([Document(page_content=item)])
                        expanded_items.extend(d.page_content for d in sub_docs)
                    else:
                        expanded_items.append(item)

                # 2. Group items into list sub-blocks <= max_chars
                curr_items: list[str] = []
                for item in expanded_items:
                    candidate = "\n".join(curr_items + [item]) if curr_items else item
                    if len(candidate) <= self.max_chars:
                        curr_items.append(item)
                    else:
                        if curr_items:
                            atomic.append(_Block("list", "\n".join(curr_items), section_path, curr_items))
                        curr_items = [item]
                if curr_items:
                    atomic.append(_Block("list", "\n".join(curr_items), section_path, curr_items))
            else:
                # Oversized paragraph or heading fallback to recursive splitting
                sub_docs = self._fallback_splitter.split_documents([Document(page_content=block.text)])
                for sub_doc in sub_docs:
                    atomic.append(_Block(block.block_type, sub_doc.page_content, section_path))

        return atomic

    def _assemble_chunks(self, blocks: list[_Block]) -> list[str]:
        """Combines blocks into chunks up to max_chars with overlap carryover."""
        chunks: list[str] = []
        current: list[_Block] = []

        for b in blocks:
            cand_text = "\n\n".join(x.text for x in current + [b]) if current else b.text
            if len(cand_text) <= self.max_chars:
                current.append(b)
            else:
                if current:
                    chunks.append("\n\n".join(x.text for x in current))

                    # Calculate carryover complete blocks <= overlap_chars
                    carry: list[_Block] = []
                    total_len = 0
                    for prev_b in reversed(current):
                        cand_len = len(prev_b.text) if not carry else total_len + 2 + len(prev_b.text)
                        if cand_len <= self.overlap_chars:
                            carry.insert(0, prev_b)
                            total_len = cand_len
                        else:
                            break

                    # Rule 7: Never carry a heading alone
                    if len(carry) == 1 and carry[0].block_type == "heading":
                        carry = []

                    cand_with_carry = "\n\n".join(x.text for x in carry + [b]) if carry else b.text
                    if len(cand_with_carry) <= self.max_chars:
                        current = carry + [b]
                    else:
                        current = [b]
                else:
                    chunks.append(b.text)
                    current = []

        if current:
            chunks.append("\n\n".join(x.text for x in current))

        return chunks
