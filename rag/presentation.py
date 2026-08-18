from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rag.errors import ProviderUnavailableError
from rag.models import Citation


PROVIDER_UNAVAILABLE_MESSAGE = (
    "The AI provider is temporarily unavailable. Please try again shortly."
)
GENERIC_QUERY_ERROR_MESSAGE = (
    "Couldn't generate a response right now. Please try asking your question again."
)


@dataclass(frozen=True)
class CitationView:
    label: str
    excerpt: str


def citation_view(document: Document | Citation | Any) -> CitationView:
    """Creates a presentation-ready citation view supporting both Document and Citation models."""
    if isinstance(document, Citation):
        source = Path(str(document.source or "CV")).name
        page = int(document.page)
        evidence_id = str(document.evidence_id or "Source")
        excerpt = (document.excerpt or "").strip()
        return CitationView(
            label=f"{evidence_id} · {source} · page {page}",
            excerpt=excerpt,
        )

    metadata = getattr(document, "metadata", {}) or {}
    source = Path(str(metadata.get("source", "CV"))).name
    raw_page = metadata.get("page", 0)
    try:
        page = int(raw_page) + 1
    except (ValueError, TypeError):
        page = 1
    evidence_id = str(metadata.get("evidence_id", "Source"))
    page_content = getattr(document, "page_content", getattr(document, "excerpt", "")) or ""
    return CitationView(
        label=f"{evidence_id} · {source} · page {page}",
        excerpt=str(page_content).strip(),
    )


def query_error_message(exc: Exception) -> str:
    """Maps query failures to safe messages that do not expose internal details."""
    if isinstance(exc, ProviderUnavailableError):
        return PROVIDER_UNAVAILABLE_MESSAGE
    return GENERIC_QUERY_ERROR_MESSAGE
