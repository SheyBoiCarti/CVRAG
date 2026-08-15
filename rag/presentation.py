from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from rag.errors import ProviderUnavailableError


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


def citation_view(document: Document) -> CitationView:
    source = Path(str(document.metadata.get("source", "CV"))).name
    page = int(document.metadata.get("page", 0)) + 1
    evidence_id = str(document.metadata.get("evidence_id", "Source"))
    return CitationView(
        label=f"{evidence_id} · {source} · page {page}",
        excerpt=document.page_content.strip(),
    )


def query_error_message(exc: Exception) -> str:
    """Maps query failures to safe messages that do not expose internal details."""
    if isinstance(exc, ProviderUnavailableError):
        return PROVIDER_UNAVAILABLE_MESSAGE
    return GENERIC_QUERY_ERROR_MESSAGE
