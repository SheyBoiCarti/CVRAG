from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document


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
