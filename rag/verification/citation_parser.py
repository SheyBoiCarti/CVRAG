"""Deterministic sentence segmentation, clause splitting, and citation tag parser."""

from __future__ import annotations

import re

from rag.models import Claim

_CITATION_TAG_PATTERN = re.compile(
    r"\[\s*(?:S\d+(?:\s*,\s*S\d+)*)\s*\]",
    re.IGNORECASE,
)
_SINGLE_S_TAG_PATTERN = re.compile(
    r"S(\d+)",
    re.IGNORECASE,
)
_HEADING_PATTERN = re.compile(r"^\s*#{1,6}\s+")
_LIST_ITEM_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+")
_STANDALONE_CONNECTIVE_PATTERN = re.compile(
    r"^(?:in summary|in conclusion|furthermore|moreover|however|according to the cv|"
    r"based on the cv|based on the provided information|to summarize)[:,\.]?$",
    re.IGNORECASE,
)
_ABSTENTION_PATTERN = re.compile(
    r"(?:don't have enough supported information|couldn't find that information|not listed in the cv)",
    re.IGNORECASE,
)
_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "ltd.",
    "pty.",
    "pty ltd.",
    "inc.",
    "dr.",
    "prof.",
    "mr.",
    "ms.",
    "mrs.",
    "vs.",
    "v5.0",
    "v1.0",
)


def _extract_citation_ids(raw_clause: str) -> tuple[str, ...]:
    """Extracts all [S<num>] citation identifiers from a clause, preserving order and deduplicating."""
    citation_ids: list[str] = []
    seen: set[str] = set()

    for match in _CITATION_TAG_PATTERN.finditer(raw_clause):
        bracketed = match.group(0)
        for s_match in _SINGLE_S_TAG_PATTERN.finditer(bracketed):
            num = int(s_match.group(1))
            if num > 0:
                cit_id = f"S{num}"
                if cit_id not in seen:
                    seen.add(cit_id)
                    citation_ids.append(cit_id)

    return tuple(citation_ids)


def _clean_claim_text(raw_text: str) -> str:
    """Strips citation markers, leading list symbols, and cleans up punctuation/spacing."""
    cleaned = _LIST_ITEM_PREFIX_PATTERN.sub("", raw_text).strip()
    cleaned = _CITATION_TAG_PATTERN.sub("", cleaned)
    # Strip leading coordinating conjunctions from split clauses
    cleaned = re.sub(r"^(?:and|but|while|whereas|moreover)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Normalize dangling punctuation: e.g., "word , and" -> "word, and", "word ." -> "word."
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned and not cleaned.endswith((".", "!", "?", ":", ";")):
        cleaned = cleaned + "."
    return cleaned


def _classify_claim(raw_text: str, clean_text: str) -> tuple[bool, str]:
    """Classifies a claim as factual or nonfactual. All factual claims are 'core' importance."""
    stripped = raw_text.strip()
    clean_stripped = clean_text.strip()

    if not clean_stripped:
        return False, "core"

    if _HEADING_PATTERN.match(stripped):
        return False, "core"

    if stripped.endswith("?"):
        return False, "core"

    if _ABSTENTION_PATTERN.search(stripped):
        return False, "core"

    clean_no_punct = clean_stripped.rstrip(".,;:!?").strip()
    if _STANDALONE_CONNECTIVE_PATTERN.match(clean_no_punct):
        return False, "core"

    # Default conservative classification: affirmative assertions are factual core claims
    return True, "core"


def _split_into_clauses(sentence: str) -> list[str]:
    """Splits compound sentences at semicolons and coordinating conjunctions with explicit subjects."""
    if not sentence.strip():
        return []

    # First split at semicolons
    semi_parts = re.split(r";\s*", sentence)
    clauses: list[str] = []

    for part in semi_parts:
        if not part.strip():
            continue
        # Split at comma followed by coordinating conjunction and explicit subject
        conj_parts = re.split(
            r",\s+(?:and|but|while|whereas|moreover)\s+(?=(?:he|she|they|it|the\s+\w+|sheharyar)\b)",
            part,
            flags=re.IGNORECASE,
        )
        for c in conj_parts:
            if c.strip():
                clauses.append(c.strip())

    return clauses if clauses else [sentence.strip()]


def _split_into_sentences(text: str) -> list[str]:
    """Segments raw text into sentences while respecting markdown lists and common abbreviations."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences: list[str] = []

    for line in lines:
        if _LIST_ITEM_PREFIX_PATTERN.match(line) or _HEADING_PATTERN.match(line):
            sentences.append(line)
            continue

        # Split prose by sentence ending punctuation followed by space and capital letter/digit
        # Protect abbreviations first
        protected = line
        placeholders: dict[str, str] = {}
        for idx, abbr in enumerate(_ABBREVIATIONS):
            if abbr in protected.lower():
                pattern = re.compile(re.escape(abbr), re.IGNORECASE)
                token = f"__ABBR_{idx}__"
                placeholders[token] = abbr
                protected = pattern.sub(token, protected)

        raw_splits = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'#\-])", protected)
        for s in raw_splits:
            restored = s
            for token, orig in placeholders.items():
                restored = restored.replace(token, orig)
            if restored.strip():
                sentences.append(restored.strip())

    return sentences


def parse_claims(text: str) -> tuple[Claim, ...]:
    """Parses text into ordered, atomic Claim models with extracted citation identifiers.

    Fail-closed: returns an empty tuple or valid claims without crashing on malformed input.
    """
    if not text or not text.strip():
        return ()

    try:
        sentences = _split_into_sentences(text)
        claims: list[Claim] = []
        claim_counter = 1

        for sentence in sentences:
            clauses = _split_into_clauses(sentence)
            for clause in clauses:
                citation_ids = _extract_citation_ids(clause)
                clean_text = _clean_claim_text(clause)
                if not clean_text:
                    continue

                factual, importance = _classify_claim(clause, clean_text)
                claims.append(
                    Claim(
                        claim_id=f"C{claim_counter}",
                        text=clean_text,
                        citation_ids=citation_ids,
                        factual=factual,
                        importance=importance,
                    )
                )
                claim_counter += 1

        return tuple(claims)
    except Exception:
        # Fail-closed: return conservative fallback
        clean_fallback = _clean_claim_text(text)
        cits = _extract_citation_ids(text)
        return (
            Claim(
                claim_id="C1",
                text=clean_fallback or text,
                citation_ids=cits,
                factual=True,
                importance="core",
            ),
        )


__all__ = ["parse_claims"]
