"""Verification package for claim citation parsing, structural validation, and support checking."""

from __future__ import annotations

from rag.verification.citation_parser import parse_claims
from rag.verification.models import SupportVerdict

__all__ = [
    "SupportVerdict",
    "parse_claims",
]
