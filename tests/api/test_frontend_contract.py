"""OpenAPI schema contract test verifying frontend data contracts against FastAPI."""

from __future__ import annotations

from backend.app.main import create_app
from rag.settings import Settings


def test_openapi_schema_matches_frontend_contract():
    """Validates that FastAPI OpenAPI schema exactly defines the contracts required by frontend."""
    settings = Settings(NVIDIA_API_KEY="test-dummy-key")
    app = create_app(settings=settings)
    openapi = app.openapi()

    assert "paths" in openapi
    assert "/api/v1/chat" in openapi["paths"]
    assert "/api/v1/chat/stream" in openapi["paths"]

    schemas = openapi.get("components", {}).get("schemas", {})

    # Verify ChatRequest schema
    assert "ChatRequest" in schemas
    chat_req_props = schemas["ChatRequest"].get("properties", {})
    assert "question" in chat_req_props
    assert "history" in chat_req_props

    # Verify ChatResponse schema
    assert "ChatResponse" in schemas
    chat_resp_props = schemas["ChatResponse"].get("properties", {})
    required_response_fields = {
        "answer",
        "citations",
        "claim_verifications",
        "decision",
        "reason_codes",
        "retrieval_query",
        "pipeline_version",
    }
    for field in required_response_fields:
        assert field in chat_resp_props, f"Missing {field} in ChatResponse schema"

    # Verify CitationSchema
    assert "CitationSchema" in schemas
    citation_props = schemas["CitationSchema"].get("properties", {})
    required_citation_fields = {
        "evidence_id",
        "chunk_id",
        "source",
        "page",
        "excerpt",
        "support_status",
    }
    for field in required_citation_fields:
        assert field in citation_props, f"Missing {field} in CitationSchema"

    # Verify ClaimVerificationSchema
    assert "ClaimVerificationSchema" in schemas
    claim_props = schemas["ClaimVerificationSchema"].get("properties", {})
    required_claim_fields = {
        "claim_id",
        "text",
        "factual",
        "importance",
        "citation_ids",
        "status",
        "reason_code",
    }
    for field in required_claim_fields:
        assert field in claim_props, f"Missing {field} in ClaimVerificationSchema"
