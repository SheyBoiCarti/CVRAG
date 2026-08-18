import dataclasses
import json
import sys
import pytest

from rag.models import (
    PipelineDecision,
    EvidenceSupportStatus,
    ClaimVerificationStatus,
    ClaimImportance,
    ReadinessState,
    ChatTurn,
    Evidence,
    Citation,
    ClaimVerification,
    AnswerResult,
    ReadinessResult,
    PipelineEvent,
)


def test_chat_turn_creation_and_immutability():
    turn = ChatTurn(user="What is your experience?", assistant="I have 5 years in AI.")
    assert turn.user == "What is your experience?"
    assert turn.assistant == "I have 5 years in AI."

    # Immutability
    with pytest.raises(dataclasses.FrozenInstanceError):
        turn.user = "New question"  # type: ignore

    # Serialization
    assert turn.to_dict() == {
        "user": "What is your experience?",
        "assistant": "I have 5 years in AI.",
    }
    assert dataclasses.asdict(turn) == {
        "user": "What is your experience?",
        "assistant": "I have 5 years in AI.",
    }


def test_evidence_creation_and_immutability():
    ev = Evidence(
        evidence_id="ev-1",
        chunk_id="chunk-42",
        content="Shehan is a machine learning engineer.",
        source="resume.pdf",
        page=1,
    )
    assert ev.evidence_id == "ev-1"
    assert ev.chunk_id == "chunk-42"
    assert ev.content == "Shehan is a machine learning engineer."
    assert ev.source == "resume.pdf"
    assert ev.page == 1
    assert ev.metadata == {}

    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.page = 2  # type: ignore

    ev_with_meta = Evidence(
        evidence_id="ev-2",
        chunk_id="chunk-43",
        content="Graduated in 2020.",
        source="cv.pdf",
        page=2,
        metadata={"section": "education", "score": 0.95},
    )
    assert ev_with_meta.metadata == {"section": "education", "score": 0.95}
    assert ev_with_meta.to_dict() == {
        "evidence_id": "ev-2",
        "chunk_id": "chunk-43",
        "content": "Graduated in 2020.",
        "source": "cv.pdf",
        "page": 2,
        "metadata": {"section": "education", "score": 0.95},
    }


def test_citation_creation_and_immutability():
    cit = Citation(
        evidence_id="ev-1",
        chunk_id="chunk-42",
        source="resume.pdf",
        page=1,
        excerpt="Shehan is a machine learning engineer.",
    )
    assert cit.evidence_id == "ev-1"
    assert cit.chunk_id == "chunk-42"
    assert cit.source == "resume.pdf"
    assert cit.page == 1
    assert cit.excerpt == "Shehan is a machine learning engineer."
    assert cit.support_status == "not_checked"

    with pytest.raises(dataclasses.FrozenInstanceError):
        cit.support_status = "supported"  # type: ignore

    cit_supported = Citation(
        evidence_id="ev-1",
        chunk_id="chunk-42",
        source="resume.pdf",
        page=1,
        excerpt="Shehan is a machine learning engineer.",
        support_status="supported",
    )
    assert cit_supported.support_status == "supported"
    assert cit_supported.to_dict() == {
        "evidence_id": "ev-1",
        "chunk_id": "chunk-42",
        "source": "resume.pdf",
        "page": 1,
        "excerpt": "Shehan is a machine learning engineer.",
        "support_status": "supported",
    }


def test_claim_verification_creation_and_immutability():
    cv = ClaimVerification(
        claim_id="claim-1",
        text="Candidate has experience with Python.",
    )
    assert cv.claim_id == "claim-1"
    assert cv.text == "Candidate has experience with Python."
    assert cv.factual is True
    assert cv.importance == "core"
    assert cv.citation_ids == ()
    assert cv.status == "not_checked"
    assert cv.reason_code == ""

    with pytest.raises(dataclasses.FrozenInstanceError):
        cv.status = "supported"  # type: ignore

    # Sequence conversion to tuple
    cv_with_citations = ClaimVerification(
        claim_id="claim-2",
        text="Candidate led a team of 5.",
        factual=True,
        importance="optional",
        citation_ids=["ev-1", "ev-2"],  # type: ignore
        status="supported",
        reason_code="exact_match",
    )
    assert cv_with_citations.citation_ids == ("ev-1", "ev-2")
    assert cv_with_citations.to_dict() == {
        "claim_id": "claim-2",
        "text": "Candidate led a team of 5.",
        "factual": True,
        "importance": "optional",
        "citation_ids": ["ev-1", "ev-2"],
        "status": "supported",
        "reason_code": "exact_match",
    }


def test_answer_result_creation_and_immutability():
    cit = Citation(
        evidence_id="ev-1",
        chunk_id="chunk-1",
        source="resume.pdf",
        page=1,
        excerpt="Lead Engineer at Tech Corp.",
        support_status="supported",
    )
    cv = ClaimVerification(
        claim_id="cl-1",
        text="Worked at Tech Corp.",
        citation_ids=("ev-1",),
        status="supported",
    )

    result = AnswerResult(
        answer="The candidate worked at Tech Corp as Lead Engineer.",
        citations=[cit],  # type: ignore
        claim_verifications=[cv],  # type: ignore
        decision="answered",
        reason_codes=["high_confidence"],  # type: ignore
        retrieval_query="Tech Corp experience",
        pipeline_version="dense-v1",
    )

    assert result.answer == "The candidate worked at Tech Corp as Lead Engineer."
    assert result.citations == (cit,)
    assert result.claim_verifications == (cv,)
    assert result.decision == "answered"
    assert result.reason_codes == ("high_confidence",)
    assert result.retrieval_query == "Tech Corp experience"
    assert result.pipeline_version == "dense-v1"

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.answer = "New answer"  # type: ignore

    d = result.to_dict()
    assert d["answer"] == "The candidate worked at Tech Corp as Lead Engineer."
    assert d["decision"] == "answered"
    assert d["reason_codes"] == ["high_confidence"]
    assert d["retrieval_query"] == "Tech Corp experience"
    assert d["pipeline_version"] == "dense-v1"
    assert len(d["citations"]) == 1
    assert d["citations"][0]["evidence_id"] == "ev-1"
    assert len(d["claim_verifications"]) == 1
    assert d["claim_verifications"][0]["claim_id"] == "cl-1"


def test_readiness_result_creation_and_immutability():
    ready = ReadinessResult(
        state="ready",
        pipeline_version="dense-v1",
        index_fingerprint="sha256:abc123",
    )
    assert ready.state == "ready"
    assert ready.is_ready is True
    assert ready.reason_codes == ()
    assert ready.pipeline_version == "dense-v1"
    assert ready.index_fingerprint == "sha256:abc123"

    with pytest.raises(dataclasses.FrozenInstanceError):
        ready.state = "not_ready"  # type: ignore

    not_ready = ReadinessResult(
        state="not_ready",
        reason_codes=["vector_db_missing", "index_outdated"],  # type: ignore
        pipeline_version=None,
        index_fingerprint=None,
    )
    assert not_ready.state == "not_ready"
    assert not_ready.is_ready is False
    assert not_ready.reason_codes == ("vector_db_missing", "index_outdated")
    assert not_ready.to_dict() == {
        "state": "not_ready",
        "reason_codes": ["vector_db_missing", "index_outdated"],
        "pipeline_version": None,
        "index_fingerprint": None,
    }


def test_pipeline_event_creation_and_immutability():
    event = PipelineEvent(
        schema_version="1",
        request_id="req-12345",
        sequence=1,
        event="retrieval_started",
        data={"query": "experience with Python"},
    )
    assert event.schema_version == "1"
    assert event.request_id == "req-12345"
    assert event.sequence == 1
    assert event.event == "retrieval_started"
    assert event.data == {"query": "experience with Python"}

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.sequence = 2  # type: ignore

    assert event.to_dict() == {
        "schema_version": "1",
        "request_id": "req-12345",
        "sequence": 1,
        "event": "retrieval_started",
        "data": {"query": "experience with Python"},
    }

    # Test create factory helper
    event2 = PipelineEvent.create(
        event="generation_started",
        data={"model": "llama-3.3-70b"},
        request_id="req-12345",
        sequence=2,
    )
    assert event2.schema_version == "1"
    assert event2.request_id == "req-12345"
    assert event2.sequence == 2
    assert event2.event == "generation_started"
    assert event2.data == {"model": "llama-3.3-70b"}

    # Test SSE formatting helper
    sse_str = event2.to_sse()
    assert sse_str.startswith("event: generation_started\n")
    assert "data: " in sse_str
    assert sse_str.endswith("\n\n")
    parsed_json = json.loads(sse_str.split("data: ")[1].strip())
    assert parsed_json["schema_version"] == "1"
    assert parsed_json["request_id"] == "req-12345"
    assert parsed_json["sequence"] == 2
    assert parsed_json["event"] == "generation_started"
    assert parsed_json["data"] == {"model": "llama-3.3-70b"}


def test_no_external_dependencies_in_models():
    # Verify in a clean Python process that loading rag/models.py as a standalone module loads no external framework modules
    import subprocess

    code = (
        "import sys, importlib.util\n"
        "spec = importlib.util.spec_from_file_location('models', 'rag/models.py')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['models'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "forbidden = ('langchain', 'fastapi', 'pydantic', 'starlette', 'streamlit', 'uvicorn', 'chromadb')\n"
        "loaded = [m for m in sys.modules if any(m.startswith(f) for f in forbidden)]\n"
        "if loaded:\n"
        "    print(f'Forbidden: {loaded}')\n"
        "    sys.exit(1)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"External dependencies imported: {proc.stdout} {proc.stderr}"


def test_models_ast_imports_standard_library_only():
    import ast
    from pathlib import Path

    models_path = Path(__file__).resolve().parent.parent.parent / "rag" / "models.py"
    tree = ast.parse(models_path.read_text(encoding="utf-8"))

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imported_modules.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split(".")[0])

    # Allowable standard library modules only
    allowed_stdlib = {"__future__", "dataclasses", "json", "typing", "types"}
    disallowed = imported_modules - allowed_stdlib
    assert not disallowed, f"rag/models.py imported non-stdlib or unapproved modules: {disallowed}"


