from __future__ import annotations

from rag.models import AnswerResult
from rag.pipeline import RAGPipeline
from rag.settings import Settings
from rag.verification.policy import STANDARD_ABSTENTION_TEXT
from rag.verification.support import FakeSupportVerifier
from tests.fakes import FakeChatModel, FakeProviderFactories


def test_verified_pipeline_synchronous_empty_retrieval(tmp_path):
    # Empty documents dir -> empty retrieval
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    vec_dir = tmp_path / "vec"
    vec_dir.mkdir()

    settings = Settings(
        DOCUMENTS_DIR=doc_dir,
        VECTOR_DB_DIR=vec_dir,
        NVIDIA_API_KEY="test-key",
    )
    factories = FakeProviderFactories()
    pipeline = RAGPipeline(settings=settings, provider_factories=factories)

    res = pipeline.answer("What is his degree?")
    assert res.decision == "abstained"
    assert "no_evidence" in res.reason_codes
    assert res.citations == ()
    assert res.claim_verifications == ()


def test_verified_pipeline_synchronous_supported_answer(tmp_path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "Custom_CV.md").write_text("# CV\nSheharyar worked at Wirk as Software Engineer.\n", encoding="utf-8")
    vec_dir = tmp_path / "vec"
    vec_dir.mkdir()

    settings = Settings(
        DOCUMENTS_DIR=doc_dir,
        VECTOR_DB_DIR=vec_dir,
        NVIDIA_API_KEY="test-key",
    )
    chat_model = FakeChatModel(
        responses=["Sheharyar worked at Wirk [S1]."]
    )
    verifier = FakeSupportVerifier(default_supported=True)
    factories = FakeProviderFactories(chat_model=chat_model, support_verifier=verifier)
    pipeline = RAGPipeline(settings=settings, provider_factories=factories)

    res = pipeline.answer("Where did Sheharyar work?")
    assert res.decision == "answered"
    assert res.answer == "Sheharyar worked at Wirk [S1]."
    assert len(res.citations) == 1
    assert res.citations[0].evidence_id == "S1"
    assert res.citations[0].support_status == "supported"
    assert len(res.claim_verifications) == 1
    assert res.claim_verifications[0].status == "supported"


def test_verified_pipeline_synchronous_unsupported_correction_cycle(tmp_path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "Custom_CV.md").write_text("# CV\nSheharyar worked at Wirk.\n", encoding="utf-8")
    vec_dir = tmp_path / "vec"
    vec_dir.mkdir()

    settings = Settings(
        DOCUMENTS_DIR=doc_dir,
        VECTOR_DB_DIR=vec_dir,
        NVIDIA_API_KEY="test-key",
    )
    # First response contains "Google" (unsupported), second response is corrected to "Wirk"
    chat_model = FakeChatModel(
        responses=[
            "He worked at Google [S1].",
            "He worked at Wirk [S1].",
        ]
    )
    verifier = FakeSupportVerifier(
        rule_fn=lambda c, ev: "Google" not in c.text,
        rejection_reason="unsupported_claim",
    )
    factories = FakeProviderFactories(chat_model=chat_model, support_verifier=verifier)
    pipeline = RAGPipeline(settings=settings, provider_factories=factories)

    res = pipeline.answer("Where did he work?")
    assert res.decision == "answered"
    assert "Wirk" in res.answer
    assert res.citations[0].support_status == "supported"


def test_verified_pipeline_synchronous_correction_failure_abstains(tmp_path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "Custom_CV.md").write_text("# CV\nSheharyar worked at Wirk.\n", encoding="utf-8")
    vec_dir = tmp_path / "vec"
    vec_dir.mkdir()

    settings = Settings(
        DOCUMENTS_DIR=doc_dir,
        VECTOR_DB_DIR=vec_dir,
        NVIDIA_API_KEY="test-key",
    )
    # Both responses are unsupported
    chat_model = FakeChatModel(
        responses=[
            "He worked at Google [S1].",
            "He worked at Amazon [S1].",
        ]
    )
    verifier = FakeSupportVerifier(
        rule_fn=lambda c, ev: False,
        rejection_reason="unsupported_claim",
    )
    factories = FakeProviderFactories(chat_model=chat_model, support_verifier=verifier)
    pipeline = RAGPipeline(settings=settings, provider_factories=factories)

    res = pipeline.answer("Where did he work?")
    assert res.decision == "abstained"
    assert res.answer == STANDARD_ABSTENTION_TEXT
    assert "unsupported_claim" in res.reason_codes


def test_verified_pipeline_streaming_lifecycle_and_replace_flag(tmp_path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "Custom_CV.md").write_text("# CV\nSheharyar worked at Wirk.\n", encoding="utf-8")
    vec_dir = tmp_path / "vec"
    vec_dir.mkdir()

    settings = Settings(
        DOCUMENTS_DIR=doc_dir,
        VECTOR_DB_DIR=vec_dir,
        NVIDIA_API_KEY="test-key",
    )
    # Raw answer is unsupported, corrective generation produces corrected answer
    chat_model = FakeChatModel(
        responses=[
            "He worked at Google [S1].",
            "He worked at Wirk [S1].",
        ]
    )
    verifier = FakeSupportVerifier(
        rule_fn=lambda c, ev: "Google" not in c.text,
        rejection_reason="unsupported_claim",
    )
    factories = FakeProviderFactories(chat_model=chat_model, support_verifier=verifier)
    pipeline = RAGPipeline(settings=settings, provider_factories=factories)

    events = list(pipeline.stream("Where did he work?"))
    event_names = [e.event for e in events]
    assert event_names[0] == "retrieval_started"
    assert event_names[1] == "retrieval_completed"
    assert event_names[2] == "generation_started"
    assert "answer_delta" in event_names
    assert "verification_started" in event_names
    assert event_names[-1] == "answer_completed"

    final_event = events[-1]
    assert final_event.data["replace_streamed_answer"] is True
    result = final_event.data["result"]
    assert isinstance(result, AnswerResult)
    assert result.decision == "answered"
    assert "Wirk" in result.answer
