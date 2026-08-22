from collections.abc import Sequence

import pytest
from langchain_core.documents import Document

from rag.engine import CVChatEngine
from rag.models import AnswerResult, ChatTurn, Citation, PipelineEvent, ReadinessResult
from rag.pipeline import RAGPipeline
from rag.settings import Settings
from tests.fakes import FakeChatModel, FakeProviderFactories


class StubVectorManager:
    """In-memory vector store manager stub for deterministic integration testing."""

    def __init__(self, documents: Sequence[Document] = ()):
        self.documents = list(documents)

    def similarity_search(self, query: str, k: int = 6) -> list[Document]:
        return self.documents[:k]

    def get_document_count(self) -> int:
        return len(self.documents)

    def ensure_documents(self, chunks: list[Document]) -> int:
        self.documents.extend(chunks)
        return len(self.documents)


def test_answer_non_empty_retrieval_returns_answered_with_citations():
    doc = Document(
        page_content="Expert in Python, FastAPI, and Docker containerization.",
        metadata={
            "source": r"C:\sensitive\path\to\candidate_cv.pdf",
            "page": 0,
            "chunk_id": "chunk_skills_1",
        },
    )
    vector_mgr = StubVectorManager([doc])
    chat_model = FakeChatModel(
        responses=["Candidate has extensive experience in Python and FastAPI [S1]."]
    )
    engine = CVChatEngine(
        auto_index=False,
        vector_manager=vector_mgr,
        llm=chat_model,
    )
    settings = Settings(RETRIEVAL_K=4, PIPELINE_VERSION="dense-v1")
    pipeline = RAGPipeline(settings=settings, engine=engine)

    result = pipeline.answer("What backend technologies are listed?")

    assert isinstance(result, AnswerResult)
    assert result.decision == "answered"
    assert result.reason_codes == ()
    assert result.answer == "Candidate has extensive experience in Python and FastAPI [S1]."
    assert result.pipeline_version == "dense-v1"
    assert result.retrieval_query == "What backend technologies are listed?"

    assert len(result.citations) == 1
    citation = result.citations[0]
    assert isinstance(citation, Citation)
    assert citation.evidence_id == "S1"
    assert citation.chunk_id == "chunk_skills_1"
    assert citation.source == "candidate_cv.pdf"
    assert citation.page == 1  # 0-indexed page in doc metadata maps to 1-based page
    assert citation.excerpt == "Expert in Python, FastAPI, and Docker containerization."
    assert citation.support_status in ("supported", "not_checked")


def test_answer_empty_retrieval_abstains_without_calling_llm():
    vector_mgr = StubVectorManager([])

    class FailIfInvokedChatModel(FakeChatModel):
        def _get_next_response(self, messages):
            raise AssertionError("LLM should not be called when retrieval returns no documents")

    engine = CVChatEngine(
        auto_index=False,
        vector_manager=vector_mgr,
        llm=FailIfInvokedChatModel(),
    )
    settings = Settings(PIPELINE_VERSION="dense-v1")
    pipeline = RAGPipeline(settings=settings, engine=engine)

    result = pipeline.answer("What is the candidate's favorite color?")

    assert isinstance(result, AnswerResult)
    assert result.decision == "abstained"
    assert result.reason_codes == ("no_evidence",)
    assert result.answer == "I couldn't find that information in the CV."
    assert result.citations == ()
    assert result.claim_verifications == ()
    assert result.pipeline_version == "dense-v1"


def test_stream_non_empty_retrieval_emits_expected_event_sequence():
    doc = Document(
        page_content="Led data engineering initiatives at TechCorp.",
        metadata={
            "source": "/var/data/cvs/resume.md",
            "page": 1,
            "chunk_id": "exp_techcorp",
        },
    )
    vector_mgr = StubVectorManager([doc])
    chat_model = FakeChatModel(
        responses=["Led data engineering projects at TechCorp [S1]."]
    )
    engine = CVChatEngine(
        auto_index=False,
        vector_manager=vector_mgr,
        llm=chat_model,
    )
    settings = Settings(PIPELINE_VERSION="dense-v1")
    pipeline = RAGPipeline(settings=settings, engine=engine)

    events = list(pipeline.stream("What leadership experience is listed?", request_id="req_test_123"))

    assert len(events) >= 4
    event_names = [e.event for e in events]
    assert event_names[0] == "retrieval_started"
    assert event_names[1] == "retrieval_completed"
    assert event_names[2] == "generation_started"
    assert "verification_started" in event_names
    assert event_names[-1] == "answer_completed"

    # Verify monotonic sequences and request ID
    for i, event in enumerate(events, 1):
        assert isinstance(event, PipelineEvent)
        assert event.sequence == i
        assert event.request_id == "req_test_123"
        assert event.schema_version == "1"

    # Verify retrieval_completed payload
    assert events[1].data == {"evidence_count": 1, "degraded_reason_codes": []}

    # Verify deltas reconstruct answer
    deltas = [e.data["delta"] for e in events if e.event == "answer_delta"]
    reconstructed = "".join(deltas)
    assert reconstructed == "Led data engineering projects at TechCorp [S1]."

    # Verify terminal event payload
    terminal = events[-1]
    assert terminal.data["replace_streamed_answer"] is False
    res = terminal.data["result"]
    assert isinstance(res, AnswerResult)
    assert res.decision == "answered"
    assert res.answer == "Led data engineering projects at TechCorp [S1]."
    assert len(res.citations) == 1
    assert res.citations[0].source == "resume.md"
    assert res.citations[0].page == 2


def test_stream_empty_retrieval_emits_terminal_answer_without_generation():
    vector_mgr = StubVectorManager([])

    class FailIfInvokedChatModel(FakeChatModel):
        def _stream(self, messages, **kwargs):
            raise AssertionError("LLM stream should not be invoked for empty retrieval")
            yield

    engine = CVChatEngine(
        auto_index=False,
        vector_manager=vector_mgr,
        llm=FailIfInvokedChatModel(),
    )
    pipeline = RAGPipeline(settings=Settings(), engine=engine)

    events = list(pipeline.stream("Tell me about astronomy skills", request_id="req_empty_test"))

    assert len(events) == 3
    assert [e.event for e in events] == [
        "retrieval_started",
        "retrieval_completed",
        "answer_completed",
    ]

    assert events[0].sequence == 1
    assert events[1].sequence == 2
    assert events[1].data == {"evidence_count": 0, "degraded_reason_codes": []}

    assert events[2].sequence == 3
    assert events[2].data["replace_streamed_answer"] is False
    res = events[2].data["result"]
    assert isinstance(res, AnswerResult)
    assert res.decision == "abstained"
    assert res.reason_codes == ("no_evidence",)
    assert res.answer == "I couldn't find that information in the CV."


def test_stream_mid_stream_exception_emits_terminal_error_event():
    doc = Document(
        page_content="Senior developer skills.",
        metadata={"source": "cv.txt", "page": 0, "chunk_id": "c1"},
    )
    vector_mgr = StubVectorManager([doc])
    chat_model = FakeChatModel(
        responses=["First token Second token Third token"],
        error_after_chunks=1,
    )
    engine = CVChatEngine(
        auto_index=False,
        vector_manager=vector_mgr,
        llm=chat_model,
    )
    pipeline = RAGPipeline(settings=Settings(), engine=engine)

    events = list(pipeline.stream("What skills are listed?", request_id="req_err_test"))

    event_names = [e.event for e in events]
    assert "retrieval_started" in event_names
    assert "retrieval_completed" in event_names
    assert "generation_started" in event_names
    assert "answer_delta" in event_names
    assert event_names[-1] == "error"
    assert "answer_completed" not in event_names

    terminal = events[-1]
    assert terminal.data["code"] == "stream_interrupted"
    assert terminal.data["message"] == "The stream was interrupted."
    assert terminal.data["retryable"] is False
    assert terminal.data["partial_answer"] == "First "


def test_question_validation_and_sanitization():
    pipeline = RAGPipeline(settings=Settings(MAX_QUESTION_CHARS=50))

    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline.answer("")

    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline.answer("   \n\t  ")

    with pytest.raises(ValueError, match="at most 50 characters"):
        pipeline.answer("A" * 51)


def test_history_normalization_and_bounding():
    doc = Document(page_content="Python details", metadata={"source": "cv.md", "page": 0})
    vector_mgr = StubVectorManager([doc])
    chat_model = FakeChatModel(responses=["Answer [S1]"])
    engine = CVChatEngine(
        auto_index=False,
        vector_manager=vector_mgr,
        llm=chat_model,
    )
    settings = Settings(MAX_HISTORY_TURNS=3)
    pipeline = RAGPipeline(settings=settings, engine=engine)

    # 1. Accepts up to 20 turns, bounds to MAX_HISTORY_TURNS (3)
    twenty_turns = [
        ChatTurn(user=f"user_{i}", assistant=f"assistant_{i}")
        for i in range(20)
    ]
    pipeline.answer("Recent question", history=twenty_turns)

    # Inspect LLM recorded calls to verify history truncation
    last_call = chat_model.calls[-1]
    message_contents = [getattr(m, "content", "") for m in last_call]
    # Should only contain turns 17, 18, 19
    assert any("user_17" in c for c in message_contents)
    assert any("user_18" in c for c in message_contents)
    assert any("user_19" in c for c in message_contents)
    assert not any("user_0" in c for c in message_contents)

    # 2. More than 20 turns raises ValueError
    twenty_one_turns = [
        ChatTurn(user=f"u_{i}", assistant=f"a_{i}")
        for i in range(21)
    ]
    with pytest.raises(ValueError, match="20 turns"):
        pipeline.answer("Question", history=twenty_one_turns)

    # 3. Tuple and dict formats accepted
    tuple_history = [("How are you?", "Fine."), ("What's up?", "Nothing.")]
    pipeline.answer("Another question", history=tuple_history)

    # 4. Turn content character limits (user <= 2000, assistant <= 4000)
    huge_turn = [
        ChatTurn(user="U" * 3000, assistant="A" * 5000)
    ]
    pipeline.answer("Question with huge turn", history=huge_turn)
    last_call = chat_model.calls[-1]
    for msg in last_call:
        if getattr(msg, "type", None) == "human" and "UUU" in getattr(msg, "content", ""):
            assert len(getattr(msg, "content", "")) <= 2000
        if getattr(msg, "type", None) == "ai" and "AAA" in getattr(msg, "content", ""):
            assert len(getattr(msg, "content", "")) <= 4000


def test_readiness_reporting():
    # Ready state when vector manager has documents
    ready_mgr = StubVectorManager([Document(page_content="test")])
    ready_engine = CVChatEngine(auto_index=False, vector_manager=ready_mgr, llm=FakeChatModel())
    ready_pipeline = RAGPipeline(settings=Settings(PIPELINE_VERSION="dense-v1"), engine=ready_engine)
    res_ready = ready_pipeline.readiness()
    assert isinstance(res_ready, ReadinessResult)
    assert res_ready.state == "ready"
    assert res_ready.is_ready is True
    assert res_ready.reason_codes == ()
    assert res_ready.pipeline_version == "dense-v1"

    # Not ready state when vector manager is empty
    empty_mgr = StubVectorManager([])
    empty_engine = CVChatEngine(auto_index=False, vector_manager=empty_mgr, llm=FakeChatModel())
    empty_pipeline = RAGPipeline(settings=Settings(PIPELINE_VERSION="dense-v1"), engine=empty_engine)
    res_empty = empty_pipeline.readiness()
    assert res_empty.state == "not_ready"
    assert res_empty.is_ready is False
    assert res_empty.reason_codes == ("index_empty",)

    # Not ready state on exception
    class BrokenVectorManager:
        def get_document_count(self):
            raise RuntimeError("Database connection failure")

    broken_engine = CVChatEngine(auto_index=False, vector_manager=BrokenVectorManager(), llm=FakeChatModel())
    broken_pipeline = RAGPipeline(settings=Settings(PIPELINE_VERSION="dense-v1"), engine=broken_engine)
    res_broken = broken_pipeline.readiness()
    assert res_broken.state == "not_ready"
    assert res_broken.is_ready is False
    assert res_broken.reason_codes == ("pipeline_not_ready",)


def test_offline_pipeline_instantiation_with_provider_factories():
    factories = FakeProviderFactories(
        responses=["Deterministic response [S1]."]
    )
    settings = Settings(PIPELINE_VERSION="dense-v1")
    pipeline = RAGPipeline(settings=settings, provider_factories=factories)

    assert pipeline.settings.PIPELINE_VERSION == "dense-v1"
