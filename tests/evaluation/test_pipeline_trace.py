from evaluation.schema import EvaluationTrace
from rag.models import AnswerResult, ChatTurn
from rag.pipeline import RAGPipeline
from rag.settings import Settings
from tests.fakes import FakeChatModel, FakeEmbeddings, FakeProviderFactories


def test_pipeline_answer_with_trace_captures_all_stages():
    factories = FakeProviderFactories(
        embeddings=FakeEmbeddings(),
        chat_model=FakeChatModel(default_response="Sheharyar worked at Wirk [S1]."),
    )
    settings = Settings(
        RETRIEVAL_MODE="hybrid",
        RERANK_ENABLED=False,
    )
    pipeline = RAGPipeline(settings=settings, provider_factories=factories)

    ans, trace = pipeline.answer_with_trace("What role did Sheharyar hold at Wirk?", case_id="test-001")

    assert isinstance(ans, AnswerResult)
    assert ans.answer == "Sheharyar worked at Wirk [S1]."
    assert isinstance(trace, EvaluationTrace)
    assert trace.case_id == "test-001"
    assert trace.retrieval_query == "What role did Sheharyar hold at Wirk?"
    assert isinstance(trace.dense_results, tuple)
    assert isinstance(trace.sparse_results, tuple)
    assert isinstance(trace.fused_results, tuple)
    assert "total_ms" in trace.timings_ms
    assert trace.corpus_fingerprint != ""
    assert trace.index_fingerprint != ""
    assert trace.configuration["RETRIEVAL_MODE"] == "hybrid"


def test_pipeline_answer_with_trace_conversational_follow_up():
    factories = FakeProviderFactories(
        embeddings=FakeEmbeddings(),
        chat_model=FakeChatModel(default_response="He was a Software Engineer [S1]."),
    )
    settings = Settings(
        RETRIEVAL_MODE="dense",
        RERANK_ENABLED=False,
    )
    pipeline = RAGPipeline(settings=settings, provider_factories=factories)

    history = [ChatTurn("Where did he work?", "He worked at Wirk.")]
    ans, trace = pipeline.answer_with_trace("What did he do there?", history=history, case_id="followup-001")

    assert trace.case_id == "followup-001"
    assert len(trace.dense_results) > 0
    assert trace.configuration["RETRIEVAL_MODE"] == "dense"
