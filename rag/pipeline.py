"""Unified RAG pipeline facade exposing synchronous and streaming answer interfaces."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, Sequence
import uuid

from langchain_core.output_parsers import StrOutputParser

from rag.document_loader import CVDocumentLoader
from rag.ingestion.indexer import IndexSynchronizer
from rag.persistence.chroma_store import ChromaStoreAdapter
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.reranker import NvidiaReranker, NoOpReranker
from rag.retrieval.service import RetrievalService
from rag.engine import CVChatEngine, NO_RESULTS_ANSWER
from rag.errors import ProviderUnavailableError
from rag.models import AnswerResult, ChatTurn, Citation, PipelineEvent, ReadinessResult
from rag.provider_factories import LiveProviderFactories, ProviderFactories
from rag.settings import Settings, get_settings
from rag.vector_store import VectorStoreManager, is_transient_error, retry_transient

LOGGER = logging.getLogger(__name__)


class RAGPipeline:
    """Facade orchestrating retrieval, query reformulation, answer generation, and event streaming."""

    def __init__(
        self,
        settings: Settings | None = None,
        engine: CVChatEngine | None = None,
        provider_factories: ProviderFactories | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider_factories = provider_factories
        self._engine = engine

        if self._engine is None:
            try:
                self._engine = self._build_engine()
            except Exception as exc:
                LOGGER.debug(
                    "Deferred engine construction during pipeline initialization (type=%s)",
                    type(exc).__name__,
                )
                self._engine = None

    def _build_engine(self) -> CVChatEngine:
        """Constructs an engine instance wired to the configured settings and providers."""
        factories = self.provider_factories or LiveProviderFactories()
        embeddings = factories.create_embeddings(self.settings)
        llm = factories.create_chat_model(self.settings)
        active_index = IndexSynchronizer(settings=self.settings, embeddings=embeddings).ensure_active_index()
        vector_manager = VectorStoreManager(
            persist_dir=self.settings.VECTOR_DB_DIR,
            embeddings=embeddings,
            collection_name=active_index.collection_name,
            settings=self.settings,
        )
        reranker = NoOpReranker()
        if self.settings.RERANK_ENABLED:
            reranker = NvidiaReranker(factories.create_reranker(self.settings))
        retrieval_service = RetrievalService(
            settings=self.settings,
            dense_retriever=DenseRetriever(ChromaStoreAdapter(
                persist_dir=self.settings.VECTOR_DB_DIR, collection_name=active_index.collection_name, embeddings=embeddings,
            )),
            sparse_retriever=BM25Retriever(active_index.chunks),
            reranker=reranker,
            index_fingerprint=active_index.index_fingerprint,
        )
        loader = CVDocumentLoader(
            documents_dir=self.settings.DOCUMENTS_DIR,
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            settings=self.settings,
        )
        return CVChatEngine(
            auto_index=False,
            vector_manager=vector_manager,
            loader=loader,
            llm=llm,
            settings=self.settings,
            retrieval_service=retrieval_service,
        )

    def _get_engine(self) -> CVChatEngine:
        """Returns the active engine, constructing it on-demand if deferred."""
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    @property
    def engine(self) -> CVChatEngine:
        """Returns the active underlying CVChatEngine."""
        return self._get_engine()

    def prepare_query(
        self,
        question: str,
        chat_history: Sequence[ChatTurn | tuple[str, str]] = (),
        k: int | None = None,
    ):
        """Prepares retrieval context and reformulation for a question."""
        sanitized_question = self.settings.validate_question(question)
        bounded_history = self._normalize_history(chat_history)
        effective_k = k if k is not None else self.settings.RETRIEVAL_K
        return self._get_engine().prepare_query(
            sanitized_question,
            bounded_history,
            k=effective_k,
        )

    def stream_prepared(self, prepared: Any) -> Iterator[str]:
        """Streams token chunks for a prepared query."""
        return self._get_engine().stream_prepared(prepared)

    def cited_documents(self, answer: str, documents: Sequence[Any]) -> list[Any]:
        """Identifies cited documents from answer text."""
        return self._get_engine().cited_documents(answer, documents)

    def _normalize_history(
        self,
        history: Sequence[ChatTurn | tuple[str, str] | dict[str, str]] = (),
    ) -> list[tuple[str, str]]:
        """Validates, sanitizes, and bounds conversational history to recent turns."""
        if not history:
            return []
        if len(history) > 20:
            raise ValueError(
                f"History cannot exceed 20 turns (received {len(history)})."
            )

        normalized: list[tuple[str, str]] = []
        for turn in history:
            if isinstance(turn, ChatTurn):
                user_text, assistant_text = turn.user, turn.assistant
            elif isinstance(turn, (tuple, list)) and len(turn) == 2:
                user_text, assistant_text = str(turn[0]), str(turn[1])
            elif isinstance(turn, dict) and "user" in turn and "assistant" in turn:
                user_text, assistant_text = str(turn["user"]), str(turn["assistant"])
            else:
                raise ValueError(f"Invalid history turn format: {turn!r}")

            normalized.append(
                (str(user_text)[:2000], str(assistant_text)[:4000])
            )

        return normalized[-self.settings.MAX_HISTORY_TURNS :]

    def _build_citations(self, cited_docs: Sequence[Any]) -> tuple[Citation, ...]:
        """Constructs structured, 1-based Citation models with sanitized source basenames."""
        citations: list[Citation] = []
        for i, doc in enumerate(cited_docs, 1):
            evidence_id = str(doc.metadata.get("evidence_id") or f"S{i}")
            chunk_id = str(doc.metadata.get("chunk_id") or "")
            source_raw = str(doc.metadata.get("source") or "CV")
            source_basename = Path(source_raw).name
            raw_page = doc.metadata.get("page", 0)
            page = int(raw_page) + 1
            excerpt = doc.page_content.strip()
            citations.append(
                Citation(
                    evidence_id=evidence_id,
                    chunk_id=chunk_id,
                    source=source_basename,
                    page=page,
                    excerpt=excerpt,
                    support_status="not_checked",
                )
            )
        return tuple(citations)

    def answer(
        self,
        question: str,
        history: Sequence[ChatTurn | tuple[str, str]] = (),
    ) -> AnswerResult:
        """Synchronously answers a user question grounded in retrieved CV evidence."""
        sanitized_question = self.settings.validate_question(question)
        bounded_history = self._normalize_history(history)
        engine = self._get_engine()
        try:
            prepared = engine.prepare_query(
                sanitized_question,
                bounded_history,
                k=self.settings.RETRIEVAL_K,
            )
        except Exception as exc:
            LOGGER.warning("Retrieval preparation failed (type=%s)", type(exc).__name__)
            if is_transient_error(exc):
                raise ProviderUnavailableError() from exc
            raise

        if not prepared.documents:
            return AnswerResult(
                answer=NO_RESULTS_ANSWER,
                citations=(),
                claim_verifications=(),
                decision="abstained",
                reason_codes=("no_evidence",),
                retrieval_query=prepared.retrieval_query,
                pipeline_version=self.settings.PIPELINE_VERSION,
            )

        chain = engine.prompt | engine.llm | StrOutputParser()
        payload = {
            "context": prepared.context,
            "chat_history": prepared.history_messages,
            "question": prepared.question,
        }
        try:
            raw_answer = retry_transient(
                lambda: chain.invoke(payload),
                self.settings.MAX_RETRIES,
                operation_name="chat_generation",
                model=str(self.settings.LLM_MODEL),
            )
        except Exception as exc:
            if is_transient_error(exc):
                raise ProviderUnavailableError() from exc
            raise

        cited_docs = engine.cited_documents(raw_answer, prepared.documents)
        citations = self._build_citations(cited_docs)

        return AnswerResult(
            answer=raw_answer,
            citations=citations,
            claim_verifications=(),
            decision="answered",
            reason_codes=(),
            retrieval_query=prepared.retrieval_query,
            pipeline_version=self.settings.PIPELINE_VERSION,
        )

    def stream(
        self,
        question: str,
        history: Sequence[ChatTurn | tuple[str, str]] = (),
        request_id: str | None = None,
    ) -> Iterator[PipelineEvent]:
        """Streams fine-grained lifecycle events and token deltas for a user question."""
        sanitized_question = self.settings.validate_question(question)
        bounded_history = self._normalize_history(history)
        req_id = request_id or uuid.uuid4().hex

        return self._stream_generator(
            question=sanitized_question,
            history=bounded_history,
            request_id=req_id,
        )

    def _stream_generator(
        self,
        question: str,
        history: list[tuple[str, str]],
        request_id: str,
    ) -> Iterator[PipelineEvent]:
        """Generator yielding monotonic events according to Schema Version 1 contract."""
        seq = 1

        # 1. Retrieval started
        yield PipelineEvent.create(
            "retrieval_started",
            {},
            request_id=request_id,
            sequence=seq,
        )
        seq += 1

        # Retrieve documents
        try:
            engine = self._get_engine()
            prepared = engine.prepare_query(
                question,
                history,
                k=self.settings.RETRIEVAL_K,
            )
        except Exception as exc:
            LOGGER.warning("Retrieval preparation failed (type=%s)", type(exc).__name__)
            yield PipelineEvent.create(
                "error",
                {
                    "code": "retrieval_unavailable" if is_transient_error(exc) else "stream_interrupted",
                    "message": "Evidence retrieval failed." if is_transient_error(exc) else "The stream was interrupted.",
                    "retryable": is_transient_error(exc),
                    "partial_answer": None,
                },
                request_id=request_id,
                sequence=seq,
            )
            return

        evidence_count = len(prepared.documents)
        # 2. Retrieval completed
        yield PipelineEvent.create(
            "retrieval_completed",
            {"evidence_count": evidence_count, "degraded_reason_codes": []},
            request_id=request_id,
            sequence=seq,
        )
        seq += 1

        # Empty retrieval abstention: no generation_started, no LLM calls
        if evidence_count == 0:
            result = AnswerResult(
                answer=NO_RESULTS_ANSWER,
                citations=(),
                claim_verifications=(),
                decision="abstained",
                reason_codes=("no_evidence",),
                retrieval_query=prepared.retrieval_query,
                pipeline_version=self.settings.PIPELINE_VERSION,
            )
            yield PipelineEvent.create(
                "answer_completed",
                {"result": result, "replace_streamed_answer": False},
                request_id=request_id,
                sequence=seq,
            )
            return

        # 3. Generation started
        yield PipelineEvent.create(
            "generation_started",
            {},
            request_id=request_id,
            sequence=seq,
        )
        seq += 1

        # Stream generation tokens
        accumulated_answer = ""
        try:
            for token in engine.stream_prepared(prepared):
                if token:
                    accumulated_answer += token
                    yield PipelineEvent.create(
                        "answer_delta",
                        {"delta": token},
                        request_id=request_id,
                        sequence=seq,
                    )
                    seq += 1
        except Exception as exc:
            LOGGER.warning("Generation stream interrupted (type=%s)", type(exc).__name__)
            yield PipelineEvent.create(
                "error",
                {
                    "code": "stream_interrupted",
                    "message": "The stream was interrupted.",
                    "retryable": False,
                    "partial_answer": accumulated_answer or None,
                },
                request_id=request_id,
                sequence=seq,
            )
            return

        # Terminal answer_completed
        cited_docs = engine.cited_documents(accumulated_answer, prepared.documents)
        citations = self._build_citations(cited_docs)
        result = AnswerResult(
            answer=accumulated_answer,
            citations=citations,
            claim_verifications=(),
            decision="answered",
            reason_codes=(),
            retrieval_query=prepared.retrieval_query,
            pipeline_version=self.settings.PIPELINE_VERSION,
        )
        yield PipelineEvent.create(
            "answer_completed",
            {"result": result, "replace_streamed_answer": False},
            request_id=request_id,
            sequence=seq,
        )

    def readiness(self) -> ReadinessResult:
        """Evaluates pipeline health, initialization status, and vector index population."""
        try:
            engine = self._get_engine()
            count = engine.vector_manager.get_document_count()
            if count > 0:
                return ReadinessResult(
                    state="ready",
                    reason_codes=(),
                    pipeline_version=self.settings.PIPELINE_VERSION,
                    index_fingerprint=(engine.retrieval_service.index_fingerprint if engine.retrieval_service is not None else None),
                )
            return ReadinessResult(
                state="not_ready",
                reason_codes=("index_empty",),
                pipeline_version=self.settings.PIPELINE_VERSION,
                index_fingerprint=None,
            )
        except Exception as exc:
            LOGGER.warning(
                "Readiness check failed (type=%s)",
                type(exc).__name__,
            )
            return ReadinessResult(
                state="not_ready",
                reason_codes=("pipeline_not_ready",),
                pipeline_version=self.settings.PIPELINE_VERSION,
                index_fingerprint=None,
            )


__all__ = ["RAGPipeline"]
