from __future__ import annotations

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from rag.settings import Settings


def test_provider_factories_protocol_conformance() -> None:
    """Verifies that LiveProviderFactories and FakeProviderFactories conform to ProviderFactories."""
    from rag.provider_factories import LiveProviderFactories, ProviderFactories
    from tests.fakes import FakeProviderFactories

    live_factory = LiveProviderFactories()
    fake_factory = FakeProviderFactories()

    assert isinstance(live_factory, ProviderFactories)
    assert isinstance(fake_factory, ProviderFactories)


def test_no_clients_constructed_at_import_or_instantiation() -> None:
    """Verifies that importing and instantiating LiveProviderFactories does not construct live clients."""
    from rag.provider_factories import LiveProviderFactories

    factory = LiveProviderFactories()
    assert factory is not None


def test_live_factory_requires_api_key_for_embeddings() -> None:
    """Verifies that LiveProviderFactories.create_embeddings raises ValueError when API key is missing."""
    from rag.provider_factories import LiveProviderFactories

    factory = LiveProviderFactories()
    settings = Settings(NVIDIA_API_KEY=None)

    with pytest.raises(ValueError, match="NVIDIA_API_KEY is not configured"):
        factory.create_embeddings(settings)


def test_live_factory_requires_api_key_for_chat_model() -> None:
    """Verifies that LiveProviderFactories.create_chat_model raises ValueError when API key is missing."""
    from rag.provider_factories import LiveProviderFactories

    factory = LiveProviderFactories()
    settings = Settings(NVIDIA_API_KEY=None)

    with pytest.raises(ValueError, match="NVIDIA_API_KEY is not configured"):
        factory.create_chat_model(settings)


def test_live_factory_creates_embeddings_with_settings() -> None:
    """Verifies that LiveProviderFactories.create_embeddings creates robust embeddings with settings."""
    from rag.provider_factories import LiveProviderFactories
    from rag.vector_store import RobustNVIDIAEmbeddings

    factory = LiveProviderFactories()
    settings = Settings(
        NVIDIA_API_KEY="nvapi-test-dummy-key-12345",
        EMBEDDING_MODEL="nvidia/nv-embedqa-e5-v5",
        EMBEDDING_BATCH_SIZE=12,
        MAX_RETRIES=2,
        REQUEST_TIMEOUT=30.0,
    )

    embeddings = factory.create_embeddings(settings)
    assert isinstance(embeddings, (Embeddings, RobustNVIDIAEmbeddings))
    assert getattr(embeddings, "batch_size", None) == 12
    assert getattr(embeddings, "max_retries", None) == 2


def test_live_factory_creates_chat_model_with_settings() -> None:
    """Verifies that LiveProviderFactories.create_chat_model creates ChatNVIDIA with settings."""
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    from rag.provider_factories import LiveProviderFactories

    factory = LiveProviderFactories()
    settings = Settings(
        NVIDIA_API_KEY="nvapi-test-dummy-key-12345",
        LLM_MODEL="meta/llama-3.3-70b-instruct",
        REQUEST_TIMEOUT=35.0,
    )

    chat_model = factory.create_chat_model(settings)
    assert isinstance(chat_model, (BaseChatModel, ChatNVIDIA))
    assert chat_model.model == "meta/llama-3.3-70b-instruct"


def test_live_factory_stubs_for_reranker_and_verifier() -> None:
    """Verifies that create_reranker and create_support_verifier return stubs or None for Phase 1."""
    from rag.provider_factories import LiveProviderFactories

    factory = LiveProviderFactories()
    settings = Settings(NVIDIA_API_KEY="nvapi-test")

    assert factory.create_reranker(settings) is None
    assert factory.create_support_verifier(settings) is None


def test_fake_embeddings_deterministic_and_dimensional() -> None:
    """Verifies that FakeEmbeddings produces deterministic, dimensionally correct float vectors."""
    from tests.fakes import FakeEmbeddings

    emb = FakeEmbeddings(dimension=64)

    # Identical texts yield identical vectors
    v1 = emb.embed_query("Experience with Python and FastAPI")
    v2 = emb.embed_query("Experience with Python and FastAPI")
    assert v1 == v2
    assert len(v1) == 64
    assert all(isinstance(x, float) for x in v1)

    # Different texts yield different vectors
    v3 = emb.embed_query("Completely unrelated topic about gardening")
    assert v1 != v3

    # Document batch embedding
    docs = ["Doc 1", "Doc 2", "Doc 3"]
    doc_vecs = emb.embed_documents(docs)
    assert len(doc_vecs) == 3
    assert all(len(vec) == 64 for vec in doc_vecs)
    assert doc_vecs[0] == emb.embed_query("Doc 1")


def test_fake_chat_model_invoke_and_stream() -> None:
    """Verifies that FakeChatModel supports invoke, stream, response sequencing, and call tracking."""
    from tests.fakes import FakeChatModel

    responses = [
        "Candidate has 5 years Python experience [S1].",
        "Candidate led backend microservices migration [S2].",
    ]
    model = FakeChatModel(responses=responses)

    # First invoke
    msg1 = model.invoke([HumanMessage(content="What is the candidate's Python experience?")])
    assert isinstance(msg1, AIMessage)
    assert msg1.content == "Candidate has 5 years Python experience [S1]."

    # Stream second response
    chunks = list(model.stream([HumanMessage(content="Tell me about system architecture.")]))
    assert all(isinstance(c, AIMessageChunk) for c in chunks)
    full_streamed = "".join(c.content for c in chunks)
    assert full_streamed == "Candidate led backend microservices migration [S2]."

    # Repeated calls return the last configured response
    msg3 = model.invoke([HumanMessage(content="Another question")])
    assert msg3.content == "Candidate led backend microservices migration [S2]."

    # Invocations recorded
    assert len(model.calls) == 3
    assert model.calls[0][0].content == "What is the candidate's Python experience?"


def test_fake_chat_model_dynamic_response_fn() -> None:
    """Verifies that FakeChatModel supports custom response generator functions."""
    from tests.fakes import FakeChatModel

    def echo_responder(messages: list) -> str:
        last_human = messages[-1].content
        return f"Echoing: {last_human} [S1]"

    model = FakeChatModel(response_fn=echo_responder)
    res = model.invoke("Hello RAG")
    assert res.content == "Echoing: Hello RAG [S1]"


def test_fake_chat_model_simulated_stream_error() -> None:
    """Verifies that FakeChatModel can simulate mid-stream provider failures."""
    from tests.fakes import FakeChatModel

    model = FakeChatModel(
        responses=["Chunk one Chunk two Chunk three"],
        error_after_chunks=1,
    )

    gen = model.stream("Test question")
    # First chunk succeeds
    chunk1 = next(gen)
    assert chunk1.content.strip() == "Chunk"

    # Second chunk triggers configured error
    with pytest.raises(RuntimeError, match="Simulated mid-stream provider error"):
        next(gen)


def test_fake_provider_factories_offline_creation() -> None:
    """Verifies that FakeProviderFactories creates offline components without requiring credentials."""
    from tests.fakes import FakeChatModel, FakeEmbeddings, FakeProviderFactories

    settings = Settings(NVIDIA_API_KEY=None)
    factory = FakeProviderFactories(
        responses=["Custom offline answer [S1]."],
        embedding_dimension=128,
    )

    embeddings = factory.create_embeddings(settings)
    assert isinstance(embeddings, (Embeddings, FakeEmbeddings))
    vec = embeddings.embed_query("test query")
    assert len(vec) == 128

    chat_model = factory.create_chat_model(settings)
    assert isinstance(chat_model, (BaseChatModel, FakeChatModel))
    resp = chat_model.invoke("test")
    assert resp.content == "Custom offline answer [S1]."

    reranker = factory.create_reranker(settings)
    verifier = factory.create_support_verifier(settings)
    assert reranker is not None
    assert verifier is not None
