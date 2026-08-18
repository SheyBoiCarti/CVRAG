from __future__ import annotations

from rag import runtime


def test_get_engine_reuses_initialized_resource(monkeypatch):
    created = []

    class FakeEngine:
        def __init__(self, auto_index):
            created.append(auto_index)

    monkeypatch.setattr(runtime, "CVChatEngine", FakeEngine)
    runtime.get_engine.clear()
    try:
        first = runtime.get_engine("llm", "embedding", "collection")
        second = runtime.get_engine("llm", "embedding", "collection")

        assert first is second
        assert created == [True]
    finally:
        runtime.get_engine.clear()


def test_get_pipeline_reuses_initialized_resource(monkeypatch):
    created = []

    class FakePipeline:
        def __init__(self, settings=None):
            self.settings = settings
            created.append(self)

    monkeypatch.setattr(runtime, "RAGPipeline", FakePipeline)
    runtime.get_pipeline.clear()
    try:
        first = runtime.get_pipeline("llm", "embedding", "collection")
        second = runtime.get_pipeline("llm", "embedding", "collection")

        assert first is second
        assert len(created) == 1
    finally:
        runtime.get_pipeline.clear()
