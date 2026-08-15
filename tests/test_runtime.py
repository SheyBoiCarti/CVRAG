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
