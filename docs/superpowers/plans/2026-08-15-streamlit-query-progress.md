# Streamlit Query Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distinguish CV retrieval from answer generation in Streamlit and terminate provider failures with safe, useful feedback.

**Architecture:** Promote query preparation and prepared-query streaming to public `CVChatEngine` operations, while retaining `query()` and `query_stream()` as compatible wrappers. The Streamlit app will place separate spinners around those operations, classify provider failures, and log only stage and exception type. The NVIDIA embedding wrapper will set the timeout on both underlying transport clients because its public constructor ignores the keyword.

**Tech Stack:** Python 3.10+, Streamlit 1.61.1, LangChain Core 1.5.4, langchain-nvidia-ai-endpoints 1.4.3, ChromaDB, pytest.

## Global Constraints

- Do not add background workers, asynchronous job infrastructure, provider switching, or retrieval-relevance changes.
- Preserve citations, conversation history, follow-up rewriting, deduplication, synchronous `query()`, and existing `query_stream()` callers.
- Do not log prompts, CV contents, exception messages, or secrets.
- Write every behavioral test before its production change and observe the expected failure.

---

### Task 1: Public staged-query engine interface

**Files:**
- Modify: `tests/test_engine.py`
- Modify: `rag/engine.py`

**Interfaces:**
- Produces: `CVChatEngine.prepare_query(question: str, chat_history: list[tuple[str, str]] | None = None, k: int | None = None) -> PreparedQuery`
- Produces: `CVChatEngine.stream_prepared(prepared: PreparedQuery) -> Iterator[str]`
- Preserves: `CVChatEngine.query_stream(...) -> tuple[Iterator[str], list[Document]]`

- [x] **Step 1: Write the failing staged-query tests**

Add a fake vector manager and fake streaming model, then assert that `prepare_query()` returns evidence without invoking the model and `stream_prepared()` invokes generation afterward. Also assert an empty prepared query returns `NO_RESULTS_ANSWER` without invoking the model.

```python
prepared = engine.prepare_query("Why should I hire this person?")
assert prepared.documents == [expected_document]
assert model.stream_calls == 0
assert "".join(engine.stream_prepared(prepared)) == "Strong evidence [S1]."
assert model.stream_calls == 1
```

- [x] **Step 2: Verify the focused tests fail for the missing public methods**

Run: `python -m pytest -q tests/test_engine.py -k "prepare_query or stream_prepared" --basetemp .pytest_tmp/engine-red`

Expected: FAIL with `AttributeError` for `prepare_query` or `stream_prepared`.

- [x] **Step 3: Implement the minimal public interface**

Rename `_prepare` to `prepare_query`, add defaults to its history and `k` parameters, and add `stream_prepared` that returns the no-results iterator or builds and streams the existing prompt chain. Update `query()` and `query_stream()` to reuse these methods.

- [x] **Step 4: Verify the engine tests pass**

Run: `python -m pytest -q tests/test_engine.py --basetemp .pytest_tmp/engine-green`

Expected: all engine tests PASS.

### Task 2: Propagate the configured embedding timeout

**Files:**
- Modify: `tests/test_vector_store.py`
- Modify: `rag/vector_store.py`

**Interfaces:**
- Preserves: `RobustNVIDIAEmbeddings(..., timeout: float = 45.0)`
- Guarantees: `wrapper.client._client.timeout == timeout` and `wrapper.client._async_client.timeout == timeout`

- [x] **Step 1: Write the failing transport-timeout test**

```python
embeddings = RobustNVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key="test-key",
    timeout=7.5,
)
assert embeddings.client._client.timeout == 7.5
assert embeddings.client._async_client.timeout == 7.5
```

- [x] **Step 2: Verify it fails with the current 60-second transport default**

Run: `python -m pytest -q tests/test_vector_store.py::test_embedding_timeout_reaches_transport_clients --basetemp .pytest_tmp/timeout-red`

Expected: FAIL because the actual timeout is `60`, not `7.5`.

- [x] **Step 3: Set both client timeouts after NVIDIA client construction**

After constructing `NVIDIAEmbeddings`, assign the validated `timeout` value to its synchronous and asynchronous private clients. Keep current batching and retry behavior unchanged.

- [x] **Step 4: Verify vector-store tests pass**

Run: `python -m pytest -q tests/test_vector_store.py --basetemp .pytest_tmp/timeout-green`

Expected: all vector-store tests PASS.

### Task 3: Safe provider-facing errors and separate Streamlit stages

**Files:**
- Modify: `tests/test_presentation.py`
- Modify: `rag/presentation.py`
- Modify: `app.py`

**Interfaces:**
- Produces: `query_error_message(exc: Exception) -> str`
- Uses: `CVChatEngine.prepare_query(...)` during `Searching CV...`
- Uses: `CVChatEngine.stream_prepared(prepared)` during `Generating answer...`

- [x] **Step 1: Write failing safe-message tests**

```python
assert query_error_message(ProviderUnavailableError()) == (
    "The AI provider is temporarily unavailable. Please try again shortly."
)
assert query_error_message(RuntimeError("secret detail")) == (
    "Couldn't generate a response right now. Please try asking your question again."
)
assert "secret detail" not in query_error_message(RuntimeError("secret detail"))
```

- [x] **Step 2: Verify the presentation tests fail for the missing helper**

Run: `python -m pytest -q tests/test_presentation.py --basetemp .pytest_tmp/presentation-red`

Expected: collection FAIL because `query_error_message` is not defined.

- [x] **Step 3: Implement the safe-message mapping**

Add the two constant messages and `query_error_message()` to `rag/presentation.py`, classifying only `ProviderUnavailableError` as temporary provider unavailability.

- [x] **Step 4: Rewire the Streamlit query block**

Import logging, `ProviderUnavailableError`, `NO_RESULTS_ANSWER`, and `query_error_message`. During retrieval set `stage = "retrieval"` and call `engine.prepare_query()` inside `Searching CV...`. For evidence-backed results set `stage = "generation"` and call `st.write_stream(engine.stream_prepared(prepared))` inside `Generating answer...`; otherwise write `NO_RESULTS_ANSWER`. On failure, log `stage` and `type(exc).__name__` only, render `query_error_message(exc)`, and clear sources.

- [x] **Step 5: Verify presentation and engine tests pass**

Run: `python -m pytest -q tests/test_presentation.py tests/test_engine.py --basetemp .pytest_tmp/ui-green`

Expected: all selected tests PASS.

### Task 4: Full verification and deployment diagnostic

**Files:**
- Verify: `app.py`
- Verify: `rag/engine.py`
- Verify: `rag/vector_store.py`
- Verify: `rag/presentation.py`
- Verify: `tests/`

- [x] **Step 1: Run syntax and offline tests**

Run: `python -m compileall -q app.py rag tests test_pipeline.py ui`

Run: `python -m pytest -q --basetemp .pytest_tmp/full`

Expected: compile command exits 0 and all tests PASS.

- [x] **Step 2: Check installed dependencies**

Run: `python -m pip check`

Expected: `No broken requirements found.`

- [x] **Step 3: Run the exact live question with phase timings**

Run a short Python diagnostic that constructs `CVChatEngine`, times `prepare_query("Why should I hire this person?")`, consumes `stream_prepared(prepared)`, and asserts the answer is non-empty.

Expected: retrieval and generation both complete; output reports separate durations and a non-zero answer length without printing document contents, prompts, or credentials.

- [x] **Step 4: Audit the diff**

Run: `git diff --check` and `git diff -- app.py rag tests`

Expected: no whitespace errors; only scoped staged-progress, error-handling, timeout, and test changes appear.

- [ ] **Step 5: Commit the implementation**

```powershell
git add -- app.py rag/engine.py rag/vector_store.py rag/presentation.py tests/test_engine.py tests/test_vector_store.py tests/test_presentation.py
git commit -m "fix: clarify deployed query progress"
```
