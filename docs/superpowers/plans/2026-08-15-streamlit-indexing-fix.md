# Streamlit Cold-Start Indexing Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Streamlit cold-start indexing usefully diagnosable, execute once per server process, and recover cleanly from provider failures while retaining NVIDIA embeddings and Chroma.

**Architecture:** Keep the versioned persistent Chroma collection and wrap engine construction in a Streamlit process-wide resource cache. Serialize the empty-index transition with a double-checked lock, and centralize bounded provider diagnostics so retries expose status and safe error context without logging payloads or credentials.

**Tech Stack:** Python 3.14, Streamlit 1.61.1, LangChain, `langchain-nvidia-ai-endpoints` 1.4.3, Chroma 1.5.9, pytest.

## Global Constraints

- Preserve the existing RAG architecture and retrieval behavior.
- Keep `vector_db/` ignored and untracked.
- Do not log API keys, authorization headers, document contents, request payloads, or embedding vectors.
- Keep retries bounded and retry only connection failures, timeouts, HTTP 429, and HTTP 5xx responses.
- Do not change the Python runtime or pinned dependency versions unless verification reveals a new incompatibility.
- The manual Streamlit setting must use `EMBEDDING_MODEL="nvidia/nv-embedqa-e5-v5"` or omit that override.

---

### Task 1: Safe provider diagnostics

**Files:**
- Modify: `rag/vector_store.py`
- Modify: `tests/test_vector_store.py`

**Interfaces:**
- Produces: `provider_error_details(exc: Exception) -> tuple[int | None, str]`
- Changes: `retry_transient(operation, max_attempts: int, *, operation_name: str = "provider_request", model: str = "unknown")`
- Consumes: NVIDIA's generic exception format, including `[500] Something went wrong with the request.`

- [ ] **Step 1: Write failing diagnostic tests**

Add tests with literal expectations proving that an exception's bracketed status is extracted, whitespace is normalized, bearer/API-key shapes are redacted, and output is bounded:

```python
from rag.vector_store import provider_error_details


def test_provider_error_details_extracts_status_and_safe_summary():
    exc = Exception("[500] Something went wrong with the request.\n{'status': 500}")
    assert provider_error_details(exc) == (
        500,
        "[500] Something went wrong with the request. {'status': 500}",
    )


def test_provider_error_details_redacts_credentials():
    exc = Exception(
        "Authorization: Bearer secret-token NVIDIA_API_KEY=nvapi-secret-value"
    )
    status, detail = provider_error_details(exc)
    assert status is None
    assert "secret-token" not in detail
    assert "nvapi-secret-value" not in detail
    assert detail == "Authorization: Bearer [REDACTED] NVIDIA_API_KEY=[REDACTED]"
```

- [ ] **Step 2: Write a failing retry-log test**

Use `caplog` and patched sleep/jitter to prove the real retry helper logs operation, model, status, type, and sanitized detail but never the injected credential:

```python
def test_retry_log_has_safe_provider_context(monkeypatch, caplog):
    monkeypatch.setattr("rag.vector_store.time.sleep", lambda _delay: None)
    monkeypatch.setattr("rag.vector_store.random.uniform", lambda _a, _b: 0)

    def fail():
        raise Exception("[503] unavailable NVIDIA_API_KEY=nvapi-secret")

    with caplog.at_level("WARNING"), pytest.raises(Exception):
        retry_transient(
            fail,
            max_attempts=2,
            operation_name="embed_documents",
            model="nvidia/nv-embedqa-e5-v5",
        )

    assert "operation=embed_documents" in caplog.text
    assert "model=nvidia/nv-embedqa-e5-v5" in caplog.text
    assert "status=503" in caplog.text
    assert "nvapi-secret" not in caplog.text
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_vector_store.py -q --basetemp=.pytest_tmp/task1-red`

Expected: collection fails because `provider_error_details` is absent and `retry_transient` does not accept diagnostic keyword arguments.

- [ ] **Step 4: Implement minimal safe diagnostics**

In `rag/vector_store.py`, add compiled expressions for bracketed HTTP status and credential shapes, then implement:

```python
STATUS_PATTERN = re.compile(r"\[(\d{3})\]")
BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[^\s,;]+")
API_KEY_PATTERN = re.compile(
    r"(?i)((?:nvidia_)?api[_ -]?key\s*[:=]\s*)[^\s,;]+"
)
NVAPI_PATTERN = re.compile(r"\bnvapi-[A-Za-z0-9_-]+\b")
MAX_PROVIDER_DETAIL_CHARS = 300


def provider_error_details(exc: Exception) -> tuple[int | None, str]:
    raw_status = getattr(exc, "status_code", None)
    status = raw_status if isinstance(raw_status, int) else None
    detail = " ".join(str(exc).split())
    if status is None:
        match = STATUS_PATTERN.search(detail)
        status = int(match.group(1)) if match else None
    detail = BEARER_PATTERN.sub(r"\1[REDACTED]", detail)
    detail = API_KEY_PATTERN.sub(r"\1[REDACTED]", detail)
    detail = NVAPI_PATTERN.sub("[REDACTED]", detail)
    return status, detail[:MAX_PROVIDER_DETAIL_CHARS] or "no provider detail"
```

Update `is_transient_error()` to use the parsed status and retain timeout/connection/rate-limit name and phrase checks. Remove broad bare-number substring matching so unrelated text containing `500` is not classified as an HTTP failure.

Extend `retry_transient()` with keyword-only operation/model context. On retry, log at warning; on the final or permanent failure, log once at error. Both messages use only `provider_error_details()`, exception type, operation, model, and attempt counters.

Pass `operation_name` and `self.client.model` from both `RobustNVIDIAEmbeddings` methods. Pass generation context from `CVChatEngine.query()` without changing query semantics.

- [ ] **Step 5: Run focused and full tests and verify GREEN**

Run: `python -m pytest tests/test_vector_store.py tests/test_engine.py -q --basetemp=.pytest_tmp/task1-green`

Expected: all focused tests pass.

Run: `python -m pytest -q --basetemp=.pytest_tmp/task1-full`

Expected: 29 or more tests pass with no failures.

- [ ] **Step 6: Commit Task 1**

```powershell
git add rag/vector_store.py rag/engine.py tests/test_vector_store.py
git commit -m "fix: expose safe NVIDIA provider diagnostics"
```

### Task 2: Exactly-once in-process cold-start indexing

**Files:**
- Modify: `rag/engine.py`
- Modify: `tests/test_engine.py`

**Interfaces:**
- Changes: `CVChatEngine.ensure_indexed() -> int` retains its public signature.
- Produces: one document load/add transition when concurrent engines share an empty collection in one process.

- [ ] **Step 1: Write a failing double-check test**

Add a fake manager whose first count is empty and second count is populated, plus a loader that must not run:

```python
def test_ensure_indexed_rechecks_after_acquiring_lock():
    class ManagerPopulatedByAnotherInitializer:
        counts = iter([0, 7])

        def get_document_count(self):
            return next(self.counts)

        def ensure_documents(self, chunks):
            raise AssertionError("documents must not be embedded twice")

    class LoaderThatMustNotRun:
        def load_and_split(self):
            raise AssertionError("documents must not be loaded after the recheck")

    engine = object.__new__(CVChatEngine)
    engine.vector_manager = ManagerPopulatedByAnotherInitializer()
    engine.loader = LoaderThatMustNotRun()

    assert engine.ensure_indexed() == 7
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_engine.py::test_ensure_indexed_rechecks_after_acquiring_lock -q`

Expected: FAIL because the current method loads and calls `ensure_documents()` immediately after the first empty count.

- [ ] **Step 3: Implement the double-checked index lock**

Import `threading`, create `_INDEX_LOCK = threading.Lock()` at module scope, and implement:

```python
def ensure_indexed(self) -> int:
    count = self.vector_manager.get_document_count()
    if count > 0:
        return count
    with _INDEX_LOCK:
        count = self.vector_manager.get_document_count()
        if count > 0:
            return count
        chunks = self.loader.load_and_split()
        return self.vector_manager.ensure_documents(chunks)
```

- [ ] **Step 4: Run focused and full tests and verify GREEN**

Run: `python -m pytest tests/test_engine.py -q --basetemp=.pytest_tmp/task2-green`

Run: `python -m pytest -q --basetemp=.pytest_tmp/task2-full`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add rag/engine.py tests/test_engine.py
git commit -m "fix: serialize cold-start vector indexing"
```

### Task 3: Cache the engine across Streamlit sessions and present safe startup errors

**Files:**
- Create: `rag/runtime.py`
- Create: `tests/test_runtime.py`
- Modify: `app.py`

**Interfaces:**
- Produces: `get_engine(llm_model: str, embedding_model: str, collection_name: str) -> CVChatEngine`
- Consumes: the effective model and collection strings as cache keys.
- Changes: `app.py` stores only chat history in session state and obtains the shared engine from `get_engine()`.

- [ ] **Step 1: Write a failing cached-resource behavior test**

Create `tests/test_runtime.py`. Patch only the external engine construction and assert the returned real cache behavior rather than mock call metadata:

```python
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
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_runtime.py -q`

Expected: collection/import failure because `rag.runtime` does not exist.

- [ ] **Step 3: Implement the cached engine factory**

Create `rag/runtime.py`:

```python
import streamlit as st

from rag.engine import CVChatEngine


@st.cache_resource(show_spinner=False)
def get_engine(
    llm_model: str,
    embedding_model: str,
    collection_name: str,
) -> CVChatEngine:
    del llm_model, embedding_model, collection_name
    return CVChatEngine(auto_index=True)
```

The arguments deliberately participate in Streamlit's cache key so model/schema changes create a new engine even though construction reads the effective values from `Config`.

- [ ] **Step 4: Replace session-local engine creation and add a friendly failure state**

In `app.py`, import `logging`, `get_engine`, and `provider_error_details`. Define `LOGGER`. Replace the `st.session_state.engine` block with:

```python
try:
    with st.spinner("Initializing AI engine..."):
        engine = get_engine(
            Config.LLM_MODEL or "",
            Config.EMBEDDING_MODEL or "",
            Config.COLLECTION_NAME,
        )
except Exception as exc:
    status, detail = provider_error_details(exc)
    LOGGER.error(
        "AI engine initialization failed (embedding_model=%s, status=%s, "
        "type=%s, detail=%s)",
        Config.EMBEDDING_MODEL,
        status if status is not None else "unknown",
        type(exc).__name__,
        detail,
    )
    st.error(
        "The AI search index could not be initialized. "
        "Please try again later; the deployment owner can check the server logs."
    )
    st.stop()
```

Do not render `str(exc)` to the user.

- [ ] **Step 5: Run focused and full tests and verify GREEN**

Run: `python -m pytest tests/test_runtime.py tests/test_engine.py tests/test_vector_store.py -q --basetemp=.pytest_tmp/task3-green`

Run: `python -m pytest -q --basetemp=.pytest_tmp/task3-full`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git add app.py rag/runtime.py tests/test_runtime.py
git commit -m "fix: cache Streamlit engine initialization"
```

### Task 4: Deployment guidance and end-to-end verification

**Files:**
- Modify: `README.md`
- Review only: `.gitignore`, `.streamlit/config.toml`, `.env.example`, `requirements.in`, `requirements.txt`

**Interfaces:**
- Produces: exact operator instructions for removing the stale model override and understanding cold-start persistence.

- [ ] **Step 1: Update deployment documentation**

Document that Streamlit Secrets override `.env` and repository defaults, show only the supported embedding model, warn that `vector_db/` is intentionally not deployed, and explain that a fresh container performs one indexing pass. Add a troubleshooting entry directing operators to compare the effective model name/status in safe logs.

- [ ] **Step 2: Review deployment files without speculative changes**

Confirm:

- `.gitignore` still excludes `.env`, `.streamlit/secrets.toml`, and `vector_db/` while tracking `documents/Custom_CV.md`;
- `.streamlit/config.toml` contains only presentation/privacy settings relevant to the app;
- `.env.example` already uses `nvidia/nv-embedqa-e5-v5`;
- `requirements.txt` remains fully pinned and `requirements.in` retains compatible ranges; and
- no `runtime.txt` or Python downgrade is introduced because Python 3.14 passed the relevant path.

- [ ] **Step 3: Run offline verification**

Run:

```powershell
python -m compileall -q app.py rag tests test_pipeline.py ui
python -m pytest -q --basetemp=.pytest_tmp/final
python -m pip check
git check-ignore vector_db/chroma.sqlite3
git ls-files vector_db
```

Expected: compilation succeeds; all tests pass; dependencies are consistent; the ignore check succeeds; and `git ls-files vector_db` returns no paths.

- [ ] **Step 4: Verify an existing index avoids document embeddings**

Run a small diagnostic with a fake embeddings implementation and the same temporary Chroma directory twice. Assert the first engine creates the expected chunk count and the second `ensure_indexed()` returns that count without calling `embed_documents` again.

- [ ] **Step 5: Verify a live true cold start**

With the local `.env` credential and a newly created temporary directory outside `vector_db/`, construct `RobustNVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", ...)`, load the tracked CV chunks, index them once, and execute a similarity query. Print counts only, never text or vectors. Explicitly release Chroma references before removing the Windows temporary directory.

Expected: initial count 0; final count equals 20 (or the current loader count); retrieval returns at least one result; no provider retries occur.

- [ ] **Step 6: Review the final diff and commit documentation**

Run: `git diff main...HEAD --check` and `git diff main...HEAD --stat`.

Then:

```powershell
git add README.md
git commit -m "docs: clarify Streamlit embedding configuration"
```

- [ ] **Step 7: Request code review and apply verification-before-completion**

Use `superpowers:requesting-code-review`, address any substantiated findings, then use `superpowers:verification-before-completion` and rerun the final verification commands before reporting completion.
