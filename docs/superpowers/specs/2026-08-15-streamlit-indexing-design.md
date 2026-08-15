# Streamlit Cold-Start Indexing Fix Design

## Problem and confirmed cause

The application succeeds locally because the local checkout already has a populated, ignored `vector_db/`. `CVChatEngine(auto_index=True)` calls `ensure_indexed()`, but the existing collection count is 20, so local startup does not call NVIDIA document embeddings.

Streamlit Community Cloud clones only tracked files. Because `vector_db/` is ignored and untracked, each fresh deployment starts with an empty Chroma collection and embeds the tracked CV during startup. Streamlit Secrets currently sets `EMBEDDING_MODEL="nvidia/nv-embedcode-7b-v1"`, overriding the repository default `nvidia/nv-embedqa-e5-v5`. A live diagnostic request reproduced NVIDIA's generic HTTP 500 with the overridden model identifier, while the repository-default model successfully embedded a diagnostic string and completed a fresh 20-chunk CV index plus retrieval.

The cloud log's five retry attempts also match the `MAX_RETRIES=5` secret rather than the repository default of three, independently proving that Streamlit Secrets are overriding repository configuration. Python 3.14 is not causal: the same pinned dependencies on Python 3.14.6 completed the successful cold-start test.

## Scope

The fix will preserve the current LangChain, NVIDIA Embeddings, and persistent local Chroma architecture. It will:

- make the required Streamlit Secrets change explicit;
- safely expose the provider status and error summary in server logs;
- avoid repeated engine creation and indexing work across Streamlit reruns and sessions in one process;
- serialize and double-check cold-start indexing inside the process;
- show visitors a stable, non-sensitive startup failure message; and
- add focused regression tests.

It will not commit Chroma data, replace NVIDIA, change retrieval behavior, broadly refactor the RAG engine, or change the Python runtime.

## Startup architecture

`app.py` will obtain the engine through a function decorated with `st.cache_resource`. The cached resource will construct `CVChatEngine(auto_index=True)` once per Streamlit server process. The engine is safe to share because conversation history remains in `st.session_state` and the engine holds no user-specific mutable state.

`CVChatEngine.ensure_indexed()` will retain the fast collection-count check. When empty, it will acquire a module-level process lock, check the collection count again, load chunks, and add them. The second check prevents two simultaneous sessions from both embedding the CV during first startup. A new cloud container may rebuild once because Community Cloud storage is not a durable deployment artifact; ordinary reruns and additional sessions will reuse the cached resource and populated collection.

The collection name remains derived from the schema version and embedding model. Thus a model change selects a different collection instead of mixing incompatible vector dimensions. Document-content invalidation continues to use the project's existing schema-version policy.

## Provider diagnostics and errors

`rag/vector_store.py` will add a small diagnostic formatter that extracts an HTTP status from an exception attribute or the NVIDIA integration's `[500]` message format. It will normalize and bound the message, then redact likely bearer tokens, NVIDIA API keys, and key-value secret patterns.

Retry logs will include:

- operation (`embed_documents`, `embed_query`, or generation where used);
- configured model identifier;
- attempt and maximum attempts;
- exception type;
- parsed status when available; and
- sanitized provider summary.

Logs will never include request headers, API keys, input texts, CV contents, embeddings, or complete response objects. The final exhausted attempt will also be logged before the original exception is re-raised with its traceback relationship intact.

Startup will catch the internal exception, log the safe diagnostic, and show a user-facing message stating that the AI index could not be initialized and that the deployment configuration/provider logs should be checked. The browser will not receive provider response bodies or secrets.

## Configuration and deployment

The repository default remains:

```toml
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
```

The Streamlit Cloud secret must be changed to that value or the `EMBEDDING_MODEL` override must be removed so the repository default applies. The existing `NVIDIA_API_KEY` remains in Streamlit Secrets. `DOCUMENTS_DIR`, `VECTOR_DB_DIR`, timeout, batch size, and LLM settings do not need to change.

No runtime pin will be added. `requirements.txt` already pins the application dependencies, installation succeeds, `pip check` reports no broken requirements, and the relevant cold-start path works under Python 3.14 locally.

## Testing and verification

Tests will be added before implementation for:

- parsing status and retaining a useful provider summary;
- redacting key and bearer-token shapes from diagnostics;
- logging useful retry fields without sensitive inputs;
- not retrying a permanent 4xx response;
- double-checked indexing when another initializer populates the collection;
- one index build for concurrent in-process initialization; and
- process-cached engine construction where it can be tested without running a Streamlit server.

Verification will include the complete offline test suite, compilation, dependency validation, existing-index startup, and another fresh temporary Chroma index using the supported NVIDIA embedding model. The cold-start check will assert an initial count of zero, a final count matching loaded chunks, and successful similarity retrieval without printing CV text or vectors.

## Operational risks

Community Cloud filesystem data can be recreated when the app restarts or is redeployed, so one embedding rebuild per new container remains expected. Multiple independent server processes would each have their own Streamlit resource cache and Python lock; Chroma's shared persistence and deterministic IDs reduce duplication, but strict cross-process exactly-once behavior would require a filesystem lock or external vector service and is outside this deployment's demonstrated needs.

Provider availability and quota limits remain external risks. Bounded transient retries remain in place, now with enough safe context to distinguish rate limits, authentication errors, invalid models, and upstream failures.
