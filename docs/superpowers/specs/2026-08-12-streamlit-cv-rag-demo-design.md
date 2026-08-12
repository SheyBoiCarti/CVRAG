# Streamlit CV RAG Demo Design

## Goal

Turn the existing fixed-CV LangChain application into a polished, reliable portfolio demo deployed on Streamlit Community Cloud. Keep the implementation intentionally small: one Streamlit application, one fixed CV, Chroma for retrieval, and NVIDIA endpoints for embeddings and generation.

## Scope

The demo answers questions about the owner's real CV. The CV is stored in a private GitHub repository for the initial deployment. Visitors cannot upload documents, select other candidates, change models, or administer the vector index.

This project will not introduce FastAPI, Redis, authentication, multi-tenancy, a background worker, Docker, or a separate frontend. Those are unnecessary for a fixed-document portfolio demo.

If the repository is made public later, the owner must intentionally accept that the committed CV is public. Removing the PDF in a later commit will not remove it from Git history; a clean public repository is required if the original PDF should remain private.

## Architecture

```text
Streamlit UI
    -> one process-cached CVChatEngine
        -> fixed CV document loader
        -> local Chroma index
        -> NVIDIA embedding endpoint
        -> NVIDIA chat endpoint
```

`st.cache_resource` will provide one engine per Streamlit process instead of one engine per browser session. Conversation messages remain session-specific. The index is created when absent and is not mutable through the public interface.

Streamlit Community Cloud may restart the application and does not provide durable application-local storage guarantees. Rebuilding this two-page CV index on a cold start is acceptable for the demo. The implementation must avoid rebuilding it on every Streamlit rerun or for every visitor session.

## User Interface

The public interface will use a centered, responsive chat layout. It will contain:

- A concise “Chat with my CV” heading and short explanation.
- Suggested Summary, Skills, Experience, and Education questions before the first message.
- Streaming assistant answers.
- Compact citation expanders showing escaped excerpts, source filename, and page number.
- A small “Clear conversation” control.
- Explicit initialization, missing-document, empty-result, provider-error, and retry states.

The interface will remove:

- The sidebar.
- Index statistics and the public re-index button.
- Model, embedding, and vector-database implementation details.
- CRT, hardware-console, screw, vent, and “system online” presentation.
- Raw document content rendered with `unsafe_allow_html=True`.

The Streamlit toolbar will use viewer/minimal mode, detailed browser errors will be disabled, and the styling will include mobile breakpoints, accessible focus states, sufficient contrast, and reduced-motion behavior.

## Document Ingestion and Indexing

The loader will ingest only supported CV files and will explicitly exclude `documents/README.md` and hidden files. For this fixed demo, PDF remains the required production input; TXT and Markdown support may remain for local tests but operational documentation must never be indexed.

Each chunk will retain stable metadata:

- Source filename.
- Page number when supplied by the PDF loader.
- Deterministic chunk index.
- Deterministic chunk identifier derived from source, page, chunk index, and content.

Chunking will remain close to the existing 700-character/140-character-overlap configuration because the corpus is only one short CV. The implementation will correct misleading “tokenization” terminology but will not add OCR, hybrid search, or reranking without evidence that this demo needs them.

The public app will never delete or rebuild a live collection on user input. Startup will create the index only when it is absent or incompatible with the current embedding/index schema. Index compatibility metadata will include an index schema version and embedding model name.

## Retrieval and Generation

Retrieval will remain dense similarity search with a small fixed top-k suitable for the CV. Empty retrieval must produce a clear “not found in the CV” response without fabricating details.

The last four user/assistant turns will be retained for conversational continuity. Follow-up questions whose meaning depends on earlier messages will be converted to standalone retrieval queries. This rewrite must be bounded and skipped for clearly standalone questions so the demo does not add an unnecessary model call to every request.

Question length, history length, retrieved context, and output length will have explicit limits. Retrieved CV text will be delimited and labeled as untrusted evidence. The model must not follow instructions found inside the document and must state when requested information is absent.

Answers will include stable citation identifiers. The UI will map those identifiers to source/page excerpts rather than implying that every retrieved chunk was used in the answer.

## Secrets and Configuration

`NVIDIA_API_KEY` will be supplied through Streamlit Secrets in deployment and through a local `.env` file during development. `.env.txt` will not be supported. No populated key will be committed.

The repository will include:

- A placeholder-only `.env.example`.
- Ignore rules covering `.env`, `.env.*`, local Chroma data, Python artifacts, and unintended documents while retaining `.env.example`.
- `.streamlit/config.toml` with minimal/viewer toolbar mode, hidden detailed errors, disabled usage telemetry, and the application theme.
- Pinned, compatible runtime dependencies.

Configuration will validate required values without exposing secret contents in the browser or logs.

## Errors and Resilience

User-facing failures will be concise and actionable. Internal logs will record exception types and a request correlation identifier without logging API keys, full prompts, CV content, or full provider responses.

Only transient network, timeout, rate-limit, and server errors will be retried. Backoff will be short, capped, and jittered. Authentication, validation, and other permanent errors will fail immediately. A streaming response will not restart after content has already reached the user; it will end with a retry action.

If the CV cannot be loaded or the index cannot be prepared, the chat input will be disabled and the interface will explain the problem. The application must not show a false online/ready indicator.

## Testing

Automated tests will not call NVIDIA endpoints or consume API credit. Provider, embeddings, and vector retrieval dependencies will be replaceable with fakes or mocks.

The test suite will cover:

- Loading the intended CV while excluding `documents/README.md` and hidden/unsupported files.
- Stable chunk identifiers and source/page/chunk metadata.
- Question-length and history-limit validation.
- Safe citation rendering with HTML-like document content.
- Conversion of conversation history and conditional follow-up rewriting.
- Empty retrieval behavior.
- Transient versus permanent provider failure behavior.
- Stream failure behavior without duplicate output.
- Engine resource caching at the Streamlit boundary where practical.

A manual deployment smoke-test checklist will cover cold start, suggested prompts, free-form and follow-up questions, streaming, citations, clear conversation, narrow mobile layout, missing secret behavior, and provider failure behavior.

## Deployment

The README will document these Streamlit Community Cloud steps:

1. Deploy the private GitHub repository with `app.py` as the entry point.
2. Add `NVIDIA_API_KEY` through Streamlit Secrets.
3. Confirm the fixed CV exists in the deployment checkout.
4. Wait for the first cold-start index build.
5. Run the smoke-test checklist.

The demo is considered ready when secrets are not tracked, the fixed CV indexes without the README, repeated sessions reuse the engine, the public UI has no administration controls or raw HTML injection, tests pass without network access, and the deployed smoke test succeeds.
