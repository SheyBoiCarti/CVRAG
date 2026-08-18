# AI Portfolio Knowledge Base // Interactive RAG Assistant & FastAPI Backend

An interactive, evidence-grounded Retrieval-Augmented Generation (RAG) knowledge base that indexes a candidate's complete professional corpus, including CVs, academic theses, project case studies, system architecture notes, and certifications. Built with **FastAPI**, **Streamlit**, **LangChain**, **NVIDIA AI Endpoints**, and **ChromaDB**.

Recruiters, hiring managers, and engineers can explore broad career highlights or deep technical project details simultaneously through an interactive web UI or a high-performance, OpenAI-compatible HTTP/SSE API with verifiable document and page-level citations.

---

## Key Highlights

- **Multi-Document Ingestion**: Automatically scans, parses, and indexes any collection of `.pdf`, `.md`, or `.txt` files placed in `documents/` (e.g., CVs, thesis papers, project writeups, and technical documentation).
- **Cross-Document Semantic Search**: Powered by NVIDIA QA & Text Embeddings (`nvidia/nv-embedqa-e5-v5`) stored in a local, persistent ChromaDB vector store, enabling cross-referencing across diverse documents.
- **Bullet-Aware Text Splitting**: Preserves resume formatting, technical lists, and project sections with tuned chunk sizes (`chunk_size=700`, `overlap=140`) to maintain semantic context continuity.
- **Conversational Context Reformulation**: Automatically detects and reformulates pronoun-heavy follow-up questions (e.g., *"What technologies did he use for that project?"*) into standalone search queries using conversational history.
- **Evidence-Grounded Streaming**: Synthesizes responses via Meta Llama 3.3 70B Instruct (`meta/llama-3.3-70b-instruct`) with strict evidence grounding and granular source citation expanders.
- **FastAPI Production Backend**: Async, production-grade REST & SSE streaming API (`/v1/chat/completions`) with OpenAI compatibility, structured health checks (`/health/live`, `/health/ready`), CORS validation, body size caps, and concurrency throttling.
- **100% Offline Test Suite**: Comprehensive mocked unit and integration test suite that runs in seconds without making network requests or consuming API credits.

---

## Architecture

```text
       ┌────────────────────────┐       ┌────────────────────────┐
       │ Streamlit UI (Port 8501)│       │ FastAPI Backend (8000) │
       │ (app.py)               │       │ (backend.app.main:app) │
       └───────────┬────────────┘       └───────────┬────────────┘
                   │                                │
                   │       ┌────────────────────────┘
                   ▼       ▼
              CVChatEngine / RAGPipeline
                   ├── Multi-Document Ingestion (PDF / MD / TXT)
                   ├── Vector Store Manager (ChromaDB + NVIDIA Embeddings)
                   └── Conversational Synthesis (Prompt Injection Defense + NVIDIA NIM)
```

The RAG engine initializes on startup, builds or connects to the local ChromaDB index, and provides isolated session memory for each connected user.

---

## Quick Start & Local Setup

### 1. Clone & Environment Setup

Use Python 3.12 (or 3.10+) to create a virtual environment:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

### 2. Configure Environment Variables

Copy the example environment template:

```powershell
Copy-Item .env.example .env
```

Open `.env` and add your [NVIDIA NIM / AI Foundation API Key](https://build.nvidia.com/):

```ini
NVIDIA_API_KEY=your_nvidia_api_key_here
```

### 3. Add Your Documents

Place any number of CVs, thesis documents, project case studies, or notes into the `documents/` directory:

```text
documents/
  ├── resume.pdf
  ├── thesis_paper.pdf
  ├── project_case_study.md
  └── system_architecture.txt
```

> **Privacy Note**: Files placed in `documents/` (except `.gitkeep`) are automatically excluded from version control by `.gitignore` to keep personal documents and contact details private when pushing to public repositories.

### 4. Run the Application

#### Option A: Run the FastAPI Backend

Start the FastAPI HTTP & SSE service on port 8000:

```powershell
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is available at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### Option B: Run the Streamlit Frontend

Start the interactive Streamlit chat interface on port 8501:

```powershell
streamlit run app.py
```

#### Option C: Run Both Concurrently

You can run both the FastAPI backend and Streamlit UI concurrently in separate terminals for full full-stack development.

---

## API Endpoints

The FastAPI backend exposes the following REST and streaming endpoints:

### Health Probes

- **`GET /health/live`**: Liveness probe returning status `200 OK` with `{"status": "alive"}`.
- **`GET /health/ready`**: Readiness probe returning status `200 OK` when the vector database and pipeline are fully initialized (`{"status": "ready", "pipeline_version": "dense-v1", "index_fingerprint": "..."}`), or `503 Service Unavailable` with detailed reason codes when initializing or degraded.

### Chat Completions

- **`POST /v1/chat/completions`**: OpenAI-compatible endpoint supporting both standard JSON responses and Server-Sent Events (SSE) streaming (`stream: true`).
  
#### Request Example (Streaming):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.3-70b-instruct",
    "messages": [
      {"role": "user", "content": "What is the candidate experience with Kubernetes and distributed systems?"}
    ],
    "stream": true
  }'
```

---

## Document Ingestion & Chunking Rules

- **Supported File Types**: `.pdf`, `.md`, and `.txt` files placed directly in `documents/`.
- **Automatic Exclusions**: Hidden files (prefixed with `.`), nested subdirectories, and files named `README.md`.
- **Idempotent Indexing**: Document chunks are hashed with SHA-256 identities to ensure repeat runs or additional files do not create duplicate vector entries.
- **Cross-Document Citations**: Each generated response cites the exact document source filename and page number so claims can be audited directly against the original text.

---

## Running Offline Tests & Diagnostics

Validate the codebase and run test suites:

```powershell
# Verify syntax and compilation across all modules
python -m compileall -q app.py backend rag tests test_pipeline.py ui

# Run offline pytest suite with isolated temporary directory (zero network calls)
python -m pytest -q --basetemp=.pytest_tmp

# Validate package dependency consistency
python -m pip check
```

Run the live diagnostic script to verify vector indexing and embedding connectivity against your local documents:

```powershell
# Verify document loading and ChromaDB vector search
python test_pipeline.py

# Optionally test one live LLM synthesis request
python test_pipeline.py --with-llm
```

---

## Deploying to Streamlit Community Cloud

1. Push your repository to GitHub.
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/) and click **New app**.
3. Select your repository, set the branch to `main`, and specify `app.py` as the entry file.
4. In the app settings under **Secrets**, configure your API key:

   ```toml
   NVIDIA_API_KEY = "your-nvidia-api-key-here"
   EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
   ```

   Streamlit Secrets override repository defaults and local `.env` values. Remove any stale `EMBEDDING_MODEL = "nvidia/nv-embedcode-7b-v1"` entry; that model identifier is not used by this deployment.

5. Click **Deploy**. The application will initialize the vector index once per fresh server process and reuse it across reruns and visitor sessions.
6. Verify your deployment against the [Deployment Smoke Test Checklist](docs/deployment-smoke-test.md).

`vector_db/` is intentionally ignored and is not copied to Streamlit Community Cloud. A fresh container therefore performs one NVIDIA document-embedding pass before serving searches. Chroma data is reused for the lifetime of that container, but may be rebuilt after a reboot or redeployment because Community Cloud's local filesystem is not a durable deployment artifact.

If startup fails, inspect the server log entry for `operation`, `model`, `status`, and `detail`. These diagnostics are bounded and redact credential-shaped values. A model of `nvidia/nv-embedcode-7b-v1` means Streamlit Secrets are overriding the supported repository default; update it to `nvidia/nv-embedqa-e5-v5` and reboot the app.
