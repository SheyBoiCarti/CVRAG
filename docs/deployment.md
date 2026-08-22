# Deployment, Containerization, and Operations Guide

This guide describes the container architecture, configuration, local workflows, and deployment procedures for the **CV RAG Portfolio Knowledge Base**.

---

## 1. System Architecture

The application is structured as a modular monolith backend paired with a high-performance Next.js 16 frontend:

```text
  [ Browser Client ]
          |
   (same-origin HTTP / SSE)
          v
  [ Next.js Frontend ] (Port 3000, Node.js 24 LTS)
    - Standalone server or Vercel edge/serverless proxy
    - Server-only BACKEND_URL runtime routing (absent from browser bundles)
          |
     (private HTTP / SSE)
          v
  [ FastAPI Backend ] (Port 8000, Python 3.12)
    - Lifespan index synchronization & validation
    - Dense/Hybrid retrieval + RRF fusion + Nvidia reranking
    - Deterministic claim parsing & NVIDIA NeMo support verification
          |
    +-----+-----+
    |           |
    v           v
[ ChromaDB ] [ NVIDIA NIM APIs ]
(/data/vector_db)  (Embeddings, LLM, Reranker, Verifier)
```

---

## 2. Pinned Runtime Specifications

- **Backend Runtime**: Python 3.12 (Official `python:3.12-slim-bookworm` base image).
- **Frontend Runtime**: Node.js 24 LTS (`node:24-bookworm-slim` base image), Next.js 16, React 19, TypeScript strict mode, and npm.
- **Security Identity**: All container processes execute as non-root user `10001:10001` (`appuser` / `nextjs`).

---

## 3. Local Development with Docker Compose

### Prerequisites
- Docker Engine 24+ & Docker Compose v2+
- (Optional) NVIDIA NIM API key for live inference; or run with `PROVIDER_MODE=fake` for zero-credential offline testing.

### Quickstart (Offline Fake Provider Mode)
```bash
# Start backend and frontend containers
PROVIDER_MODE=fake docker compose up -d --build

# Wait for backend readiness probe
python scripts/wait-for-ready.py --url http://localhost:8000/api/v1/health/ready

# Open the frontend in your browser
open http://localhost:3000
```

### Quickstart (Live NVIDIA NIM Mode)
1. Copy `.env.example` to `.env` and set your `NVIDIA_API_KEY`:
   ```bash
   cp .env.example .env
   # Edit .env and set NVIDIA_API_KEY=nvapi-...
   ```
2. Start the Compose stack:
   ```bash
   docker compose up -d --build
   ```
3. Verify readiness:
   ```bash
   python scripts/wait-for-ready.py
   ```

### Stopping the Stack
```bash
docker compose down -v
```

---

## 4. Running Outside Docker

Both the FastAPI backend and Next.js frontend can be run directly on the host machine.

### Backend (Python 3.12)
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Start backend with Uvicorn
uvicorn backend.app.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### Frontend (Node.js 24)
```bash
cd frontend
npm install
npm run dev
# Accessible at http://localhost:3000 (proxies to http://127.0.0.1:8000 by default)
```

---

## 5. Production Deployment Strategies

### Option A: Single-Instance Backend Deployment (Persistent Volume)
For single-instance VM or container hosting (e.g., AWS ECS, GCP Cloud Run with volume, DigitalOcean):
- Run `Dockerfile.backend` with port `8000` exposed.
- Mount a persistent volume at `/data/vector_db` (writable by UID `10001`).
- Mount the tracked corpus at `/app/documents` as **read-only** (`:ro`).
- Run a single Uvicorn worker (`--workers 1`) by default because index initialization and local Chroma writes are process-sensitive.
- Configure liveness probe at `/api/v1/health/live` and readiness probe at `/api/v1/health/ready`.

### Option B: Immutable Indexing Deployment
For zero-startup latency or read-only container filesystems:
1. Run an offline indexing step during CI/CD to generate `/data/vector_db`.
2. Package or mount the completed `/data/vector_db` directory read-only into the backend container.
3. Multiple API instances can safely read an immutable pre-indexed volume concurrently.

### Option C: Frontend Deployment on Vercel
1. Import the repository into Vercel with Root Directory set to `frontend`.
2. Configure the server-only environment variable `BACKEND_URL` (e.g., `https://api.yourdomain.com`).
3. The Next.js Route Handler `/api/chat` automatically proxies requests to the FastAPI backend with streaming SSE support. Browser clients never communicate directly with the backend, eliminating the need for public CORS headers on direct browser endpoints.

---

## 6. Environment Configuration Reference

| Variable | Default | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | *(Required in live mode)* | API key for NVIDIA Foundation / NIM endpoints |
| `PROVIDER_MODE` | `live` | Provider mode: `live` for real NVIDIA endpoints, `fake` for deterministic offline testing |
| `EMBEDDING_MODEL` | `nvidia/nv-embedqa-e5-v5` | NVIDIA embedding model identifier |
| `LLM_MODEL` | `meta/llama-3.3-70b-instruct` | NVIDIA language model for answer generation |
| `DOCUMENTS_DIR` | `./documents` | Directory containing canonical CV Markdown/PDF documents |
| `VECTOR_DB_DIR` | `./vector_db` | Persistent Chroma database directory |
| `CHUNK_STRATEGY` | `recursive_bullet_v1` | Document chunking strategy |
| `RETRIEVAL_MODE` | `dense` | Retrieval mode (`dense` or `hybrid`) |
| `RERANK_ENABLED` | `false` | Enable/disable NVIDIA cross-encoder reranking |
| `VERIFIER_ENABLED` | `true` | Enable/disable factual claim verification |
| `VERIFIER_FAIL_CLOSED` | `true` | Fail-closed abstention when verifier encounters errors |
| `API_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated list of allowed CORS origins for direct browser clients |
| `BACKEND_URL` | `http://127.0.0.1:8000` | Server-only backend URL used by Next.js proxy route |

---

## 7. Security, Rate Limits & Guardrails

- **Non-Root Execution**: Containers run as unprivileged UID/GID `10001:10001`.
- **Request Body Limits**: Enforced `32 KiB` maximum request body size on chat endpoints.
- **Concurrency Throttle**: Semaphore limit of `4` concurrent generation requests per backend worker.
- **Timeouts**: 45s upstream request timeout, 60s SSE idle timeout, 240s SSE hard deadline.
- **CORS Protection**: CORS headers are never sent as wildcards (`*`). Explicit origin allowlists are required for direct browser access.
- **Secret Isolation**: Secrets are injected only via environment variables. Images and logs contain no credentials.

---

## 8. Observability & Safe Logging

The backend automatically attaches a correlation ID to every request via `X-Correlation-ID` and `X-Request-ID` headers.

### Structured Log Contract
Structured logs emit operational metadata while protecting user privacy:
- `correlation_id`: Request correlation UUID.
- `pipeline_version`: Version string of the active pipeline (e.g., `dense-v1`).
- `stage`: Lifecycle stage (`retrieval`, `generation`, `verification`).
- `duration_ms`: Duration of the stage in milliseconds.
- `evidence_count`: Number of retrieved context chunks.
- `decision`: Final answer policy decision (`answered` or `abstained`).
- `reason_codes`: Safe classification reason codes (e.g., `insufficient_evidence`).

**Zero-Leakage Guarantee**: Prompts, user questions, answer text, raw document text, and authorization credentials are never logged by default.

---

## 9. Verification & Rollback Procedures

### Health Check Probes
- **Liveness Probe**: `GET /api/v1/health/live` (Returns HTTP 200 while process is running; performs no provider calls).
- **Readiness Probe**: `GET /api/v1/health/ready` (Returns HTTP 200 when index is synced and ready, HTTP 503 otherwise).

### Rollback Strategy
1. **Model Fallback**: If reranking is degraded, set `RERANK_ENABLED=false` or switch `RETRIEVAL_MODE=dense`.
2. **Verifier Fallback**: If the claim verifier experiences upstream rate limits, `VERIFIER_FAIL_CLOSED=false` or `VERIFIER_ENABLED=false` can be configured as an emergency override.
3. **Container Rollback**: Maintain previous image tags and index snapshots in persistent volume to enable immediate rollback.
