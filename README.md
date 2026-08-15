# AI Portfolio Knowledge Base // Interactive RAG Assistant

An interactive, evidence-grounded Retrieval-Augmented Generation (RAG) knowledge base that indexes a candidate's complete professional corpus, including CVs, academic theses, project case studies, system architecture notes, and certifications. Built with **Streamlit**, **LangChain**, **NVIDIA AI Endpoints**, and **ChromaDB**.

Recruiters, hiring managers, and engineers can explore broad career highlights or deep technical project details simultaneously through natural conversation, receiving real-time streamed responses with verifiable document and page-level citations.

---

## Key Highlights

- **Multi-Document Ingestion**: Automatically scans, parses, and indexes any collection of `.pdf`, `.md`, or `.txt` files placed in `documents/` (e.g., CVs, thesis papers, project writeups, and technical documentation).
- **Cross-Document Semantic Search**: Powered by NVIDIA Code & Text Embeddings (`nvidia/nv-embedcode-7b-v1`) stored in a local, persistent ChromaDB vector store, enabling cross-referencing across diverse documents.
- **Bullet-Aware Text Splitting**: Preserves resume formatting, technical lists, and project sections with tuned chunk sizes (`chunk_size=700`, `overlap=140`) to maintain semantic context continuity.
- **Conversational Context Reformulation**: Automatically detects and reformulates pronoun-heavy follow-up questions (e.g., *"What technologies did he use for that project?"*) into standalone search queries using conversational history.
- **Evidence-Grounded Streaming**: Synthesizes responses via Meta Llama 3.3 70B Instruct (`meta/llama-3.3-70b-instruct`) with strict evidence grounding and granular source citation expanders.
- **100% Offline Test Suite**: Comprehensive mocked unit test suite that runs in seconds without making network requests or consuming API credits.

---

## Architecture

```text
Streamlit Web Interface
       │
       ▼
  CVChatEngine (Session & Cache Manager)
       ├── Multi-Document Ingestion (PDF / MD / TXT)
       ├── Vector Store Manager (ChromaDB + NVIDIA Embeddings)
       └── Conversational Synthesis (Prompt Injection Defense + NVIDIA Chat Endpoint)
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

Start the Streamlit application:

```powershell
streamlit run app.py
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
# Verify syntax across all modules
python -m compileall -q app.py rag tests test_pipeline.py ui

# Run offline pytest suite (zero network calls)
python -m pytest -q

# Validate package dependencies
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
   ```

5. Click **Deploy**. The application will initialize the vector index on cold start and be ready to serve visitors.
6. Verify your deployment against the [Deployment Smoke Test Checklist](docs/deployment-smoke-test.md).
