# Chat with My CV

A focused Retrieval-Augmented Generation (RAG) portfolio demo built with Streamlit, LangChain, NVIDIA AI Endpoints, and ChromaDB. It answers questions about my fixed CV and displays the supporting CV excerpts and page numbers.

## What it demonstrates

- PDF extraction and deterministic document chunking.
- NVIDIA embeddings with a persistent local Chroma index.
- Conversational retrieval for follow-up questions.
- Streamed, evidence-grounded answers with inline source identifiers.
- A responsive Streamlit chat interface.
- Offline automated tests that do not consume API credit.

This is intentionally a fixed-CV portfolio demo, not a document-upload or multi-user recruitment platform. AI answers can be imperfect and should be verified against the displayed sources.

## Architecture

```text
Streamlit UI
    -> cached CVChatEngine
        -> fixed CV loader and chunker
        -> local versioned Chroma index
        -> NVIDIA embedding and chat endpoints
```

The engine is cached once per Streamlit process. Conversation history remains isolated in each browser session. Streamlit Community Cloud may clear local application storage during restarts, so the small CV index can be rebuilt on a cold start.

## Local setup

Use Python 3.12 for the same runtime targeted by deployment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Edit `.env` and set your NVIDIA API key:

```ini
NVIDIA_API_KEY=your_nvidia_api_key_here
```

Run the app:

```powershell
streamlit run app.py
```

Run the offline checks:

```powershell
python -m compileall -q app.py rag tests test_pipeline.py
python -m pytest -q
python -m pip check
```

The optional live diagnostic performs embedding and retrieval API requests. Generation is enabled only with the explicit flag:

```powershell
python test_pipeline.py
python test_pipeline.py --with-llm
```

## Deploy to Streamlit Community Cloud

1. Push this private repository to GitHub.
2. In Streamlit Community Cloud, create an app from the repository and select `app.py` as the entry point.
3. Open the app's **Secrets** editor and add:

   ```toml
   NVIDIA_API_KEY = "your-rotated-nvidia-api-key"
   ```

4. Deploy and wait for the first cold-start index build.
5. Complete [the deployment smoke test](docs/deployment-smoke-test.md).

Never commit `.env`, `.env.txt`, or `.streamlit/secrets.toml`. If an API key has ever been stored in a local unprotected file, rotate it before deployment.

## CV privacy

The real CV PDF is deliberately included so Streamlit can deploy this fixed-document demo from a private repository. Making the repository public also publishes the CV and its Git history. If the original PDF should remain private, create a clean public repository containing a sanitized CV rather than deleting the PDF in a later commit.

## Supported documents

The loader supports PDF, UTF-8 text, and Markdown files placed directly in `documents/`. Hidden files, nested files, and `documents/README.md` are excluded. This deployed demo uses only `documents/Sheharyar CV 2026.pdf`.
