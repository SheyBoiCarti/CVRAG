# Design Specification: Streamlit Community Cloud Deployment

**Date**: 2026-08-15  
**Topic**: Streamlit Community Cloud Deployment & Document Tracking

---

## 1. Overview & Objectives

Deploy the interactive CV RAG assistant application to Streamlit Community Cloud as a public or accessible portfolio. The deployment must:
1. Include `documents/Custom_CV.md` in git tracking while maintaining default exclusion for any other private/local document drafts.
2. Ensure runtime secrets resolution seamlessly retrieves `NVIDIA_API_KEY` from Streamlit Secrets (`st.secrets`) or environment variables (`os.getenv`).
3. Ensure automated build, dependency installation, ChromaDB vector indexing on cold start, and fast conversational responses on Streamlit Community Cloud.
4. Provide a structured step-by-step release and verification guide against `docs/deployment-smoke-test.md`.

---

## 2. Architecture & Configuration Changes

### 2.1 Git Tracking (`.gitignore`)
- Modify `.gitignore` to explicitly allow tracking of `documents/Custom_CV.md`:
  ```gitignore
  # User documents & custom CV files
  documents/*
  !documents/.gitkeep
  !documents/Custom_CV.md
  ```

### 2.2 Secrets & Runtime Configuration (`rag/config.py`)
- Enhance `Config.NVIDIA_API_KEY` loading to check `st.secrets` first if available, falling back to `os.getenv("NVIDIA_API_KEY")`.
- This ensures that if the user enters secrets via the Streamlit Cloud Dashboard (which populates `st.secrets`), the app boots seamlessly without requiring custom `.env` files in production.

### 2.3 Dependencies (`requirements.txt`)
- Verify all required packages (`streamlit`, `langchain-nvidia-ai-endpoints`, `langchain-chroma`, `chromadb`, `pypdf`, `python-dotenv`) are pinned and compatible with standard Streamlit Cloud Python runtimes.

---

## 3. Step-by-Step Deployment Procedure

1. **Local Preparation & Verification**:
   - Update `.gitignore` and `rag/config.py`.
   - Stage and commit changes including `documents/Custom_CV.md`.
   - Run offline unit tests (`pytest`).
2. **Git Push**:
   - Push commits to the remote GitHub repository on the `main` branch.
3. **Streamlit Community Cloud Setup**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io/) and select **New app**.
   - Select repository, branch `main`, and main file `app.py`.
   - In **App settings -> Secrets**, add:
     ```toml
     NVIDIA_API_KEY = "nvapi-your-key-here"
     ```
   - Click **Deploy**.
4. **Post-Deployment Verification**:
   - Validate cold start, vector indexing, suggestion queries, streaming responses, and citations per `docs/deployment-smoke-test.md`.

---

## 4. Error Handling & Edge Cases

- **Missing Secret**: If `NVIDIA_API_KEY` is missing, display a clear, user-friendly error instructing the admin to configure secrets without revealing system stack traces.
- **Empty Documents**: If `documents/` contains no readable files, notify gracefully rather than failing during vector initialization.
