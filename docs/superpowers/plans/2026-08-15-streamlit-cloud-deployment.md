# Streamlit Community Cloud Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare the codebase, track `documents/Custom_CV.md`, enhance secrets compatibility for Streamlit Community Cloud, and provide complete deployment steps.

**Architecture:** Update repository ignore rules so `documents/Custom_CV.md` is tracked in git. Enhance configuration secrets resolution to check `st.secrets` when running under Streamlit Cloud before falling back to `os.getenv`. Validate everything with offline unit tests before deploying.

**Tech Stack:** Python, Streamlit, LangChain, ChromaDB, pytest, Git, Streamlit Community Cloud.

## Global Constraints

- Never expose API keys or credentials in code, logs, or commits.
- Keep other files in `documents/` (such as local PDFs or draft notes) excluded by default except `documents/Custom_CV.md`.
- Preserve 100% offline capability for test suites.

---

### Task 1: Update `.gitignore` to Track `Custom_CV.md`

**Files:**
- Modify: `.gitignore:70-74`

**Interfaces:**
- Consumes: Existing `.gitignore` rules
- Produces: Git tracking rule allowing `documents/Custom_CV.md`

- [ ] **Step 1: Edit `.gitignore` to whitelist `documents/Custom_CV.md`**

In `.gitignore`, update the documents section:
```gitignore
# User documents & custom CV files (keep local, never commit except tracked portfolio CV)
documents/*
!documents/.gitkeep
!documents/Custom_CV.md
```

- [ ] **Step 2: Verify `git status` reflects `documents/Custom_CV.md` as untracked/trackable**

Run: `git status`
Expected: `documents/Custom_CV.md` is listed under untracked files ready to be staged.

---

### Task 2: Enhance `rag/config.py` for Streamlit Secrets Support

**Files:**
- Modify: `rag/config.py:22-26`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: `st.secrets` if available, otherwise `os.getenv`
- Produces: `Config.NVIDIA_API_KEY` resolving accurately in both local `.env` and Streamlit Cloud environments.

- [ ] **Step 1: Write test for Streamlit Secrets fallback in `tests/test_config.py`**

Add a test checking that `get_nvidia_api_key()` or `Config.load_api_key()` checks `st.secrets` when `os.environ` is empty:
```python
def test_config_retrieves_api_key_from_streamlit_secrets(monkeypatch):
    import sys
    from rag.config import Config
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    
    # Mock streamlit secrets
    class DummyStreamlit:
        secrets = {"NVIDIA_API_KEY": "nvapi-test-secret-key"}
    
    monkeypatch.setitem(sys.modules, "streamlit", DummyStreamlit)
    assert Config.get_api_key() == "nvapi-test-secret-key"
```

- [ ] **Step 2: Run test to verify expected behavior / failure**

Run: `pytest tests/test_config.py -v`

- [ ] **Step 3: Update `rag/config.py` to support `st.secrets` resolution**

Implement `get_api_key()` helper method on `Config`:
```python
    @classmethod
    def get_api_key(cls) -> str | None:
        key = os.getenv("NVIDIA_API_KEY")
        if key:
            return key
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "NVIDIA_API_KEY" in st.secrets:
                return str(st.secrets["NVIDIA_API_KEY"])
        except Exception:
            pass
        return None
```
And update `Config.validate()` to check `cls.get_api_key()`.

- [ ] **Step 4: Run unit tests to verify all tests pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS

---

### Task 3: Run Full Pre-Flight Test Suite

**Files:**
- Test: `tests/`

- [ ] **Step 1: Run complete offline pytest suite**

Run: `pytest`
Expected: All tests pass (0 failures).

---

### Task 4: Git Stage & Commit

**Files:**
- Stage: `.gitignore`, `rag/config.py`, `tests/test_config.py`, `documents/Custom_CV.md`

- [ ] **Step 1: Check git status**

Run: `git status`

- [ ] **Step 2: Stage and commit files**

```bash
git add .gitignore rag/config.py tests/test_config.py documents/Custom_CV.md
git commit -m "feat: prepare project and documents for Streamlit Community Cloud deployment"
```

- [ ] **Step 3: Push to GitHub**

```bash
git push origin main
```

---

### Task 5: Streamlit Community Cloud Configuration & Smoke Test

**Files:**
- Reference: `docs/deployment-smoke-test.md`

- [ ] **Step 1: Create application on Streamlit Cloud**
  1. Open [share.streamlit.io](https://share.streamlit.io/).
  2. Click **New app**.
  3. Select your repository, set Branch to `main`, and Main file path to `app.py`.
  4. In **App settings -> Secrets**, add:
     ```toml
     NVIDIA_API_KEY = "nvapi-your-key-here"
     ```
  5. Click **Deploy**.

- [ ] **Step 2: Perform Smoke Testing**
  Run through `docs/deployment-smoke-test.md` checklist on the deployed app URL.
