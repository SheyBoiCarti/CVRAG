"""Integration tests verifying the deployment contracts, environment assumptions, and container health behaviors."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.app.main import create_app
from rag.ingestion.indexer import IndexSynchronizer
from rag.provider_factories import FakeProviderFactories
from rag.settings import BASE_DIR, Settings
from scripts.wait_for_ready import wait_for_ready


def test_dockerfile_backend_contract():
    """Validates that Dockerfile.backend adheres to security, non-root, and healthcheck contracts."""
    dockerfile_path = BASE_DIR / "Dockerfile.backend"
    assert dockerfile_path.exists(), "Dockerfile.backend must exist at repository root."
    content = dockerfile_path.read_text(encoding="utf-8")

    # Base image must use Python 3.12 slim
    assert "python:3.12-slim" in content, "Dockerfile.backend must use python:3.12-slim base."

    # Non-root user with UID/GID 10001:10001
    assert "10001:10001" in content, "Dockerfile.backend must configure UID/GID 10001:10001."
    assert "USER 10001:10001" in content, "Dockerfile.backend must switch to non-root user."

    # Exposes port 8000
    assert "EXPOSE 8000" in content, "Dockerfile.backend must expose port 8000."

    # Uses urllib for healthcheck to avoid curl dependency in slim image
    assert "urllib.request" in content, "Dockerfile.backend healthcheck must use urllib.request."
    assert "/api/v1/health/live" in content, "Dockerfile.backend healthcheck must probe /api/v1/health/live."

    # Starts single uvicorn worker
    assert "backend.app.main:create_app" in content, "Dockerfile.backend must invoke app factory."


def test_dockerfile_frontend_contract():
    """Validates that frontend/Dockerfile adheres to multi-stage Next.js standalone and non-root contracts."""
    dockerfile_path = BASE_DIR / "frontend" / "Dockerfile"
    assert dockerfile_path.exists(), "frontend/Dockerfile must exist."
    content = dockerfile_path.read_text(encoding="utf-8")

    # Base image must pin Node 24
    assert "node:24" in content, "frontend/Dockerfile must use node:24."

    # Multi-stage standalone build
    assert "AS builder" in content, "frontend/Dockerfile must have a builder stage."
    assert "AS runner" in content, "frontend/Dockerfile must have a runner stage."
    assert "standalone" in content, "frontend/Dockerfile must copy standalone artifacts."

    # Port 3000 and non-root execution
    assert "EXPOSE 3000" in content, "frontend/Dockerfile must expose port 3000."
    assert "USER" in content, "frontend/Dockerfile must switch to a non-root user."


def test_dockerignore_excludes_secrets_and_private_data():
    """Validates that .dockerignore excludes sensitive secrets, databases, caches, and test artifacts."""
    dockerignore_path = BASE_DIR / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore must exist at repository root."
    content = dockerignore_path.read_text(encoding="utf-8")
    patterns = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    expected_exclusions = [
        ".env",
        ".git",
        ".venv",
        "vector_db/",
        "docs/",
        "frontend/",
    ]
    for expected in expected_exclusions:
        assert any(expected in p for p in patterns), f".dockerignore must exclude '{expected}'."


def test_docker_compose_contract():
    """Validates docker-compose.yml service topology, mount permissions, and health dependencies."""
    compose_path = BASE_DIR / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist at repository root."
    content = compose_path.read_text(encoding="utf-8")

    # Backend service definition
    assert "backend:" in content, "Compose must define backend service."
    assert "8000:8000" in content, "Backend port 8000 must be mapped."
    assert ":ro" in content, "Documents volume must be mounted read-only (:ro)."
    assert "vector_data:/data/vector_db" in content, "Backend must mount named vector_data volume."

    # Frontend service definition
    assert "frontend:" in content, "Compose must define frontend service."
    assert "3000:3000" in content, "Frontend port 3000 must be mapped."
    assert "BACKEND_URL=http://backend:8000" in content, "Frontend must configure server-only BACKEND_URL."
    assert "condition: service_healthy" in content, "Frontend must depend on backend health."

    # No hardcoded secrets
    assert "nvapi-" not in content, "docker-compose.yml must not contain hardcoded API keys."


def test_backend_liveness_probe_offline_without_credentials():
    """Verifies that liveness endpoint returns 200 OK without requiring an API key or initialized vector store."""
    settings = Settings(NVIDIA_API_KEY=None, PROVIDER_MODE="fake")
    app = create_app(settings=settings)
    client = TestClient(app)

    resp = client.get("/api/v1/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "live"}


def test_backend_readiness_probe_reflects_state_and_safe_reasons(tmp_path):
    """Verifies that readiness probe safely reports not_ready when uninitialized and ready when populated."""
    empty_doc_dir = tmp_path / "docs"
    empty_doc_dir.mkdir()
    vec_dir = tmp_path / "vector_db"
    vec_dir.mkdir()

    # Empty documents directory -> not_ready
    settings = Settings(
        DOCUMENTS_DIR=empty_doc_dir,
        VECTOR_DB_DIR=vec_dir,
        PROVIDER_MODE="fake",
    )
    app = create_app(settings=settings)
    with TestClient(app) as client:
        resp = client.get("/api/v1/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["state"] == "not_ready"
        assert "pipeline_not_ready" in data["reason_codes"] or "index_empty" in data["reason_codes"]

    # Valid documents directory with fake provider -> ready
    (empty_doc_dir / "Custom_CV.md").write_text("# Candidate CV\nSoftware Engineer with Python skills.\n", encoding="utf-8")
    app_ready = create_app(settings=settings)
    with TestClient(app_ready) as client:
        resp = client.get("/api/v1/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "ready"
        assert data["reason_codes"] == []
        assert data["pipeline_version"] == settings.PIPELINE_VERSION


def test_read_only_documents_and_persistent_volume_lifecycle(tmp_path):
    """Validates warm restart index reuse and failure-safe rebuild behavior on persistent volume."""
    doc_dir = tmp_path / "documents"
    doc_dir.mkdir()
    cv_file = doc_dir / "Custom_CV.md"
    cv_file.write_text("# Candidate CV\nExpert in Distributed Systems.\n", encoding="utf-8")

    vec_dir = tmp_path / "vector_db"
    vec_dir.mkdir()

    settings = Settings(
        DOCUMENTS_DIR=doc_dir,
        VECTOR_DB_DIR=vec_dir,
        PROVIDER_MODE="fake",
    )
    factories = FakeProviderFactories()

    # 1. Initial build (cold start)
    syncer1 = IndexSynchronizer(settings=settings, embeddings=factories.create_embeddings(settings))
    idx1 = syncer1.ensure_active_index()
    assert idx1.index_fingerprint is not None

    # 2. Warm restart (re-sync with same corpus should immediately reuse existing active index)
    syncer2 = IndexSynchronizer(settings=settings, embeddings=factories.create_embeddings(settings))
    idx2 = syncer2.ensure_active_index()
    assert idx2.index_id == idx1.index_id
    assert idx2.index_fingerprint == idx1.index_fingerprint

    # 3. Document modification triggers atomic replacement
    cv_file.write_text("# Candidate CV\nExpert in Distributed Systems and Cloud Architecture.\n", encoding="utf-8")
    syncer3 = IndexSynchronizer(settings=settings, embeddings=factories.create_embeddings(settings))
    idx3 = syncer3.ensure_active_index()
    assert idx3.index_fingerprint != idx1.index_fingerprint


def test_correlation_id_middleware_propagation():
    """Verifies that requests receive and propagate X-Correlation-ID headers across requests."""
    settings = Settings(PROVIDER_MODE="fake")
    app = create_app(settings=settings)
    client = TestClient(app)

    # 1. Custom correlation ID provided by client
    custom_id = "test-corr-id-12345"
    resp = client.get("/api/v1/health/live", headers={"X-Correlation-ID": custom_id})
    assert resp.status_code == 200
    assert resp.headers.get("X-Correlation-ID") == custom_id
    assert resp.headers.get("X-Request-ID") == custom_id

    # 2. Generated correlation ID when omitted
    resp2 = client.get("/api/v1/health/live")
    assert resp2.status_code == 200
    assert "X-Correlation-ID" in resp2.headers
    assert len(resp2.headers["X-Correlation-ID"]) > 0


def test_cors_policy_direct_browser_vs_proxy():
    """Verifies that CORS headers are returned for configured origins and rejected for unlisted origins."""
    settings = Settings(
        API_CORS_ORIGINS=("http://localhost:3000", "https://portfolio.example.com"),
        PROVIDER_MODE="fake",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    # Allowed origin
    resp_allowed = client.options(
        "/api/v1/health/live",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp_allowed.headers.get("access-control-allow-origin") == "http://localhost:3000"

    # Disallowed origin
    resp_disallowed = client.options(
        "/api/v1/health/live",
        headers={
            "Origin": "http://evil-site.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in resp_disallowed.headers


def test_wait_for_ready_script_success_and_timeout():
    """Verifies that wait-for-ready script polls successfully and handles timeouts properly."""
    # Mock successful response
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"state": "ready", "pipeline_version": "dense-v1"}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        code = wait_for_ready("http://localhost:8000/api/v1/health/ready", timeout=5.0, interval=0.1)
        assert code == 0

    # Mock timeout
    with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
        code_timeout = wait_for_ready("http://localhost:8000/api/v1/health/ready", timeout=0.2, interval=0.05)
        assert code_timeout == 1
