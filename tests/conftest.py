import gc
import os
import socket
from pathlib import Path

import pytest
from chromadb.api.client import SharedSystemClient


def pytest_configure(config):
    # Set a unique session basetemp inside .pytest_tmp per process PID on Windows
    # to prevent file lock collisions between pytest test runs.
    base = Path(__file__).resolve().parent.parent / ".pytest_tmp" / f"run_{os.getpid()}"
    base.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(base)


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Unit tests must not access the network")

    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.fixture(autouse=True)
def cleanup_file_handles():
    yield
    try:
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    gc.collect()
