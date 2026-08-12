import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Unit tests must not access the network")

    monkeypatch.setattr(socket, "create_connection", denied)
