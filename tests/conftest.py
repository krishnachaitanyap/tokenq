"""Test fixtures.

Each test gets a fresh DB in a tempdir. We do this by overriding TOKENQ_HOME
*before* importing tokenq.config, so DB_PATH points into the tmp dir.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point tokenq at a temp $HOME so tests don't touch the user's real DB.

    Embeddings are off by default — the ONNX model loads in ~80ms/call which
    multiplies test runtime by ~10x. Tests that want embeddings should request
    the `embed_on` fixture, which flips the env back on for that test.
    """
    monkeypatch.setenv("TOKENQ_HOME", str(tmp_path))
    monkeypatch.setenv("TOKENQ_EMBED_ENABLED", "0")

    # Drop any cached tokenq.* modules so they re-read the env var.
    for name in list(sys.modules):
        if name == "tokenq" or name.startswith("tokenq."):
            del sys.modules[name]

    return tmp_path


@pytest.fixture
def embed_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable real fastembed embeddings for a single test. Implies tmp_home
    has already cleared the module cache."""
    monkeypatch.setenv("TOKENQ_EMBED_ENABLED", "1")
    import sys
    for name in list(sys.modules):
        if name == "tokenq" or name.startswith("tokenq."):
            del sys.modules[name]
