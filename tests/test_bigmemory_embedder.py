"""Tests for the optional fastembed-backed embedding layer.

Two test surfaces:
  - Graceful-degradation path: with embeddings disabled (the `tmp_home` default),
    add() must not break, semantic_search() returns [], and hybrid_search()
    falls back to lexical-only.
  - Live-embedder path: tests opt in via the `embed_on` fixture, which loads
    the real fastembed model (slow on first run; cached after). Marked with
    `@pytest.mark.embeddings` so they can be skipped on CI legs that don't
    want the model download.
"""
from __future__ import annotations

import pytest


# ---------- graceful degradation (embeddings off) ----------

@pytest.mark.asyncio
async def test_add_without_embedder_stores_null_embedding(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    item = await s.add(content="hello", kind="note")
    # No exception; row exists.
    assert item.id is not None


@pytest.mark.asyncio
async def test_semantic_search_returns_empty_when_disabled(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    await s.add(content="some content", kind="fact")
    out = await s.semantic_search("anything")
    assert out == []


@pytest.mark.asyncio
async def test_hybrid_falls_back_to_lexical_when_disabled(tmp_home):
    """Hybrid mode must transparently degrade — callers shouldn't need to
    runtime-check whether the embedder loaded."""
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    await s.add(content="dashboard refresh function", kind="fact")
    await s.add(content="auth handler raises 401", kind="fact")
    hits = await s.hybrid_search("dashboard")
    assert len(hits) == 1
    assert "dashboard" in hits[0].content


@pytest.mark.asyncio
async def test_backfill_noop_when_disabled(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    await s.add(content="x", kind="note")
    filled = await s.backfill_embeddings()
    assert filled == 0


@pytest.mark.asyncio
async def test_stats_reports_embedder_unavailable(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    st = await s.stats()
    assert st["embedder_available"] is False
    assert st["embedded_items"] == 0


# ---------- pure-utility unit tests (no model load) ----------

def test_pack_unpack_roundtrip():
    from tokenq.bigmemory.embedder import _pack, unpack
    blob = _pack([0.1, -0.2, 0.3, 0.4])
    assert isinstance(blob, bytes)
    out = unpack(blob)
    assert len(out) == 4
    for a, b in zip(out, [0.1, -0.2, 0.3, 0.4]):
        assert abs(a - b) < 1e-6


def test_cosine_bytes_identical_vectors():
    from tokenq.bigmemory.embedder import _pack, cosine_bytes
    v = _pack([1.0, 2.0, 3.0])
    assert abs(cosine_bytes(v, v) - 1.0) < 1e-6


def test_cosine_bytes_orthogonal():
    from tokenq.bigmemory.embedder import _pack, cosine_bytes
    a = _pack([1.0, 0.0, 0.0])
    b = _pack([0.0, 1.0, 0.0])
    assert abs(cosine_bytes(a, b)) < 1e-6


def test_cosine_bytes_handles_empty():
    from tokenq.bigmemory.embedder import cosine_bytes
    assert cosine_bytes(b"", b"") == 0.0
    assert cosine_bytes(b"\x00\x00\x80?", b"") == 0.0


# ---------- live-embedder path (opt-in via embed_on fixture) ----------

@pytest.mark.embeddings
@pytest.mark.asyncio
async def test_embed_on_write_populates_blob(tmp_home, embed_on):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    await s.add(content="the auth middleware uses JWT tokens", kind="fact")
    st = await s.stats()
    assert st["embedder_available"] is True
    assert st["embedded_items"] == 1


@pytest.mark.embeddings
@pytest.mark.asyncio
async def test_semantic_search_recovers_paraphrase(tmp_home, embed_on):
    """The whole point of embeddings: queries that share *meaning* but not
    *tokens* with the stored content. FTS5/BM25 returns nothing here; cosine
    over bge-small recovers the right item."""
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    await s.add(
        content="the rate limit handler raises a 429 error",
        kind="fact",
    )
    await s.add(
        content="dashboard polls /api/stats every 5 seconds",
        kind="fact",
    )
    # No token overlap with the stored "rate limit handler" sentence.
    scored = await s.semantic_search("throttle middleware error response")
    assert len(scored) >= 1
    top, sim = scored[0]
    assert "rate limit" in top.content
    assert sim > 0.4


@pytest.mark.embeddings
@pytest.mark.asyncio
async def test_hybrid_search_outperforms_lexical_on_paraphrase(tmp_home, embed_on):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    await s.add(content="the rate limit handler raises a 429", kind="fact")
    await s.add(content="dashboard polls /api/stats", kind="fact")
    await s.add(content="auth uses bearer JWT tokens", kind="fact")

    # Lexical alone misses this paraphrase.
    lex = await s.search("throttle middleware")
    assert lex == []

    # Hybrid surfaces the rate-limit row via semantic.
    hyb = await s.hybrid_search("throttle middleware")
    assert len(hyb) >= 1
    assert "rate limit" in hyb[0].content


@pytest.mark.embeddings
@pytest.mark.asyncio
async def test_backfill_fills_pre_embedding_rows(tmp_home):
    """Insert with embeddings off, then turn them on and backfill. Simulates a
    user who installs fastembed after they already have a populated DB."""
    # Phase 1 — embeddings off.
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    await s.add(content="row one", kind="note")
    await s.add(content="row two", kind="note")
    pre = await s.stats()
    assert pre["embedded_items"] == 0

    # Phase 2 — turn embeddings on, drop module cache, run backfill.
    import os, sys
    os.environ["TOKENQ_EMBED_ENABLED"] = "1"
    for name in list(sys.modules):
        if name == "tokenq" or name.startswith("tokenq."):
            del sys.modules[name]
    from tokenq.bigmemory.store import BigMemoryStore as Store2
    s2 = Store2()
    await s2.init()
    filled = await s2.backfill_embeddings(max_rows=10)
    assert filled == 2
    post = await s2.stats()
    assert post["embedded_items"] == 2


@pytest.mark.embeddings
@pytest.mark.asyncio
async def test_mcp_search_modes(tmp_home, embed_on):
    """End-to-end through the MCP tool — confirm the `mode` argument routes
    to the right retriever."""
    from starlette.testclient import TestClient
    from tokenq.bigmemory.mcp import create_app
    from tokenq.bigmemory.store import BigMemoryStore

    with TestClient(create_app(BigMemoryStore())) as c:
        c.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "memory_save", "arguments": {
                "content": "the rate limit handler raises 429", "kind": "fact",
            }},
        })
        for mode in ("lexical", "semantic", "hybrid"):
            r = c.post("/mcp", json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "memory_search", "arguments": {
                    "query": "rate limit", "mode": mode,
                }},
            }).json()
            assert r["result"]["isError"] is False
            assert f"mode={mode}" in r["result"]["content"][0]["text"]
