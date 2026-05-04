"""Tests for /api/report (JSON) and /api/report.pdf (binary)."""
from __future__ import annotations

import time

import httpx


async def _seed(now: float) -> None:
    """Drop a few rows across the last 30 days so window math is exercised."""
    from tokenq.storage import init_db, log_request

    await init_db()

    # 30 minutes ago — Sonnet, modest savings.
    await log_request(
        ts=now - 1800,
        model="claude-sonnet-4-6",
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        saved_tokens=200,
        saved_by_compress=200,
        latency_ms=300,
        status_code=200,
        estimated_cost_usd=0.012,
        stream=0,
    )
    # 5 hours ago — Opus, heavy cache hit.
    await log_request(
        ts=now - 5 * 3600,
        model="claude-opus-4-7",
        input_tokens=500,
        output_tokens=200,
        cache_read_tokens=10000,
        cache_creation_tokens=0,
        saved_tokens=10000,
        saved_by_cache=10000,
        latency_ms=900,
        status_code=200,
        estimated_cost_usd=0.05,
        stream=1,
    )
    # 5 days ago — Haiku, in 7d window but not in 24h.
    await log_request(
        ts=now - 5 * 86400,
        model="claude-haiku-4-5",
        input_tokens=2000,
        output_tokens=1000,
        saved_tokens=400,
        saved_by_dedup=400,
        latency_ms=200,
        status_code=200,
        estimated_cost_usd=0.007,
    )


async def _client():
    from tokenq.dashboard.app import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_report_preset_24h_excludes_older(tmp_home):
    now = time.time()
    await _seed(now)
    async with await _client() as client:
        r = await client.get("/api/report?preset=24h")
    assert r.status_code == 200
    body = r.json()
    # Window is the last 24h; the 5-day-old Haiku row must be excluded.
    assert body["totals"]["requests"] == 2
    models = {m["model"] for m in body["by_model"]}
    assert "claude-haiku-4-5" not in models
    # Comparison block must be populated and consistent.
    assert body["comparison"]["with_tokenq_usd"] > 0
    assert body["comparison"]["without_tokenq_usd"] >= body["comparison"]["with_tokenq_usd"]
    # Each by-model row carries a saved_usd field.
    for m in body["by_model"]:
        assert "saved_usd" in m


async def test_report_preset_7d_includes_older(tmp_home):
    now = time.time()
    await _seed(now)
    async with await _client() as client:
        r = await client.get("/api/report?preset=7d")
    assert r.status_code == 200
    body = r.json()
    assert body["totals"]["requests"] == 3
    # 7d span > 3d → buckets switch to daily.
    assert body["window"]["bucket_seconds"] == 86400


async def test_report_custom_range(tmp_home):
    now = time.time()
    await _seed(now)
    # Window covering only the most recent two requests (6 hours ago → now).
    start = now - 6 * 3600
    end = now + 60
    async with await _client() as client:
        r = await client.get(f"/api/report?from={start}&to={end}")
    assert r.status_code == 200
    body = r.json()
    assert body["totals"]["requests"] == 2
    assert body["window"]["from"] == start
    assert body["window"]["to"] == end


async def test_report_rejects_bad_preset(tmp_home):
    async with await _client() as client:
        r = await client.get("/api/report?preset=banana")
    assert r.status_code == 400


async def test_report_rejects_inverted_range(tmp_home):
    now = time.time()
    async with await _client() as client:
        r = await client.get(f"/api/report?from={now}&to={now - 100}")
    assert r.status_code == 400


async def test_report_pdf_returns_pdf_bytes(tmp_home):
    now = time.time()
    await _seed(now)
    async with await _client() as client:
        r = await client.get("/api/report.pdf?preset=7d")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert "attachment" in r.headers.get("content-disposition", "")
    assert r.content[:4] == b"%PDF"
    assert len(r.content) > 1000


async def test_report_pdf_empty_window(tmp_home):
    """A window with zero rows should still produce a valid PDF, not crash."""
    from tokenq.storage import init_db

    await init_db()
    async with await _client() as client:
        r = await client.get("/api/report.pdf?preset=1h")
    assert r.status_code == 200
    assert r.content[:4] == b"%PDF"
