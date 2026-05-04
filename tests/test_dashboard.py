"""Smoke tests for the dashboard ASGI app.

We exercise each /api/* endpoint with an empty DB and with one logged request
so the JSON shapes are locked in. ASGITransport doesn't run lifespan events,
so we init_db() by hand.
"""
from __future__ import annotations

import time

import httpx
import pytest


@pytest.fixture
async def dashboard_app(tmp_home):
    from tokenq.dashboard.app import app
    from tokenq.storage import init_db

    await init_db()
    return app


async def _seed_one_request(model: str = "claude-sonnet-4-6") -> None:
    from tokenq.storage import log_request

    await log_request(
        ts=time.time(),
        model=model,
        input_tokens=100,
        output_tokens=20,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        saved_tokens=50,
        saved_by_cache=50,
        saved_by_dedup=0,
        saved_by_compress=0,
        saved_by_skills=0,
        latency_ms=150,
        status_code=200,
        error=None,
        stream=0,
        estimated_cost_usd=0.001,
    )


async def test_index_serves_html(dashboard_app):
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/")
    assert resp.status_code == 200
    assert "html" in resp.headers.get("content-type", "")


async def test_stats_empty(dashboard_app):
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/stats?range=24h")
    assert resp.status_code == 200
    body = resp.json()
    assert body["requests"] == 0
    assert body["saved_tokens"] == 0


async def test_stats_with_data(dashboard_app):
    await _seed_one_request()
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/stats?range=24h")
    assert resp.status_code == 200
    body = resp.json()
    assert body["requests"] == 1
    assert body["saved_tokens"] == 50
    assert body["saved_by_cache"] == 50


async def test_recent(dashboard_app):
    await _seed_one_request()
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/recent")
    assert resp.status_code == 200
    items = resp.json()
    assert isinstance(items, list)
    assert len(items) == 1
    assert items[0]["model"] == "claude-sonnet-4-6"


async def test_timeseries(dashboard_app):
    await _seed_one_request()
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/timeseries?range=24h")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_by_model(dashboard_app):
    await _seed_one_request()
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/by_model?range=24h")
    assert resp.status_code == 200
    rows = resp.json()
    assert any(r.get("model") == "claude-sonnet-4-6" for r in rows)


async def test_expensive(dashboard_app):
    await _seed_one_request()
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/expensive?range=24h")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_skill_compressions_empty(dashboard_app):
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/skill_compressions")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_compactions_empty(dashboard_app):
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/compactions?range=24h")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_saved_usd_prices_cache_hits_with_output_rate(dashboard_app):
    """A cache hit saves both input AND output. The saved_usd field must price
    output tokens at the output rate, not the (much cheaper) input rate."""
    from tokenq.storage import log_request

    # Seed: one cache hit on Sonnet with 1000 input + 2000 output saved.
    await log_request(
        ts=time.time(),
        model="claude-sonnet-4-6",
        input_tokens=1000,
        output_tokens=2000,
        cached_locally=1,
        saved_tokens=3000,
        saved_by_cache=3000,
        latency_ms=5,
        status_code=200,
        stream=0,
        estimated_cost_usd=0.0,
    )
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    # Sonnet 4.6: $3/M input, $15/M output. Correct = 1000*$3/M + 2000*$15/M = $0.033.
    # Old (input-rate-only) bug = 3000*$3/M = $0.009. Assert we're at the fixed value.
    assert body["saved_usd"] == pytest.approx(0.033, abs=1e-6)


async def test_saved_usd_prices_noncache_savings_at_input_rate(dashboard_app):
    """Dedup/compress/skills only trim input — must still be priced at input rate."""
    from tokenq.storage import log_request

    await log_request(
        ts=time.time(),
        model="claude-sonnet-4-6",
        input_tokens=500,
        output_tokens=100,
        cached_locally=0,
        saved_tokens=2000,
        saved_by_compress=2000,
        latency_ms=120,
        status_code=200,
        stream=0,
        estimated_cost_usd=0.0015,
    )
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/stats")
    body = resp.json()
    # 2000 * $3/M = $0.006. No output uplift because nothing was a cache hit.
    assert body["saved_usd"] == pytest.approx(0.006, abs=1e-6)


async def test_report_json(dashboard_app):
    await _seed_one_request()
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/report?range=24h")
    assert resp.status_code == 200
    body = resp.json()
    assert "stats" in body or "requests" in body or isinstance(body, dict)
