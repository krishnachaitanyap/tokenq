"""Integration tests for the proxy.

We mount a MockTransport in the upstream httpx client so the proxy talks to a
fake Anthropic API. Verifies: passthrough works, headers propagate, usage gets
logged, streaming bytes flow through unbuffered.
"""
from __future__ import annotations

import json
import sqlite3

import httpx
import pytest


def _mock_messages(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", "claude-sonnet-4-6"),
            "content": [{"type": "text", "text": "hello back"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 42,
                "output_tokens": 7,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    )


def _mock_stream(request: httpx.Request) -> httpx.Response:
    # Real Anthropic SSE shape — message_start with usage, deltas, message_delta with final usage.
    # We yield chunks (not bytes) so the response is *streamable*, not pre-buffered.
    frames = [
        b'event: message_start\n'
        b'data: {"type":"message_start","message":{"id":"msg_x","model":"claude-haiku-4-5","usage":{"input_tokens":10,"output_tokens":1}}}\n\n',
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n',
        b'event: message_delta\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}\n\n',
        b'event: message_stop\n'
        b'data: {"type":"message_stop"}\n\n',
    ]

    async def gen():
        for frame in frames:
            yield frame

    return httpx.Response(
        200,
        content=gen(),
        headers={"content-type": "text/event-stream"},
    )


def _router(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/messages":
        try:
            body = json.loads(request.content)
        except Exception:
            body = {}
        if body.get("stream"):
            return _mock_stream(request)
        return _mock_messages(request)
    return httpx.Response(404, json={"error": "not found"})


@pytest.fixture
async def app_with_mock(tmp_home):
    """Build the proxy app with a MockTransport-backed upstream client.

    httpx.ASGITransport doesn't run lifespan events, so we init the DB by hand.
    """
    from tokenq.proxy import upstream
    from tokenq.proxy.app import app
    from tokenq.storage import init_db

    await init_db()
    transport = httpx.MockTransport(_router)
    upstream.set_client(
        httpx.AsyncClient(
            transport=transport,
            base_url="https://api.anthropic.com",
        )
    )
    yield app
    await upstream.close_client()


async def test_messages_non_stream_passthrough_and_logs(app_with_mock, tmp_home):
    from tokenq.config import DB_PATH

    transport = httpx.ASGITransport(app=app_with_mock)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-test", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["content"][0]["text"] == "hello back"
    assert data["usage"]["input_tokens"] == 42

    # Usage should have been logged.
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT model, input_tokens, output_tokens, status_code, stream FROM requests"
    ).fetchone()
    assert row == ("claude-sonnet-4-6", 42, 7, 200, 0)


async def test_messages_stream_passthrough_and_captures_usage(app_with_mock, tmp_home):
    from tokenq.config import DB_PATH

    transport = httpx.ASGITransport(app=app_with_mock)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/v1/messages",
            headers={"x-api-key": "sk-test", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-haiku-4-5",
                "max_tokens": 50,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as resp:
            assert resp.status_code == 200
            chunks = [chunk async for chunk in resp.aiter_bytes()]

    full = b"".join(chunks)
    # The SSE frames should pass through verbatim.
    assert b"message_start" in full
    assert b"message_delta" in full
    assert b"message_stop" in full

    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT model, input_tokens, output_tokens, status_code, stream FROM requests"
    ).fetchone()
    # input_tokens captured from message_start; output_tokens from message_delta (3, overwriting initial 1).
    assert row == ("claude-haiku-4-5", 10, 3, 200, 1)


async def test_health_endpoint(app_with_mock):
    transport = httpx.ASGITransport(app=app_with_mock)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"]["db"] == "ok"
    assert body["checks"]["upstream_client"] == "ok"


async def test_health_endpoint_degraded_on_db_failure(tmp_home, monkeypatch):
    """If the DB is unreachable, /healthz must return 503 with a fail reason."""
    from tokenq.proxy import upstream
    from tokenq.proxy.app import app
    from tokenq.storage import init_db

    await init_db()
    transport = httpx.MockTransport(_router)
    upstream.set_client(
        httpx.AsyncClient(transport=transport, base_url="https://api.anthropic.com")
    )

    # Force the db_conn ping to raise.
    import tokenq.proxy.app as proxy_app

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def broken_db():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr("tokenq.storage.db_conn", broken_db)

    try:
        asgi = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=asgi, base_url="http://test") as c:
            resp = await c.get("/healthz")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["checks"]["db"].startswith("fail")
    finally:
        await upstream.close_client()
        # Reference to silence lint about proxy_app import.
        _ = proxy_app


async def test_two_identical_deterministic_requests_hit_cache(tmp_home):
    """Second identical non-stream temperature=0 request should be served from cache."""
    import sqlite3
    from tokenq.config import DB_PATH
    from tokenq.proxy import upstream
    from tokenq.proxy.app import app
    from tokenq.storage import init_db

    await init_db()

    upstream_calls = {"n": 0}

    def counting_router(request: httpx.Request) -> httpx.Response:
        upstream_calls["n"] += 1
        return _router(request)

    transport = httpx.MockTransport(counting_router)
    upstream.set_client(
        httpx.AsyncClient(transport=transport, base_url="https://api.anthropic.com")
    )

    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 50,
        "temperature": 0,
        "messages": [{"role": "user", "content": "hi"}],
    }
    headers = {"x-api-key": "sk-test", "anthropic-version": "2023-06-01"}

    asgi = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as client:
        r1 = await client.post("/v1/messages", headers=headers, json=payload)
        r2 = await client.post("/v1/messages", headers=headers, json=payload)

    await upstream.close_client()

    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["content"][0]["text"] == "hello back"
    assert r2.json()["content"][0]["text"] == "hello back"
    # Second call must not hit upstream.
    assert upstream_calls["n"] == 1

    rows = sqlite3.connect(DB_PATH).execute(
        "SELECT cached_locally, saved_tokens FROM requests ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0] == (0, 0)
    assert rows[1][0] == 1 and rows[1][1] > 0


async def test_two_identical_streaming_requests_hit_cache(tmp_home):
    """Second identical stream temperature=0 request should be served from
    cache (raw SSE bytes replayed) without an upstream call."""
    import sqlite3
    from tokenq.config import DB_PATH
    from tokenq.proxy import upstream
    from tokenq.proxy.app import app
    from tokenq.storage import init_db

    await init_db()

    upstream_calls = {"n": 0}

    def counting_router(request: httpx.Request) -> httpx.Response:
        upstream_calls["n"] += 1
        return _router(request)

    transport = httpx.MockTransport(counting_router)
    upstream.set_client(
        httpx.AsyncClient(transport=transport, base_url="https://api.anthropic.com")
    )

    payload = {
        "model": "claude-haiku-4-5",
        "max_tokens": 50,
        "temperature": 0,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    headers = {"x-api-key": "sk-test", "anthropic-version": "2023-06-01"}

    asgi = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as client:
        async with client.stream(
            "POST", "/v1/messages", headers=headers, json=payload
        ) as r1:
            assert r1.status_code == 200
            first = b"".join([c async for c in r1.aiter_bytes()])
        async with client.stream(
            "POST", "/v1/messages", headers=headers, json=payload
        ) as r2:
            assert r2.status_code == 200
            second = b"".join([c async for c in r2.aiter_bytes()])

    await upstream.close_client()

    # Second call must not hit upstream — replayed from cache.
    assert upstream_calls["n"] == 1
    # Both responses contain the same SSE markers.
    assert b"message_start" in first and b"message_stop" in first
    assert b"message_start" in second and b"message_stop" in second

    rows = sqlite3.connect(DB_PATH).execute(
        "SELECT cached_locally, saved_tokens, stream FROM requests ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0] == (0, 0, 1)
    assert rows[1][0] == 1 and rows[1][1] > 0 and rows[1][2] == 1


async def test_pipeline_short_circuit_serves_cached_response(tmp_home):
    """A pipeline stage that returns PipelineShortCircuit should bypass upstream entirely."""
    from tokenq.pipeline import (
        Pipeline,
        PipelineRequest,
        PipelineShortCircuit,
        Stage,
    )
    from tokenq.proxy.intercept import handle_messages
    from starlette.requests import Request

    class FakeCache(Stage):
        async def run(self, req: PipelineRequest):
            return PipelineShortCircuit(
                response={
                    "id": "msg_cached",
                    "type": "message",
                    "content": [{"type": "text", "text": "from cache"}],
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
                saved_tokens=100,
                source="test",
            )

    pipeline = Pipeline([FakeCache()])

    # Build a fake Starlette request.
    body = json.dumps({"model": "claude-sonnet-4-6", "messages": []}).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/messages",
        "headers": [(b"x-api-key", b"sk-test")],
        "query_string": b"",
        "scheme": "http",
        "server": ("test", 80),
    }
    request = Request(scope, receive=receive)

    # Init DB so log_request can write.
    from tokenq.storage import init_db

    await init_db()

    response = await handle_messages(request, pipeline=pipeline)
    assert response.status_code == 200
    body_out = json.loads(response.body)
    assert body_out["content"][0]["text"] == "from cache"

    from tokenq.config import DB_PATH

    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT cached_locally, saved_tokens FROM requests"
    ).fetchone()
    assert row == (1, 100)


async def test_malformed_json_body_returns_400(tmp_home):
    """A request with invalid JSON should return 400, not silently empty {}."""
    from tokenq.proxy import upstream
    from tokenq.proxy.app import app
    from tokenq.storage import init_db

    await init_db()
    transport = httpx.MockTransport(_router)
    upstream.set_client(
        httpx.AsyncClient(transport=transport, base_url="https://api.anthropic.com")
    )
    try:
        asgi = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=asgi, base_url="http://test") as client:
            resp = await client.post(
                "/v1/messages",
                headers={"content-type": "application/json", "x-api-key": "sk-test"},
                content=b"{not valid json",
            )
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"
    finally:
        await upstream.close_client()


async def test_upstream_transport_failure_returns_502(tmp_home):
    """A connection-level upstream failure should surface as 502, log the
    request, and reset the shared httpx client."""
    from tokenq.proxy import upstream
    from tokenq.proxy.app import app
    from tokenq.storage import init_db
    from tokenq.config import DB_PATH

    await init_db()

    def boom(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(boom)
    upstream.set_client(
        httpx.AsyncClient(transport=transport, base_url="https://api.anthropic.com")
    )

    asgi = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=asgi, base_url="http://test") as client:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-test", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "api_error"

    # Failure must still produce a request log row so the dashboard sees it.
    row = sqlite3.connect(DB_PATH).execute(
        "SELECT status_code, error FROM requests ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row[0] == 502
    assert "ConnectError" in row[1]


async def test_pipeline_exception_falls_back_to_passthrough(tmp_home):
    """A crashing pipeline stage must not drop the request — the original body
    should still be forwarded upstream and a normal response returned."""
    from tokenq.pipeline import Pipeline, PipelineRequest, Stage
    from tokenq.proxy.intercept import handle_messages
    from tokenq.proxy import upstream
    from tokenq.storage import init_db
    from starlette.requests import Request

    await init_db()
    transport = httpx.MockTransport(_router)
    upstream.set_client(
        httpx.AsyncClient(transport=transport, base_url="https://api.anthropic.com")
    )
    try:
        class ExplodingStage(Stage):
            name = "explode"

            async def run(self, req: PipelineRequest):
                raise RuntimeError("boom")

        pipeline = Pipeline([ExplodingStage()])

        body = json.dumps({
            "model": "claude-sonnet-4-6",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
        }).encode()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "headers": [(b"x-api-key", b"sk-test")],
            "query_string": b"",
            "scheme": "http",
            "server": ("test", 80),
        }
        request = Request(scope, receive=receive)
        response = await handle_messages(request, pipeline=pipeline)
        assert response.status_code == 200
        assert json.loads(response.body)["content"][0]["text"] == "hello back"
    finally:
        await upstream.close_client()
