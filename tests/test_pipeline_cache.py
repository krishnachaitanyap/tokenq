"""Tests for ExactMatchCache."""
from __future__ import annotations


async def test_miss_then_hit_short_circuits(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.cache import ExactMatchCache
    from tokenq.storage import init_db

    await init_db()
    stage = ExactMatchCache()

    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 100,
        "temperature": 0,
        "messages": [{"role": "user", "content": "hello"}],
    }
    req = PipelineRequest(body=body, headers={})

    miss = await stage.run(req)
    assert miss is req
    assert "cache_key" in req.metadata

    upstream_response = {
        "id": "msg_1",
        "type": "message",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "hi back"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    await stage.after(req, upstream_response)

    req2 = PipelineRequest(body=dict(body), headers={})
    hit = await stage.run(req2)
    from tokenq.pipeline import PipelineShortCircuit

    assert isinstance(hit, PipelineShortCircuit)
    assert hit.response["content"][0]["text"] == "hi back"
    assert hit.saved_tokens == 7
    assert hit.source == "cache"


async def test_stream_and_non_stream_get_distinct_keys(tmp_home):
    """Stream is now part of the cache key so a streamed request can't be
    served by a non-stream cached response (different replay format)."""
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.cache import ExactMatchCache, _cache_key
    from tokenq.storage import init_db

    await init_db()
    stage = ExactMatchCache()
    base = {"model": "x", "temperature": 0, "messages": []}
    stream_body = {**base, "stream": True}
    plain_body = {**base}

    stream_req = PipelineRequest(body=stream_body, headers={})
    plain_req = PipelineRequest(body=plain_body, headers={})
    await stage.run(stream_req)
    await stage.run(plain_req)
    assert "cache_key" in stream_req.metadata
    assert "cache_key" in plain_req.metadata
    assert stream_req.metadata["cache_key"] != plain_req.metadata["cache_key"]
    assert _cache_key(stream_body) != _cache_key(plain_body)


async def test_stream_after_then_hit_replays_sse(tmp_home):
    from tokenq.pipeline import PipelineRequest, PipelineShortCircuit
    from tokenq.pipeline.cache import ExactMatchCache
    from tokenq.storage import init_db

    await init_db()
    stage = ExactMatchCache()
    body = {
        "model": "claude-haiku-4-5",
        "max_tokens": 50,
        "temperature": 0,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    req = PipelineRequest(body=body, headers={})
    miss = await stage.run(req)
    assert miss is req

    raw_sse = (
        b'event: message_start\ndata: {"type":"message_start"}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta",'
        b'"delta":{"type":"text_delta","text":"ok"}}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta",'
        b'"delta":{"stop_reason":"end_turn"}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )
    captured = {
        "input_tokens": 9,
        "output_tokens": 2,
        "stop_reason": "end_turn",
        "tool_use": False,
        "error": None,
    }
    await stage.after_stream(req, raw_sse, captured)

    req2 = PipelineRequest(body=dict(body), headers={})
    hit = await stage.run(req2)
    assert isinstance(hit, PipelineShortCircuit)
    assert hit.stream_response == raw_sse
    assert hit.response is None
    assert hit.saved_tokens == 11
    assert hit.input_tokens == 9
    assert hit.output_tokens == 2
    assert hit.source == "cache"


async def test_stream_does_not_cache_tool_use(tmp_home):
    from tokenq.config import DB_PATH
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.cache import ExactMatchCache
    from tokenq.storage import init_db
    import sqlite3

    await init_db()
    stage = ExactMatchCache()
    body = {
        "model": "x",
        "temperature": 0,
        "stream": True,
        "messages": [{"role": "user", "content": "list files"}],
    }
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)
    captured = {
        "input_tokens": 1,
        "output_tokens": 1,
        "stop_reason": "tool_use",
        "tool_use": True,
        "error": None,
    }
    await stage.after_stream(req, b"event: message_start\ndata: {}\n\n", captured)

    rows = sqlite3.connect(DB_PATH).execute("SELECT COUNT(*) FROM cache").fetchone()
    assert rows[0] == 0


async def test_skip_when_temp_nonzero(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.cache import ExactMatchCache
    from tokenq.storage import init_db

    await init_db()
    stage = ExactMatchCache()
    body = {"model": "x", "temperature": 0.7, "messages": []}
    req = PipelineRequest(body=body, headers={})
    out = await stage.run(req)
    assert out is req
    assert "cache_key" not in req.metadata


async def test_does_not_cache_tool_use(tmp_home):
    from tokenq.config import DB_PATH
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.cache import ExactMatchCache
    from tokenq.storage import init_db
    import sqlite3

    await init_db()
    stage = ExactMatchCache()
    body = {
        "model": "x",
        "max_tokens": 50,
        "temperature": 0,
        "messages": [{"role": "user", "content": "list files"}],
    }
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)
    response = {
        "id": "msg_t",
        "type": "message",
        "model": "x",
        "content": [{"type": "tool_use", "name": "ls", "input": {}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    await stage.after(req, response)

    rows = sqlite3.connect(DB_PATH).execute("SELECT COUNT(*) FROM cache").fetchone()
    assert rows[0] == 0


async def test_ttl_expires_old_entries(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.cache import ExactMatchCache
    from tokenq.storage import init_db
    import aiosqlite
    from tokenq.config import DB_PATH

    await init_db()
    stage = ExactMatchCache(ttl_sec=1)
    body = {"model": "x", "max_tokens": 10, "temperature": 0, "messages": []}
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)
    response = {
        "id": "m",
        "type": "message",
        "model": "x",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    await stage.after(req, response)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE cache SET created_at = created_at - 10"
        )
        await db.commit()

    req2 = PipelineRequest(body=dict(body), headers={})
    out = await stage.run(req2)
    assert out is req2
