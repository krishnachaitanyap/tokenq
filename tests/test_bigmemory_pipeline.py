"""Tests for BigMemoryStage — capture-only pipeline stage."""
from __future__ import annotations

import pytest


def _msg(role, *blocks):
    return {"role": role, "content": list(blocks)}


def _tool_use(uid, name, **inp):
    return {"type": "tool_use", "id": uid, "name": name, "input": inp}


def _tool_result(uid, content):
    return {"type": "tool_result", "tool_use_id": uid, "content": content}


@pytest.mark.asyncio
async def test_captures_large_tool_results(tmp_home):
    from tokenq.bigmemory.pipeline import BigMemoryStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    store = BigMemoryStore()
    stage = BigMemoryStage(store=store, min_tokens=20)
    body = {"messages": [
        _msg("assistant", _tool_use("u1", "Read", file_path="/repo/auth.py")),
        _msg("user", _tool_result("u1", "x" * 200)),  # ~50 tokens, captured
        _msg("assistant", _tool_use("u2", "Bash", command="ls")),
        _msg("user", _tool_result("u2", "tiny")),     # below threshold, skipped
    ]}
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)
    assert req.metadata["bigmemory_captured"] == 1
    items = await store.recent()
    assert len(items) == 1
    assert items[0].kind == "tool_result"
    assert items[0].source == "Read:/repo/auth.py"


@pytest.mark.asyncio
async def test_does_not_mutate_request_body(tmp_home):
    from tokenq.bigmemory.pipeline import BigMemoryStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    stage = BigMemoryStage(store=BigMemoryStore(), min_tokens=20)
    payload = "x" * 500
    body = {"messages": [
        _msg("assistant", _tool_use("u1", "Read", file_path="/x")),
        _msg("user", _tool_result("u1", payload)),
    ]}
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)
    # body still references the original tool_result content unchanged
    assert body["messages"][1]["content"][0]["content"] == payload


@pytest.mark.asyncio
async def test_dedupes_across_repeat_runs(tmp_home):
    from tokenq.bigmemory.pipeline import BigMemoryStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    store = BigMemoryStore()
    stage = BigMemoryStage(store=store, min_tokens=20)
    body = {"messages": [
        _msg("assistant", _tool_use("u1", "Read", file_path="/x")),
        _msg("user", _tool_result("u1", "y" * 200)),
    ]}
    for _ in range(3):
        await stage.run(PipelineRequest(body=body, headers={}))
    st = await store.stats()
    assert st["total_items"] == 1


@pytest.mark.asyncio
async def test_tool_result_with_list_content(tmp_home):
    """tool_result.content can be a list of {type:text} blocks too."""
    from tokenq.bigmemory.pipeline import BigMemoryStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    store = BigMemoryStore()
    stage = BigMemoryStage(store=store, min_tokens=20)
    body = {"messages": [
        _msg("assistant", _tool_use("u1", "Bash", command="cat foo.txt")),
        _msg("user", {
            "type": "tool_result",
            "tool_use_id": "u1",
            "content": [
                {"type": "text", "text": "line1\n" * 50},
                {"type": "text", "text": "line2\n" * 50},
            ],
        }),
    ]}
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)
    items = await store.recent()
    assert len(items) == 1
    assert "line1" in items[0].content and "line2" in items[0].content


@pytest.mark.asyncio
async def test_swallows_errors_to_protect_request(tmp_home, monkeypatch):
    """A broken store must not raise out of the pipeline stage."""
    from tokenq.bigmemory.pipeline import BigMemoryStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    class Broken(BigMemoryStore):
        async def init(self):
            raise RuntimeError("boom")

    stage = BigMemoryStage(store=Broken(), min_tokens=20)
    body = {"messages": [
        _msg("user", _tool_result("u1", "x" * 200)),
    ]}
    req = PipelineRequest(body=body, headers={})
    # Should NOT raise.
    result = await stage.run(req)
    assert result is req
