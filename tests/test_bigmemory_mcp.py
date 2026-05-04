"""Tests for the bigmemory MCP HTTP server."""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient


def _client(tmp_home):
    from tokenq.bigmemory.mcp import create_app
    from tokenq.bigmemory.store import BigMemoryStore
    return TestClient(create_app(BigMemoryStore()))


def _rpc(client, method, params=None, rpc_id=1):
    return client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params or {}},
    ).json()


def test_initialize_returns_protocol_version(tmp_home):
    with _client(tmp_home) as c:
        r = _rpc(c, "initialize", {"protocolVersion": "2024-11-05"})
        assert r["result"]["protocolVersion"] == "2024-11-05"
        assert r["result"]["serverInfo"]["name"] == "tokenq-bigmemory"
        assert "tools" in r["result"]["capabilities"]


def test_tools_list_exposes_all_tools(tmp_home):
    with _client(tmp_home) as c:
        r = _rpc(c, "tools/list")
        names = sorted(t["name"] for t in r["result"]["tools"])
        assert names == sorted([
            "memory_search", "memory_save", "memory_recent",
            "memory_forget", "memory_stats",
            "memory_set_profile", "memory_profile", "memory_expire",
            "memory_backfill_embeddings",
        ])
        for t in r["result"]["tools"]:
            assert t["inputSchema"]["type"] == "object"


def test_save_then_search_roundtrip(tmp_home):
    with _client(tmp_home) as c:
        r = _rpc(c, "tools/call", {
            "name": "memory_save",
            "arguments": {"content": "compaction layer never fires for Claude Code traffic", "kind": "fact"},
        })
        assert r["result"]["isError"] is False
        r = _rpc(c, "tools/call", {
            "name": "memory_search",
            "arguments": {"query": "compaction"},
        })
        assert "compaction" in r["result"]["content"][0]["text"]


def test_forget_then_search_returns_no_results(tmp_home):
    with _client(tmp_home) as c:
        save = _rpc(c, "tools/call", {
            "name": "memory_save",
            "arguments": {"content": "ephemeral note", "kind": "note"},
        })
        # Pull the id back out of the human text "saved memory #N (...)"
        text = save["result"]["content"][0]["text"]
        item_id = int(text.split("#", 1)[1].split(" ", 1)[0])
        r = _rpc(c, "tools/call", {"name": "memory_forget", "arguments": {"id": item_id}})
        assert r["result"]["isError"] is False
        r = _rpc(c, "tools/call", {"name": "memory_search", "arguments": {"query": "ephemeral"}})
        assert "No memories matched" in r["result"]["content"][0]["text"]


def test_notification_returns_204_no_body(tmp_home):
    with _client(tmp_home) as c:
        r = c.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert r.status_code == 204
        assert r.content == b""


def test_unknown_method_returns_jsonrpc_error(tmp_home):
    with _client(tmp_home) as c:
        r = _rpc(c, "bogus/method", rpc_id=42)
        assert r["error"]["code"] == -32601
        assert r["id"] == 42


def test_tool_call_with_bad_args_returns_user_error(tmp_home):
    with _client(tmp_home) as c:
        r = _rpc(c, "tools/call", {"name": "memory_search", "arguments": {}})
        assert r["result"]["isError"] is True
        assert "query" in r["result"]["content"][0]["text"]


def test_unknown_tool_returns_method_not_found(tmp_home):
    with _client(tmp_home) as c:
        r = _rpc(c, "tools/call", {"name": "no_such_tool", "arguments": {}})
        assert r["error"]["code"] == -32601


def test_invalid_json_returns_400_parse_error(tmp_home):
    with _client(tmp_home) as c:
        r = c.post("/mcp", content=b"not-json", headers={"content-type": "application/json"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == -32700


def test_batched_requests(tmp_home):
    with _client(tmp_home) as c:
        r = c.post("/mcp", json=[
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ])
        assert r.status_code == 200
        bodies = r.json()
        assert {b["id"] for b in bodies} == {1, 2}


def test_health_endpoint(tmp_home):
    with _client(tmp_home) as c:
        r = c.get("/healthz")
        assert r.status_code == 200
        assert r.json()["service"] == "tokenq-bigmemory"


def test_set_profile_then_profile_lists_it(tmp_home):
    with _client(tmp_home) as c:
        r = _rpc(c, "tools/call", {
            "name": "memory_set_profile",
            "arguments": {"key": "user.role", "value": "ML engineer"},
        })
        assert r["result"]["isError"] is False
        r = _rpc(c, "tools/call", {"name": "memory_profile", "arguments": {}})
        text = r["result"]["content"][0]["text"]
        assert "ML engineer" in text
        assert "user.role" in text


def test_set_profile_supersedes_via_mcp(tmp_home):
    with _client(tmp_home) as c:
        _rpc(c, "tools/call", {
            "name": "memory_set_profile",
            "arguments": {"key": "user.role", "value": "data scientist"},
        })
        _rpc(c, "tools/call", {
            "name": "memory_set_profile",
            "arguments": {"key": "user.role", "value": "ML engineer"},
        })
        r = _rpc(c, "tools/call", {"name": "memory_profile", "arguments": {}})
        text = r["result"]["content"][0]["text"]
        assert "ML engineer" in text
        assert "data scientist" not in text


def test_save_with_topic_key_supersedes_prior(tmp_home):
    with _client(tmp_home) as c:
        _rpc(c, "tools/call", {
            "name": "memory_save",
            "arguments": {
                "content": "auth uses sessions",
                "kind": "fact",
                "topic_key": "auth.method",
            },
        })
        _rpc(c, "tools/call", {
            "name": "memory_save",
            "arguments": {
                "content": "auth uses JWT",
                "kind": "fact",
                "topic_key": "auth.method",
            },
        })
        r = _rpc(c, "tools/call", {"name": "memory_search", "arguments": {"query": "auth"}})
        text = r["result"]["content"][0]["text"]
        assert "JWT" in text
        assert "sessions" not in text


def test_expire_returns_count(tmp_home):
    with _client(tmp_home) as c:
        r = _rpc(c, "tools/call", {"name": "memory_expire", "arguments": {}})
        assert r["result"]["isError"] is False
        assert "pruned" in r["result"]["content"][0]["text"]


def test_set_profile_rejects_empty_key(tmp_home):
    with _client(tmp_home) as c:
        r = _rpc(c, "tools/call", {
            "name": "memory_set_profile",
            "arguments": {"key": "", "value": "x"},
        })
        assert r["result"]["isError"] is True
