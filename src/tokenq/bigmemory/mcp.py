"""Pure-Python MCP server over HTTP.

Speaks JSON-RPC 2.0 (MCP wire format). Implements the minimum surface a
Claude client uses to discover and call tools:

  - initialize                   handshake; returns server capabilities + info
  - notifications/initialized    client ack; no response
  - tools/list                   enumerate exposed tools + JSON Schemas
  - tools/call                   invoke a tool by name with arguments

Transport: a single POST endpoint accepts a JSON-RPC request and returns the
JSON-RPC response. No SSE / streaming — keeps the spec surface tiny while
remaining compatible with `mcp-remote`-style HTTP MCP clients.

Tools exposed:
  memory_search        FTS5 search over stored memories
  memory_save          write a new memory item
  memory_recent        list most-recent memories
  memory_forget        delete a memory by id
  memory_stats         counts + tokens by kind
  memory_set_profile   write/update a stable profile fact (supersedes prior)
  memory_profile       list the active profile (stable identity facts)
  memory_expire        run the TTL/decay sweep, return pruned count
"""
from __future__ import annotations

import contextlib
import json
from typing import Any, Awaitable, Callable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ..logging import get_logger
from .store import BigMemoryStore, MemoryItem

_log = get_logger("bigmemory.mcp")

PROTOCOL_VERSION = "2024-11-05"  # MCP spec version we implement
SERVER_NAME = "tokenq-bigmemory"
SERVER_VERSION = "0.1.0"

# JSON-RPC 2.0 error codes
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603


def _rpc_result(rpc_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _rpc_error(rpc_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": rpc_id, "error": err}


def _item_to_text(it: MemoryItem) -> str:
    src = f" [{it.source}]" if it.source else ""
    key = f" key={it.topic_key}" if it.topic_key else ""
    return f"#{it.id} ({it.kind}, {it.tokens} tok{key}){src}\n{it.content}"


TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "memory_search",
        "description": (
            "Search the local bigmemory store. Returns the top-N most relevant "
            "items. Use this to recall prior tool_results, file contents, or "
            "saved facts without re-running the original tool call.\n\n"
            "Modes: 'hybrid' (default — RRF over BM25 + cosine, best recall on "
            "paraphrased queries), 'lexical' (BM25/FTS5 only, exact-token), "
            "'semantic' (cosine over embeddings only, paraphrase-friendly). "
            "Hybrid silently falls back to lexical when the embedder isn't installed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                "kind": {
                    "type": "string",
                    "description": "Optional kind filter: tool_result | turn_summary | fact | note | preference | correction | procedure | profile.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "lexical", "semantic"],
                    "default": "hybrid",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_save",
        "description": (
            "Persist a new memory item. Use for facts, decisions, or summaries "
            "you want to recall in a future session. Duplicates (same content) "
            "are silently merged by content hash. If `topic_key` is set, prior "
            "items with the same key are superseded — use this to refresh a "
            "fact (e.g. topic_key='user.role') without leaving stale copies."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "kind": {
                    "type": "string",
                    "default": "note",
                    "description": (
                        "Memory kind. Affects TTL: correction=365d, preference=90d, "
                        "procedure=60d, fact=30d, note=14d, turn_summary=7d, "
                        "inferred=7d, tool_result=3d, profile=immortal."
                    ),
                },
                "source": {"type": "string", "description": "Optional origin label."},
                "scope": {
                    "type": "string",
                    "enum": ["global", "project", "session"],
                    "default": "session",
                },
                "topic_key": {
                    "type": "string",
                    "description": "Canonical key — sets supersession behavior.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_recent",
        "description": "List the most-recently captured memory items in reverse chronological order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                "kind": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_forget",
        "description": "Delete a memory item by id. Returns whether a row was removed.",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        },
    },
    {
        "name": "memory_stats",
        "description": "Return total item counts and token volume, broken down by kind.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "memory_set_profile",
        "description": (
            "Set a stable profile fact about the user/project (kind='profile', "
            "scope='global', long-lived). Calling again with the same `key` "
            "supersedes the previous value — use this for things like the "
            "user's role, preferred languages, or project-wide constants."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Stable key, e.g. 'user.role'."},
                "value": {"type": "string", "description": "The fact's current value."},
                "source": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "memory_profile",
        "description": (
            "Return the active stable profile — long-lived facts (kind='profile' "
            "or scope='global') ordered by topic_key. This is the 'who is this "
            "user, what is this project' snapshot, separate from recent context."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
            },
        },
    },
    {
        "name": "memory_expire",
        "description": (
            "Run the TTL/decay sweep. Deletes items whose decayed confidence has "
            "fallen below 0.05 *and* whose strength is ≤ 1, plus superseded rows "
            "older than 30 days. Returns the pruned count. Safe to call any time."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "memory_backfill_embeddings",
        "description": (
            "Compute and store embeddings for memory items that don't have one "
            "yet. Useful after installing fastembed on a database that already "
            "has rows. Bounded by `max_rows` so a single call stays cheap."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_rows": {"type": "integer", "default": 1024, "minimum": 1, "maximum": 100000},
            },
        },
    },
]


async def _tool_memory_search(store: BigMemoryStore, args: dict[str, Any]) -> dict[str, Any]:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("`query` (non-empty string) is required")
    limit = int(args.get("limit") or 10)
    kind = args.get("kind") if isinstance(args.get("kind"), str) else None
    mode = args.get("mode") if args.get("mode") in ("hybrid", "lexical", "semantic") else "hybrid"

    if mode == "lexical":
        items = await store.search(query, limit=limit, kind=kind)
    elif mode == "semantic":
        scored = await store.semantic_search(query, limit=limit, kind=kind)
        items = [it for it, _sim in scored]
    else:  # hybrid
        items = await store.hybrid_search(query, limit=limit, kind=kind)

    if not items:
        text = f"No memories matched: {query!r}"
    else:
        text = (
            f"Found {len(items)} memory item(s) for {query!r} (mode={mode}):\n\n"
            + "\n\n---\n\n".join(_item_to_text(i) for i in items)
        )
    return {"content": [{"type": "text", "text": text}], "isError": False}


async def _tool_memory_save(store: BigMemoryStore, args: dict[str, Any]) -> dict[str, Any]:
    content = args.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("`content` (non-empty string) is required")
    kind = args.get("kind") if isinstance(args.get("kind"), str) else "note"
    source = args.get("source") if isinstance(args.get("source"), str) else None
    scope_arg = args.get("scope")
    scope = scope_arg if scope_arg in ("global", "project", "session") else "session"
    topic_key = args.get("topic_key") if isinstance(args.get("topic_key"), str) else None
    item = await store.add(
        content=content, kind=kind, source=source, scope=scope, topic_key=topic_key,
    )
    return {
        "content": [{
            "type": "text",
            "text": f"saved memory #{item.id} ({item.kind}, {item.tokens} tok)",
        }],
        "isError": False,
    }


async def _tool_memory_recent(store: BigMemoryStore, args: dict[str, Any]) -> dict[str, Any]:
    limit = int(args.get("limit") or 20)
    kind = args.get("kind") if isinstance(args.get("kind"), str) else None
    items = await store.recent(limit=limit, kind=kind)
    if not items:
        text = "No memories yet."
    else:
        text = "\n\n---\n\n".join(_item_to_text(i) for i in items)
    return {"content": [{"type": "text", "text": text}], "isError": False}


async def _tool_memory_forget(store: BigMemoryStore, args: dict[str, Any]) -> dict[str, Any]:
    item_id = args.get("id")
    if not isinstance(item_id, int):
        raise ValueError("`id` (integer) is required")
    removed = await store.delete(item_id)
    text = f"removed memory #{item_id}" if removed else f"no memory with id #{item_id}"
    return {"content": [{"type": "text", "text": text}], "isError": not removed}


async def _tool_memory_stats(store: BigMemoryStore, _args: dict[str, Any]) -> dict[str, Any]:
    s = await store.stats()
    text = json.dumps(s, indent=2)
    return {"content": [{"type": "text", "text": text}], "isError": False}


async def _tool_memory_set_profile(store: BigMemoryStore, args: dict[str, Any]) -> dict[str, Any]:
    key = args.get("key")
    value = args.get("value")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("`key` (non-empty string) is required")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("`value` (non-empty string) is required")
    source = args.get("source") if isinstance(args.get("source"), str) else None
    item = await store.set_profile(key=key, value=value, source=source)
    return {
        "content": [{
            "type": "text",
            "text": f"profile #{item.id} key={key} updated",
        }],
        "isError": False,
    }


async def _tool_memory_profile(store: BigMemoryStore, args: dict[str, Any]) -> dict[str, Any]:
    limit = int(args.get("limit") or 50)
    items = await store.profile(limit=limit)
    if not items:
        text = "Profile is empty."
    else:
        text = "\n\n---\n\n".join(_item_to_text(i) for i in items)
    return {"content": [{"type": "text", "text": text}], "isError": False}


async def _tool_memory_expire(store: BigMemoryStore, _args: dict[str, Any]) -> dict[str, Any]:
    pruned = await store.expire()
    return {
        "content": [{"type": "text", "text": f"pruned {pruned} item(s)"}],
        "isError": False,
    }


async def _tool_memory_backfill_embeddings(
    store: BigMemoryStore, args: dict[str, Any]
) -> dict[str, Any]:
    max_rows = int(args.get("max_rows") or 1024)
    filled = await store.backfill_embeddings(max_rows=max_rows)
    return {
        "content": [{"type": "text", "text": f"backfilled embeddings for {filled} row(s)"}],
        "isError": False,
    }


ToolHandler = Callable[[BigMemoryStore, dict[str, Any]], Awaitable[dict[str, Any]]]

TOOL_HANDLERS: dict[str, ToolHandler] = {
    "memory_search": _tool_memory_search,
    "memory_save": _tool_memory_save,
    "memory_recent": _tool_memory_recent,
    "memory_forget": _tool_memory_forget,
    "memory_stats": _tool_memory_stats,
    "memory_set_profile": _tool_memory_set_profile,
    "memory_profile": _tool_memory_profile,
    "memory_expire": _tool_memory_expire,
    "memory_backfill_embeddings": _tool_memory_backfill_embeddings,
}


async def _dispatch(store: BigMemoryStore, msg: dict[str, Any]) -> dict[str, Any] | None:
    """Dispatch a single JSON-RPC message. Returns None for notifications."""
    if msg.get("jsonrpc") != "2.0":
        return _rpc_error(msg.get("id"), ERR_INVALID_REQUEST, "expected jsonrpc:2.0")
    method = msg.get("method")
    rpc_id = msg.get("id")
    params = msg.get("params") or {}

    # Notifications (no id) — process and return nothing
    if rpc_id is None:
        if method == "notifications/initialized":
            return None
        return None

    if method == "initialize":
        return _rpc_result(rpc_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })

    if method == "tools/list":
        return _rpc_result(rpc_id, {"tools": TOOLS_SCHEMA})

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        handler = TOOL_HANDLERS.get(name) if isinstance(name, str) else None
        if handler is None:
            return _rpc_error(rpc_id, ERR_METHOD_NOT_FOUND, f"unknown tool: {name!r}")
        try:
            result = await handler(store, args if isinstance(args, dict) else {})
        except ValueError as e:
            return _rpc_result(rpc_id, {
                "content": [{"type": "text", "text": f"error: {e}"}],
                "isError": True,
            })
        except Exception as e:
            _log.exception("tool_call_failed", extra={"tool": name})
            return _rpc_result(rpc_id, {
                "content": [{"type": "text", "text": f"internal error: {type(e).__name__}: {e}"}],
                "isError": True,
            })
        return _rpc_result(rpc_id, result)

    if method == "ping":
        return _rpc_result(rpc_id, {})

    return _rpc_error(rpc_id, ERR_METHOD_NOT_FOUND, f"unknown method: {method!r}")


def create_app(store: BigMemoryStore | None = None) -> Starlette:
    """Build a Starlette ASGI app exposing the MCP server at POST /mcp.

    Also serves GET /healthz so the dashboard / supervisor can check liveness,
    and GET /mcp returns the protocol info as a friendly probe for browsers.
    """
    store = store or BigMemoryStore()

    @contextlib.asynccontextmanager
    async def _lifespan(_app: Starlette):
        await store.init()
        _log.info("bigmemory_mcp_started")
        try:
            yield
        finally:
            _log.info("bigmemory_mcp_stopped")

    async def health(_req: Request) -> JSONResponse:
        s = await store.stats()
        return JSONResponse({
            "status": "ok",
            "service": SERVER_NAME,
            "version": SERVER_VERSION,
            "protocol": PROTOCOL_VERSION,
            "items": s["total_items"],
        })

    async def mcp_info(_req: Request) -> JSONResponse:
        return JSONResponse({
            "service": SERVER_NAME,
            "protocol": PROTOCOL_VERSION,
            "transport": "http+jsonrpc",
            "endpoint": "POST /mcp",
            "tools": [t["name"] for t in TOOLS_SCHEMA],
        })

    async def mcp_endpoint(request: Request) -> Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(
                _rpc_error(None, ERR_PARSE, "invalid JSON"),
                status_code=400,
            )

        # JSON-RPC supports batched requests (a JSON array of messages).
        if isinstance(payload, list):
            results = []
            for msg in payload:
                if not isinstance(msg, dict):
                    results.append(_rpc_error(None, ERR_INVALID_REQUEST, "expected object"))
                    continue
                r = await _dispatch(store, msg)
                if r is not None:
                    results.append(r)
            if not results:
                return Response(status_code=204)
            return JSONResponse(results)

        if not isinstance(payload, dict):
            return JSONResponse(
                _rpc_error(None, ERR_INVALID_REQUEST, "expected object or array"),
                status_code=400,
            )

        result = await _dispatch(store, payload)
        if result is None:
            # Notification — JSON-RPC says no response body.
            return Response(status_code=204)
        return JSONResponse(result)

    routes = [
        Route("/healthz", health, methods=["GET"]),
        Route("/mcp", mcp_info, methods=["GET"]),
        Route("/mcp", mcp_endpoint, methods=["POST"]),
    ]
    return Starlette(routes=routes, lifespan=_lifespan)


app = create_app()
