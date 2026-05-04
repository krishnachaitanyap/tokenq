"""Exact-match response cache.

Pre-flight: hash the canonical request and look it up. On a fresh, deterministic,
tool-use-free hit, short-circuit with the cached response. Stream and non-stream
variants are keyed separately so a streamed request gets a streamed replay.

Post-flight: when the upstream returns a successful, deterministic, tool-use-free
response, store it under the same hash. Non-stream stores the JSON body; stream
stores the raw SSE byte sequence so it can be replayed verbatim on a hit.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import aiosqlite

from ..config import CACHE_TTL_SEC, DB_PATH
from . import PipelineRequest, PipelineShortCircuit, Stage

# `tools` is intentionally excluded: cache writes already require a non-tool_use
# response (see `_has_tool_use` gate in `after`/`after_stream`), so a hit can
# only return a tool-free response that's valid regardless of the requesting
# tools list. Including `tools` in the key would split the cache across
# otherwise-identical requests that just happen to advertise different tool
# inventories — a meaningful hit-rate loss for editors that ship tool lists
# per-request.
CACHE_KEY_FIELDS = (
    "model",
    "messages",
    "system",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "stream",
)


def _is_deterministic(body: dict[str, Any]) -> bool:
    temp = body.get("temperature")
    return temp is None or temp == 0


def _has_tool_use(response: dict[str, Any]) -> bool:
    content = response.get("content") or []
    return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)


def _cache_key(body: dict[str, Any]) -> str:
    payload = {k: body.get(k) for k in CACHE_KEY_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ExactMatchCache(Stage):
    name = "cache"

    def __init__(self, ttl_sec: int = CACHE_TTL_SEC) -> None:
        self.ttl_sec = ttl_sec

    async def run(self, req: PipelineRequest):
        body = req.body
        if not _is_deterministic(body):
            return req

        key = _cache_key(body)
        req.metadata["cache_key"] = key

        cutoff = time.time() - self.ttl_sec
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT response, input_tokens, output_tokens, created_at, is_stream "
                "FROM cache WHERE key = ?",
                (key,),
            )
            row = await cur.fetchone()
            if row is None or row["created_at"] < cutoff:
                return req

            await db.execute(
                "UPDATE cache SET hit_count = hit_count + 1, last_hit_at = ? "
                "WHERE key = ?",
                (time.time(), key),
            )
            await db.commit()

        in_tokens = row["input_tokens"] or 0
        out_tokens = row["output_tokens"] or 0
        saved = in_tokens + out_tokens

        if row["is_stream"]:
            raw = bytes(row["response"] or b"")
            # A well-formed cached SSE replay must contain a message_stop frame
            # and end with a frame terminator. Anything shorter is a truncated
            # write from a crash mid-cache; discard it rather than replay
            # garbage to the client.
            if not raw or b"message_stop" not in raw or not raw.endswith(b"\n\n"):
                await db.execute("DELETE FROM cache WHERE key = ?", (key,))
                await db.commit()
                return req
            return PipelineShortCircuit(
                stream_response=raw,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                saved_tokens=saved,
                source="cache",
            )

        try:
            response = json.loads(row["response"])
        except (json.JSONDecodeError, TypeError, ValueError):
            # Corrupt JSON — drop the entry so it gets repopulated next time.
            await db.execute("DELETE FROM cache WHERE key = ?", (key,))
            await db.commit()
            return req

        return PipelineShortCircuit(
            response=response,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            saved_tokens=saved,
            source="cache",
        )

    async def after(self, req: PipelineRequest, response: dict[str, Any]) -> None:
        key = req.metadata.get("cache_key")
        if not key:
            return
        if not _is_deterministic(req.body):
            return
        if _has_tool_use(response):
            return
        stop_reason = response.get("stop_reason")
        if stop_reason in (None, "error"):
            return

        usage = response.get("usage") or {}
        payload = json.dumps(response, separators=(",", ":"), default=str)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO cache
                    (key, response, model, input_tokens, output_tokens,
                     created_at, last_hit_at, hit_count, is_stream)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)
                """,
                (
                    key,
                    payload.encode("utf-8"),
                    response.get("model") or req.body.get("model") or "",
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    time.time(),
                    None,
                ),
            )
            await db.commit()

    async def after_stream(
        self, req: PipelineRequest, raw: bytes, captured: dict[str, Any]
    ) -> None:
        key = req.metadata.get("cache_key")
        if not key:
            return
        if not _is_deterministic(req.body):
            return
        if captured.get("tool_use"):
            return
        stop_reason = captured.get("stop_reason")
        if stop_reason in (None, "error"):
            return
        if not raw:
            return

        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO cache
                    (key, response, model, input_tokens, output_tokens,
                     created_at, last_hit_at, hit_count, is_stream)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1)
                """,
                (
                    key,
                    raw,
                    req.body.get("model") or "",
                    captured.get("input_tokens", 0),
                    captured.get("output_tokens", 0),
                    time.time(),
                    None,
                ),
            )
            await db.commit()
