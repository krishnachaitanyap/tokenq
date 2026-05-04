"""Capture-only pipeline stage. Walks every tool_result block in the request,
stores any whose content exceeds BIGMEMORY_CAPTURE_MIN_TOKENS into the local
memory store. The store dedupes by content hash, so re-capturing the same
result across many turns is cheap (one row, one INSERT collision).

We do NOT mutate the request body in v1. Active prefix injection is risky for
Anthropic prompt-cache stability and is deferred to a v2 with a careful
chunk-snap strategy. v1 is data-collection only — once the store has real
volume the bigmemory MCP server lets the client retrieve memories on demand.
"""
from __future__ import annotations

from typing import Any

from ..config import BIGMEMORY_CAPTURE_MIN_TOKENS
from ..logging import get_logger
from ..pipeline import PipelineRequest, Stage
from .store import BigMemoryStore, _estimate_tokens

_log = get_logger("bigmemory.capture")


def _flatten_tool_result(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


def _last_tool_use_for(messages: list[dict[str, Any]], tool_use_id: str) -> str | None:
    """Walk backwards to find the tool name + input that produced this result.
    Returns a compact 'tool_name:first_input_value' label, or None."""
    for msg in reversed(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("id") == tool_use_id
            ):
                name = block.get("name") or "tool"
                inp = block.get("input")
                hint = ""
                if isinstance(inp, dict):
                    for k in ("file_path", "path", "command", "url", "query", "pattern"):
                        v = inp.get(k)
                        if isinstance(v, str):
                            hint = v[:200]
                            break
                return f"{name}:{hint}" if hint else name
    return None


class BigMemoryStage(Stage):
    name = "bigmemory"

    # Run the TTL sweep every Nth pipeline pass. The sweep itself is a single
    # SELECT + DELETE so it's cheap, but we still don't want it on the hot path
    # for every request. 64 keeps a busy session sweeping every few minutes
    # without becoming a per-request cost.
    EXPIRE_EVERY_N = 64

    def __init__(
        self,
        store: BigMemoryStore | None = None,
        min_tokens: int = BIGMEMORY_CAPTURE_MIN_TOKENS,
    ) -> None:
        self.store = store or BigMemoryStore()
        self.min_tokens = min_tokens
        self._initialized = False
        self._calls_since_expire = 0

    async def _ensure_init(self) -> None:
        if not self._initialized:
            await self.store.init()
            self._initialized = True

    async def _maybe_expire(self) -> None:
        self._calls_since_expire += 1
        if self._calls_since_expire >= self.EXPIRE_EVERY_N:
            self._calls_since_expire = 0
            try:
                pruned = await self.store.expire()
                if pruned:
                    _log.info("bigmemory_expired", extra={"pruned": pruned})
            except Exception:
                _log.exception("bigmemory_expire_failed")

    async def run(self, req: PipelineRequest):
        messages = req.body.get("messages")
        if not isinstance(messages, list):
            return req
        try:
            await self._ensure_init()
            await self._maybe_expire()
            captured = 0
            for msg in messages:
                content = msg.get("content") if isinstance(msg, dict) else None
                if not isinstance(content, list):
                    continue
                for block in content:
                    if (
                        not isinstance(block, dict)
                        or block.get("type") != "tool_result"
                    ):
                        continue
                    text = _flatten_tool_result(block)
                    if _estimate_tokens(text) < self.min_tokens:
                        continue
                    source = _last_tool_use_for(messages, block.get("tool_use_id") or "")
                    await self.store.add(
                        content=text, kind="tool_result", source=source
                    )
                    captured += 1
            if captured:
                req.metadata["bigmemory_captured"] = captured
        except Exception:
            _log.exception("bigmemory_capture_failed")
        return req

