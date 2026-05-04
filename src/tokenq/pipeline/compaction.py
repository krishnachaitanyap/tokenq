"""Sliding-window transcript compaction.

When a request body exceeds COMPACT_THRESHOLD_TOKENS, replace the oldest
messages with a single summary marker so Anthropic does not pay cache_read on
stale history every turn. The bulk of cost on a long Claude Code session is
the cache_read line item (0.1x base input rate for opus, applied to the
entire prefix every turn). Trimming the prefix once cuts every subsequent
turn's cache_read proportionally.

Cache stability is the whole game. Anthropic's prompt cache only re-hits on a
byte-identical prefix, so the cut point must stay constant across many
consecutive turns. We snap it to a multiple of COMPACT_CHUNK_MESSAGES — the
cut moves only on rollover turns, which are amortized over the whole chunk.

Savings accounting is deliberately conservative: rollover events are written
to a separate compaction_events table from the post-upstream hook, only when
upstream returned cache_read_input_tokens == 0 (proving the prefix really was
freshly built this turn). We do NOT touch the per-request saved_* columns —
that path was the source of the per-turn over-counting bug in compress.py.
"""
from __future__ import annotations

import json
import time
from typing import Any

import aiosqlite

from ..config import (
    COMPACT_CHUNK_MESSAGES,
    COMPACT_KEEP_RECENT_TOKENS,
    COMPACT_THRESHOLD_TOKENS,
    DB_PATH,
)
from . import PipelineRequest, Stage


def _estimate_tokens(content: Any) -> int:
    """Cheap char/4 estimator. Same approximation used by compress.py."""
    if isinstance(content, str):
        return len(content) // 4
    if isinstance(content, list):
        total = 0
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "text" and isinstance(item.get("text"), str):
                total += len(item["text"]) // 4
            elif t == "tool_result":
                total += _estimate_tokens(item.get("content"))
            elif t == "tool_use":
                inp = item.get("input")
                if isinstance(inp, (dict, list)):
                    total += len(json.dumps(inp)) // 4
            else:
                total += 50
        return total
    return 0


def _message_tokens(msg: dict[str, Any]) -> int:
    return _estimate_tokens(msg.get("content"))


def _system_tokens(system: Any) -> int:
    """Anthropic accepts system as a string or a list of {type:text} blocks."""
    if isinstance(system, str):
        return len(system) // 4
    if isinstance(system, list):
        total = 0
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    total += len(t) // 4
        return total
    return 0


def _tools_tokens(tools: Any) -> int:
    """Tool definitions are billed too. Estimator on the JSON-serialized form."""
    if not isinstance(tools, list):
        return 0
    return len(json.dumps(tools)) // 4


def _prefix_tokens(body: dict[str, Any], msg_tokens: list[int]) -> int:
    """Total tokens Anthropic actually bills for the request prefix."""
    return (
        sum(msg_tokens)
        + _system_tokens(body.get("system"))
        + _tools_tokens(body.get("tools"))
    )


SUMMARY_MARKER = "[tokenq compacted "


def _summary_text(dropped_count: int, dropped_tokens: int) -> str:
    """Deterministic placeholder. Same inputs → same bytes → cache parity."""
    return (
        f"{SUMMARY_MARKER}{dropped_count} earlier messages "
        f"(~{dropped_tokens} tokens) to reduce upstream cache_read cost. "
        f"Recent context follows.]"
    )


def _has_tool_result(msg: dict[str, Any]) -> bool:
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def _advance_past_orphan_tool_results(
    messages: list[dict[str, Any]], cut: int
) -> int:
    """Anthropic rejects a `tool_result` whose paired `tool_use` is missing
    from the prior assistant message. If the chunk-aligned cut lands such
    that messages[cut] is a user message containing tool_result blocks, the
    paired tool_use lives in messages[cut-1] which we are about to drop.
    Walk cut forward past such messages so the kept tail starts on a clean
    boundary. The advance is deterministic in the (immutable) message
    contents, so cache stability across turns is preserved.
    """
    n = len(messages)
    while cut < n and _has_tool_result(messages[cut]):
        cut += 1
    return cut


def _already_compacted(messages: list[dict[str, Any]]) -> bool:
    """A prefix that already starts with our marker has been compacted by a
    previous pass. Re-compacting would shrink it further on every turn and
    break idempotency / cache stability."""
    if not messages:
        return False
    first = messages[0]
    content = first.get("content") if isinstance(first, dict) else None
    if isinstance(content, str):
        return content.startswith(SUMMARY_MARKER)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str) and t.startswith(SUMMARY_MARKER):
                    return True
    return False


class TranscriptCompactor(Stage):
    name = "compact"

    def __init__(
        self,
        threshold_tokens: int = COMPACT_THRESHOLD_TOKENS,
        keep_recent_tokens: int = COMPACT_KEEP_RECENT_TOKENS,
        chunk_messages: int = COMPACT_CHUNK_MESSAGES,
    ) -> None:
        self.threshold_tokens = threshold_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.chunk_messages = max(1, chunk_messages)

    async def run(self, req: PipelineRequest):
        messages = req.body.get("messages")
        # Need at least 2 messages so there is something to drop while still
        # keeping recent context. The old `chunk_messages * 2` guard was too
        # strict for Claude Code traffic, where a 200K-token transcript can
        # live in 16 messages (huge tool_results, few turns).
        if not isinstance(messages, list) or len(messages) < 2:
            return req
        if _already_compacted(messages):
            return req

        msg_tokens = [_message_tokens(m) for m in messages]
        # Compare against full prefix size (system + tools + messages), not just
        # messages — Anthropic bills all three. Claude Code's tools[] + system
        # prompt alone are routinely 30-50K tokens, so a request with 50K of
        # messages can already cross 100K of billed prefix.
        if _prefix_tokens(req.body, msg_tokens) < self.threshold_tokens:
            return req

        # Walk back accumulating tokens until we have enough recent context.
        # Everything before that index is compactable.
        recent = 0
        cut = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            recent += msg_tokens[i]
            if recent >= self.keep_recent_tokens:
                cut = i
                break

        # Snap DOWN to a chunk boundary so the cut stays byte-identical across
        # consecutive turns within the chunk window — Anthropic's prompt cache
        # only re-hits on byte-identical prefixes. For low-message-count
        # Claude Code transcripts (large tool_results, few turns) the default
        # 20-msg chunk would snap everything to 0, so cap chunk size at half
        # the available droppable range.
        effective_chunk = min(self.chunk_messages, max(1, cut // 2)) if cut else 1
        cut = (cut // effective_chunk) * effective_chunk
        if cut <= 0:
            return req

        cut = _advance_past_orphan_tool_results(messages, cut)
        if cut >= len(messages):
            return req

        dropped_tokens = sum(msg_tokens[:cut])
        summary = _summary_text(cut, dropped_tokens)
        summary_tokens = len(summary) // 4

        req.body["messages"] = [
            {"role": "user", "content": [{"type": "text", "text": summary}]}
        ] + messages[cut:]

        # Stash for the post-upstream hook. We only credit savings if upstream
        # confirms a fresh cache build (cache_read_input_tokens == 0).
        req.metadata["compact_dropped_messages"] = cut
        req.metadata["compact_dropped_tokens"] = dropped_tokens
        req.metadata["compact_summary_tokens"] = summary_tokens
        return req

    async def after(self, req: PipelineRequest, response: dict[str, Any]) -> None:
        usage = (response.get("usage") or {}) if isinstance(response, dict) else {}
        await self._maybe_log_rollover(req, int(usage.get("cache_read_input_tokens", 0)))

    async def after_stream(
        self, req: PipelineRequest, raw: bytes, captured: dict[str, Any]
    ) -> None:
        await self._maybe_log_rollover(req, int(captured.get("cache_read", 0)))

    async def _maybe_log_rollover(
        self, req: PipelineRequest, cache_read: int
    ) -> None:
        dropped = int(req.metadata.get("compact_dropped_tokens", 0))
        if dropped <= 0:
            return
        if cache_read != 0:
            # Cache hit on the new compacted prefix — not a rollover this turn.
            # The previous rollover already accounted for the recurring savings.
            return
        summary_tokens = int(req.metadata.get("compact_summary_tokens", 0))
        saved = max(0, dropped - summary_tokens)
        await _log_compaction_event(
            ts=time.time(),
            model=req.body.get("model") or "",
            dropped_messages=int(req.metadata.get("compact_dropped_messages", 0)),
            dropped_tokens=dropped,
            summary_tokens=summary_tokens,
            saved_per_turn=saved,
        )


async def _log_compaction_event(
    *,
    ts: float,
    model: str,
    dropped_messages: int,
    dropped_tokens: int,
    summary_tokens: int,
    saved_per_turn: int,
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO compaction_events
                (ts, model, dropped_messages, dropped_tokens,
                 summary_tokens, saved_per_turn)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ts, model, dropped_messages, dropped_tokens, summary_tokens, saved_per_turn),
        )
        await db.commit()
