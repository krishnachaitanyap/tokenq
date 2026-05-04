"""Active prefix injection — Phase 2 of bigmemory.

This stage takes the relevant slice of the local memory store and injects it
into every outgoing request as an extra system block, so the model can recall
prior facts and tool_results without the user having to call `memory_search`.

# The cache-stability invariant

Anthropic's prompt cache keys on a byte-identical prefix. Naive injection — a
fresh "top relevant memories" list computed per turn — would change the prefix
on every request and destroy the upstream cache. tokenq's biggest wins today
flow from that cache, so we explicitly trade off retrieval freshness for
cache stability:

  1. Identify a "session" by hash(system_prompt, first_user_message). Two
     consecutive turns of the same conversation yield the same hash.
  2. For each session, persist ONE snapshot — the memory block to inject —
     in `memory_snapshots`. Reuse it byte-for-byte across all turns in the
     session.
  3. Refresh the snapshot every REFRESH_TURNS turns or after REFRESH_SECS
     elapsed (whichever first). Each refresh costs one Anthropic
     `cache_creation` event, the same accounting model as compaction.

# Where the block lands

Anthropic's `system` field can be either a string or a list of `{type:text}`
blocks. We always coerce to list form and append the memory block as a
trailing entry with `cache_control: {type: ephemeral}`, which marks it as a
cache breakpoint. Original system content stays cached separately, so the
injection doesn't invalidate work the caller already paid for.

# Disabled by default

`TOKENQ_BIGMEMORY_INJECT=1` flips it on. Until measurement shows net positive
input-token savings (see dashboard panels added in a follow-up), the safe
default is off — we don't want a regression in existing cache hit rate.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import aiosqlite

from ..config import (
    BIGMEMORY_INJECT_BUDGET_TOKENS,
    BIGMEMORY_INJECT_ENABLED,
    BIGMEMORY_INJECT_PROFILE_FRACTION,
    BIGMEMORY_INJECT_REFRESH_SECS,
    BIGMEMORY_INJECT_REFRESH_TURNS,
)
from ..logging import get_logger
from ..pipeline import PipelineRequest, Stage
from .store import BigMemoryStore, MemoryItem, _estimate_tokens

_log = get_logger("bigmemory.inject")


SNAPSHOTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_snapshots (
    session_hash TEXT PRIMARY KEY,
    block_text TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    created_at REAL NOT NULL,
    refreshed_at REAL NOT NULL,
    use_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memory_snapshots_refreshed
    ON memory_snapshots(refreshed_at);
"""


def _system_to_text(system: Any) -> str:
    """Flatten the Anthropic `system` field to a single string for hashing."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for blk in system:
            if isinstance(blk, dict) and blk.get("type") == "text":
                t = blk.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


def _first_user_text(messages: Any) -> str:
    """The first user message defines the session. Walk forward (not back like
    skills.py) — we want session identity, not the latest query."""
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(blk.get("text") or "")
            return "\n".join(parts)
    return ""


def _last_user_text(messages: Any) -> str:
    """Latest user message — used as the retrieval query."""
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(blk.get("text") or "")
            return "\n".join(parts)
    return ""


def _session_hash(system: Any, messages: Any) -> str:
    payload = json.dumps(
        {"sys": _system_to_text(system), "msg": _first_user_text(messages)},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _format_block(profile: list[MemoryItem], recent: list[MemoryItem]) -> str:
    """Render the memory block. Format is deliberately stable: identical inputs
    always yield identical bytes — that's the cache invariant.
    """
    out = ["<bigmemory>"]
    if profile:
        out.append("## Stable profile")
        for it in profile:
            key = f"{it.topic_key}: " if it.topic_key else ""
            out.append(f"- {key}{it.content}")
    if recent:
        out.append("## Recent context")
        for it in recent:
            label = f"[{it.kind}"
            if it.source:
                label += f" {it.source}"
            label += "]"
            # Trim long blobs — the budget caps total tokens but a single huge
            # tool_result shouldn't crowd out everything else.
            content = it.content if len(it.content) <= 800 else it.content[:800] + "…"
            out.append(f"- {label} {content}")
    out.append("</bigmemory>")
    return "\n".join(out)


def _select_within_budget(
    items: list[MemoryItem], budget_tokens: int,
) -> list[MemoryItem]:
    """Greedy take-while-fits. Items already arrive ranked by relevance, so
    cutting from the tail keeps the most-relevant entries."""
    out: list[MemoryItem] = []
    used = 0
    for it in items:
        cost = _estimate_tokens(it.content) + 20  # ~20 toks for label/markup
        if used + cost > budget_tokens:
            continue
        out.append(it)
        used += cost
    return out


class BigMemoryInjectStage(Stage):
    name = "bigmemory_inject"

    def __init__(
        self,
        store: BigMemoryStore | None = None,
        *,
        enabled: bool = BIGMEMORY_INJECT_ENABLED,
        budget_tokens: int = BIGMEMORY_INJECT_BUDGET_TOKENS,
        profile_fraction: float = BIGMEMORY_INJECT_PROFILE_FRACTION,
        refresh_turns: int = BIGMEMORY_INJECT_REFRESH_TURNS,
        refresh_secs: int = BIGMEMORY_INJECT_REFRESH_SECS,
    ) -> None:
        self.store = store or BigMemoryStore()
        self.enabled = enabled
        self.budget_tokens = budget_tokens
        self.profile_fraction = profile_fraction
        self.refresh_turns = refresh_turns
        self.refresh_secs = refresh_secs
        self._initialized = False

    async def _ensure_init(self) -> None:
        if not self._initialized:
            await self.store.init()
            async with aiosqlite.connect(self.store.db_path) as db:
                await db.executescript(SNAPSHOTS_SCHEMA)
                await db.commit()
            self._initialized = True

    async def _get_snapshot(self, session_hash: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.store.db_path) as db:
            db.row_factory = aiosqlite.Row
            row = await (await db.execute(
                "SELECT * FROM memory_snapshots WHERE session_hash = ?",
                (session_hash,),
            )).fetchone()
        return dict(row) if row else None

    async def _put_snapshot(
        self, session_hash: str, block_text: str, tokens: int, now: float,
    ) -> None:
        async with aiosqlite.connect(self.store.db_path) as db:
            await db.execute(
                """
                INSERT INTO memory_snapshots
                  (session_hash, block_text, tokens, created_at, refreshed_at, use_count)
                VALUES (?, ?, ?, ?, ?, 0)
                ON CONFLICT(session_hash) DO UPDATE SET
                  block_text = excluded.block_text,
                  tokens = excluded.tokens,
                  refreshed_at = excluded.refreshed_at,
                  use_count = 0
                """,
                (session_hash, block_text, tokens, now, now),
            )
            await db.commit()

    async def _bump_use(self, session_hash: str) -> None:
        async with aiosqlite.connect(self.store.db_path) as db:
            await db.execute(
                "UPDATE memory_snapshots SET use_count = use_count + 1 "
                "WHERE session_hash = ?",
                (session_hash,),
            )
            await db.commit()

    def _is_stale(self, snap: dict[str, Any], now: float) -> bool:
        if (now - snap["refreshed_at"]) > self.refresh_secs:
            return True
        if snap["use_count"] >= self.refresh_turns:
            return True
        return False

    async def _build_block(self, query: str) -> tuple[str, int]:
        """Compose the memory block. Returns (block_text, tokens)."""
        profile_budget = int(self.budget_tokens * self.profile_fraction)
        recent_budget = self.budget_tokens - profile_budget

        profile = await self.store.profile(limit=50)
        profile = _select_within_budget(profile, profile_budget)

        recent: list[MemoryItem] = []
        if query.strip():
            recent = await self.store.hybrid_search(query, limit=20)
        # Fallback: if the query had no lexical/semantic overlap with stored
        # memories, surface the most recent ones instead. An empty recent
        # section usually means useful context exists but didn't match the
        # phrasing of this turn's query — don't punish the user for that.
        if not recent:
            recent = await self.store.recent(limit=20)
        recent = _select_within_budget(recent, recent_budget)

        if not profile and not recent:
            return "", 0
        text = _format_block(profile, recent)
        return text, _estimate_tokens(text)

    def _inject(self, body: dict[str, Any], block_text: str) -> None:
        """Append the memory block as a `cache_control: ephemeral` system block.

        Coerces string-form `system` to list form. The original content stays
        as the first list entry (preserving its prior cache state); the memory
        block becomes the new trailing entry with its own breakpoint.
        """
        original = body.get("system")
        if isinstance(original, str):
            blocks: list[dict[str, Any]] = (
                [{"type": "text", "text": original}] if original else []
            )
        elif isinstance(original, list):
            blocks = list(original)
        else:
            blocks = []
        blocks.append({
            "type": "text",
            "text": block_text,
            "cache_control": {"type": "ephemeral"},
        })
        body["system"] = blocks

    async def run(self, req: PipelineRequest):
        if not self.enabled:
            return req
        body = req.body
        if not isinstance(body, dict):
            return req
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return req
        try:
            await self._ensure_init()
            now = time.time()
            session_hash = _session_hash(body.get("system"), messages)
            snap = await self._get_snapshot(session_hash)
            block_text: str
            tokens: int
            from_cache: bool
            if snap and not self._is_stale(snap, now):
                block_text = snap["block_text"]
                tokens = snap["tokens"]
                from_cache = True
            else:
                query = _last_user_text(messages)
                block_text, tokens = await self._build_block(query)
                if not block_text:
                    return req
                await self._put_snapshot(session_hash, block_text, tokens, now)
                from_cache = False

            if not block_text:
                return req

            self._inject(body, block_text)
            await self._bump_use(session_hash)
            req.metadata["bigmemory_injected_tokens"] = tokens
            req.metadata["bigmemory_inject_session"] = session_hash[:12]
            req.metadata["bigmemory_inject_from_cache"] = from_cache
        except Exception:
            _log.exception("bigmemory_inject_failed")
        return req
