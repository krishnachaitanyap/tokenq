"""Tests for TranscriptCompactor.

The whole point of this stage is to remain stable across many consecutive
turns within a chunk window so the upstream prompt cache re-hits. The bug we
do NOT want to reintroduce is per-turn over-counting of savings — so we also
verify that compaction events are logged via the after-hook only when
upstream confirms a fresh cache build.
"""
from __future__ import annotations


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _bulky_transcript(n_messages: int, chars_per_message: int) -> list[dict]:
    """A multi-turn transcript whose total comfortably exceeds the threshold."""
    body = "x" * chars_per_message
    return [
        _msg("user" if i % 2 == 0 else "assistant", f"turn {i}: {body}")
        for i in range(n_messages)
    ]


def _claude_code_transcript(n_turns: int, tool_chars: int) -> list[dict]:
    """Realistic Claude Code shape: alternating tool_use / tool_result with
    huge tool_results. Few messages, lots of tokens."""
    msgs = []
    for i in range(n_turns):
        msgs.append({"role": "assistant", "content": [
            {"type": "text", "text": "I'll read the file."},
            {"type": "tool_use", "id": f"u{i}", "name": "Read",
             "input": {"file_path": f"/repo/file_{i}.py"}},
        ]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"u{i}",
             "content": "x" * tool_chars},
        ]})
    return msgs


async def test_below_threshold_is_noop():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.compaction import TranscriptCompactor

    messages = _bulky_transcript(40, 100)
    body = {"messages": list(messages)}
    req = PipelineRequest(body=body, headers={})
    await TranscriptCompactor(threshold_tokens=10_000_000).run(req)
    assert body["messages"] == messages


async def test_compacts_when_above_threshold_and_keeps_recent_intact():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.compaction import TranscriptCompactor

    messages = _bulky_transcript(80, 4000)  # ~80k chars total
    body = {"messages": list(messages)}
    req = PipelineRequest(body=body, headers={})
    await TranscriptCompactor(
        threshold_tokens=5_000,
        keep_recent_tokens=2_000,
        chunk_messages=10,
    ).run(req)

    out = body["messages"]
    # First message is the summary marker.
    head = out[0]["content"][0]["text"]
    assert head.startswith("[tokenq compacted")
    # Tail messages are preserved verbatim — the most recent original message
    # must be present unchanged.
    assert out[-1]["content"][0]["text"] == messages[-1]["content"][0]["text"]


async def test_cut_point_is_chunk_aligned_and_stable_across_turns():
    """Two consecutive turns within the same chunk must produce byte-identical
    compacted prefixes — that is what makes Anthropic's prompt cache re-hit.
    """
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.compaction import TranscriptCompactor

    base = _bulky_transcript(80, 4000)
    body_a = {"messages": list(base)}
    body_b = {"messages": list(base) + [_msg("user", "follow-up")]}

    compactor = TranscriptCompactor(
        threshold_tokens=5_000,
        keep_recent_tokens=2_000,
        chunk_messages=20,
    )
    req_a = PipelineRequest(body=body_a, headers={})
    req_b = PipelineRequest(body=body_b, headers={})
    await compactor.run(req_a)
    await compactor.run(req_b)

    # The prefix that upstream caches must be identical across turns.
    prefix_a = body_a["messages"][:-1]
    prefix_b = body_b["messages"][:-2]
    assert prefix_a == prefix_b
    assert body_a["messages"][0]["content"][0]["text"] == body_b["messages"][0]["content"][0]["text"]


async def test_idempotent_on_second_pass():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.compaction import TranscriptCompactor

    messages = _bulky_transcript(80, 4000)
    body = {"messages": list(messages)}
    compactor = TranscriptCompactor(
        threshold_tokens=5_000,
        keep_recent_tokens=2_000,
        chunk_messages=10,
    )
    await compactor.run(PipelineRequest(body=body, headers={}))
    snapshot = [dict(m) for m in body["messages"]]
    await compactor.run(PipelineRequest(body=body, headers={}))
    assert body["messages"] == snapshot


async def test_logs_event_only_when_upstream_built_fresh_cache(tmp_home):
    """The post-upstream hook must log a compaction_event only when
    cache_read_input_tokens == 0 (real rollover). On cache-hit turns the
    compaction is idempotent and must NOT log a duplicate event — that would
    re-introduce the per-turn over-counting bug we just fixed in compress.py.
    """
    import aiosqlite
    from tokenq.config import DB_PATH
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.compaction import TranscriptCompactor
    from tokenq.storage import init_db

    await init_db()

    messages = _bulky_transcript(80, 4000)
    compactor = TranscriptCompactor(
        threshold_tokens=5_000,
        keep_recent_tokens=2_000,
        chunk_messages=10,
    )

    # Rollover turn: upstream had to build the cache.
    req_rollover = PipelineRequest(
        body={"model": "claude-opus-4-7", "messages": list(messages)},
        headers={},
    )
    await compactor.run(req_rollover)
    await compactor.after(
        req_rollover,
        {"usage": {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 50_000}},
    )

    # Subsequent turn within the chunk: upstream served from cache.
    req_hit = PipelineRequest(
        body={"model": "claude-opus-4-7", "messages": list(messages) + [_msg("assistant", "ok")]},
        headers={},
    )
    await compactor.run(req_hit)
    await compactor.after(
        req_hit,
        {"usage": {"cache_read_input_tokens": 90_000, "cache_creation_input_tokens": 0}},
    )

    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT COUNT(*), COALESCE(SUM(saved_per_turn), 0) FROM compaction_events")
        n, saved = await cur.fetchone()

    assert n == 1, f"expected 1 rollover event, got {n}"
    assert saved > 0


# ────────────────────────────────────────────────────────────────────────────
# Bug-fix regression tests (2026-05-02): the layer was firing 0 times in 24h
# of real Claude Code traffic despite 34% of requests crossing the threshold.
# Two root causes — see compaction.py for the fixes.
# ────────────────────────────────────────────────────────────────────────────

async def test_fires_on_few_messages_huge_tool_results():
    """Bug 1: the old `len(messages) < chunk_messages * 2` guard refused to
    compact a 200K-token transcript that lived in 16 messages."""
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.compaction import TranscriptCompactor

    msgs = _claude_code_transcript(n_turns=8, tool_chars=100_000)
    body = {"messages": msgs}
    req = PipelineRequest(body=body, headers={})
    await TranscriptCompactor(
        threshold_tokens=80_000,
        keep_recent_tokens=20_000,
        chunk_messages=20,  # Default; previously would have refused all 16-msg shapes
    ).run(req)
    assert "compact_dropped_tokens" in req.metadata, (
        "compactor must fire on huge-tool_result/few-message transcripts"
    )
    assert req.metadata["compact_dropped_tokens"] > 0


async def test_threshold_counts_system_and_tools():
    """Bug 2: the threshold check used to ignore system + tools, which are
    routinely 30-50K tokens for Claude Code requests. A request with 60K of
    messages + 50K of tools should be considered above an 80K threshold."""
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.compaction import TranscriptCompactor

    # ~38K msg tokens — well below the 80K threshold on its own.
    msgs = _claude_code_transcript(n_turns=10, tool_chars=15_000)
    msg_only_body = {"messages": msgs}
    req_no_tools = PipelineRequest(body=msg_only_body, headers={})
    await TranscriptCompactor(
        threshold_tokens=80_000,
        keep_recent_tokens=10_000,
        chunk_messages=10,
    ).run(req_no_tools)
    assert "compact_dropped_tokens" not in req_no_tools.metadata, (
        "messages alone should NOT cross 80K threshold for this fixture"
    )

    # Same messages, but now add tools that push the total prefix above 80K.
    big_tools = [
        {
            "name": f"tool_{i}",
            "description": "x" * 4_000,
            "input_schema": {"type": "object", "properties": {}},
        }
        for i in range(50)  # ~50K tokens of tools — pushes prefix above 80K
    ]
    body = {"messages": list(msgs), "tools": big_tools}
    req = PipelineRequest(body=body, headers={})
    await TranscriptCompactor(
        threshold_tokens=80_000,
        keep_recent_tokens=10_000,
        chunk_messages=10,
    ).run(req)
    assert "compact_dropped_tokens" in req.metadata, (
        "system+tools must contribute to the threshold check"
    )


async def test_cut_never_orphans_tool_result():
    """Anthropic rejects a request whose first user message starts with a
    `tool_result` block whose paired `tool_use` is missing — the API error is
    `unexpected tool_use_id found in tool_result blocks`. If the chunk-aligned
    cut lands on a user(tool_result), we must advance past it so the matching
    tool_use isn't stranded in the dropped prefix.
    """
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.compaction import TranscriptCompactor

    # 40-message Claude Code transcript with a chunk size of 1 forces the
    # walkback to settle on an odd index (a user(tool_result)) — exactly the
    # orphan scenario.
    msgs = _claude_code_transcript(n_turns=20, tool_chars=10_000)
    body = {"messages": list(msgs)}
    req = PipelineRequest(body=body, headers={})
    await TranscriptCompactor(
        threshold_tokens=40_000,
        keep_recent_tokens=20_000,
        chunk_messages=1,
    ).run(req)

    out = body["messages"]
    # Sanity: compaction must have actually fired for this test to be
    # meaningful — otherwise we're checking the unmodified input.
    assert "compact_dropped_tokens" in req.metadata
    assert out[0]["content"][0]["text"].startswith("[tokenq compacted")
    # Kept tail is out[1:].
    tool_use_ids: set[str] = set()
    for m in out[1:]:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        if m.get("role") == "assistant":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tool_use_ids.add(b.get("id") or "")
        elif m.get("role") == "user":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tuid = b.get("tool_use_id") or ""
                    assert tuid in tool_use_ids, (
                        f"orphaned tool_result {tuid!r} — its tool_use was "
                        f"dropped by compaction"
                    )


async def test_does_not_recompact_already_compacted_prefix():
    """Idempotency invariant: running compaction on a body that was already
    compacted (first message is the SUMMARY_MARKER) must be a no-op. Cache
    stability depends on this."""
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.compaction import TranscriptCompactor

    msgs = _claude_code_transcript(n_turns=20, tool_chars=10_000)
    body = {"messages": msgs}
    compactor = TranscriptCompactor(
        threshold_tokens=50_000,
        keep_recent_tokens=10_000,
        chunk_messages=10,
    )
    await compactor.run(PipelineRequest(body=body, headers={}))
    snapshot = [dict(m) for m in body["messages"]]
    # Second pass on already-compacted body must change nothing.
    await compactor.run(PipelineRequest(body=body, headers={}))
    assert body["messages"] == snapshot
