"""Tests for BigMemoryInjectStage — Phase 2 active prefix injection.

The cache-stability invariant is the highest-stakes test surface in the
codebase: if injection changes the prefix between consecutive turns of the
same session, Anthropic's prompt cache misses every turn and tokenq goes
NET NEGATIVE on cost. Most tests below exist specifically to lock that
behavior down.
"""
from __future__ import annotations

import pytest


def _msg(role, text):
    return {"role": role, "content": text}


def _request(system="you are helpful", user_text="what does auth.py do?"):
    return {
        "system": system,
        "messages": [_msg("user", user_text)],
    }


# ---------- disabled-by-default ----------

@pytest.mark.asyncio
async def test_disabled_makes_no_changes(tmp_home):
    from tokenq.bigmemory.inject import BigMemoryInjectStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    store = BigMemoryStore()
    await store.init()
    await store.set_profile(key="user.role", value="ML engineer")

    stage = BigMemoryInjectStage(store=store, enabled=False)
    body = _request()
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)
    assert body["system"] == "you are helpful"
    assert "bigmemory_injected_tokens" not in req.metadata


@pytest.mark.asyncio
async def test_default_pipeline_inject_stage_is_disabled(tmp_home):
    """The wired-up default pipeline must default to disabled — flipping it
    on without explicit measurement could regress the existing cache wins."""
    from tokenq import pipeline as p

    pipe = p.default_pipeline
    inject_stages = [s for s in pipe.stages if s.name == "bigmemory_inject"]
    assert len(inject_stages) == 1
    assert inject_stages[0].enabled is False


# ---------- empty store ----------

@pytest.mark.asyncio
async def test_empty_store_no_injection(tmp_home):
    from tokenq.bigmemory.inject import BigMemoryInjectStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    store = BigMemoryStore()
    await store.init()
    stage = BigMemoryInjectStage(store=store, enabled=True)
    body = _request()
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)
    # System unchanged when there's nothing to inject.
    assert body["system"] == "you are helpful"


# ---------- injection mechanics ----------

@pytest.mark.asyncio
async def test_injection_appends_block_with_cache_control(tmp_home):
    from tokenq.bigmemory.inject import BigMemoryInjectStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    store = BigMemoryStore()
    await store.init()
    await store.set_profile(key="user.role", value="ML engineer")

    stage = BigMemoryInjectStage(store=store, enabled=True)
    body = _request()
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    assert isinstance(body["system"], list)
    assert body["system"][0] == {"type": "text", "text": "you are helpful"}
    last = body["system"][-1]
    assert last["type"] == "text"
    assert "ML engineer" in last["text"]
    assert last["cache_control"] == {"type": "ephemeral"}
    assert req.metadata["bigmemory_injected_tokens"] > 0


@pytest.mark.asyncio
async def test_injection_handles_list_form_system(tmp_home):
    """If the caller already sent system as a list (e.g. with their own
    cache_control), we append rather than overwrite."""
    from tokenq.bigmemory.inject import BigMemoryInjectStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    store = BigMemoryStore()
    await store.init()
    await store.set_profile(key="user.role", value="ML engineer")
    stage = BigMemoryInjectStage(store=store, enabled=True)

    body = {
        "system": [
            {"type": "text", "text": "first block"},
            {"type": "text", "text": "second block", "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [_msg("user", "hi")],
    }
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)
    assert len(body["system"]) == 3
    assert body["system"][0]["text"] == "first block"
    assert body["system"][1]["text"] == "second block"
    assert "ML engineer" in body["system"][2]["text"]


# ---------- THE BIG ONE: cache stability ----------

@pytest.mark.asyncio
async def test_block_is_byte_identical_across_turns_in_session(tmp_home):
    """The cache invariant. Two requests with the same (system, first_user_msg)
    MUST produce byte-identical injected blocks within the refresh window —
    otherwise Anthropic's prompt cache misses on every turn and tokenq's
    biggest savings line item evaporates.
    """
    from tokenq.bigmemory.inject import BigMemoryInjectStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    store = BigMemoryStore()
    await store.init()
    await store.set_profile(key="user.role", value="ML engineer")
    await store.add(content="auth uses JWT", kind="fact")

    stage = BigMemoryInjectStage(
        store=store, enabled=True, refresh_turns=100, refresh_secs=10_000,
    )

    blocks = []
    for turn in range(5):
        body = {
            "system": "stable system prompt",
            "messages": [
                _msg("user", "first user message — defines the session"),
                _msg("assistant", f"reply {turn}"),
                _msg("user", f"follow up question {turn}"),
            ],
        }
        req = PipelineRequest(body=body, headers={})
        # Mutate the store between turns — this MUST NOT change the snapshot.
        await store.add(content=f"some new fact captured turn {turn}", kind="fact")
        await stage.run(req)
        blocks.append(body["system"][-1]["text"])

    # All 5 injected blocks must be byte-identical despite the store changing.
    assert len(set(blocks)) == 1, (
        "Memory block diverged within session window — Anthropic prompt cache "
        "would miss every turn, destroying tokenq's biggest cost saving."
    )


@pytest.mark.asyncio
async def test_different_sessions_get_different_blocks(tmp_home):
    """Sanity check: if session_hash differs, the snapshot is computed
    independently. (Two separate conversations get separate memories.)"""
    from tokenq.bigmemory.inject import BigMemoryInjectStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    store = BigMemoryStore()
    await store.init()
    await store.set_profile(key="user.role", value="ML engineer")

    stage = BigMemoryInjectStage(store=store, enabled=True)

    body_a = {"system": "sys A", "messages": [_msg("user", "session A first msg")]}
    body_b = {"system": "sys B", "messages": [_msg("user", "session B first msg")]}
    await stage.run(PipelineRequest(body=body_a, headers={}))
    await stage.run(PipelineRequest(body=body_b, headers={}))

    # Session ids differ.
    md_a = body_a["messages"][0]
    md_b = body_b["messages"][0]
    assert md_a != md_b  # sanity, they're different requests
    # The injected blocks should be the same content (same store) BUT they
    # come from independent snapshots.
    async with __import__("aiosqlite").connect(store.db_path) as db:
        rows = await (await db.execute(
            "SELECT session_hash FROM memory_snapshots"
        )).fetchall()
    assert len({r[0] for r in rows}) == 2


# ---------- refresh policy ----------

@pytest.mark.asyncio
async def test_snapshot_refreshes_after_n_turns(tmp_home):
    """After refresh_turns uses, the snapshot is recomputed — picking up new
    facts that have been written since the last refresh."""
    from tokenq.bigmemory.inject import BigMemoryInjectStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    store = BigMemoryStore()
    await store.init()
    await store.set_profile(key="user.role", value="ML engineer")

    stage = BigMemoryInjectStage(
        store=store, enabled=True, refresh_turns=3, refresh_secs=10_000,
    )

    def _make_req():
        return PipelineRequest(
            body={"system": "sys", "messages": [_msg("user", "session start")]},
            headers={},
        )

    # First turn: cold snapshot.
    req1 = _make_req()
    await stage.run(req1)
    assert req1.metadata["bigmemory_inject_from_cache"] is False
    initial_block = req1.body["system"][-1]["text"]

    # Turns 2 and 3: should hit the snapshot.
    for _ in range(2):
        req = _make_req()
        await stage.run(req)
        assert req.metadata["bigmemory_inject_from_cache"] is True
        assert req.body["system"][-1]["text"] == initial_block

    # Add a brand-new fact between rollovers so the refresh has new content.
    await store.add(
        content="critical new fact about the auth subsystem", kind="fact",
    )

    # 4th call: use_count is now 3 (incremented on each prior call), so it's
    # past the refresh_turns threshold → recompute.
    req4 = _make_req()
    await stage.run(req4)
    assert req4.metadata["bigmemory_inject_from_cache"] is False
    refreshed_block = req4.body["system"][-1]["text"]
    assert "critical new fact" in refreshed_block
    assert refreshed_block != initial_block


@pytest.mark.asyncio
async def test_snapshot_refreshes_after_secs(tmp_home, monkeypatch):
    """If a session has been idle longer than refresh_secs, the next request
    triggers a recompute even if turn count hasn't been reached."""
    import time as _time
    from tokenq.bigmemory.inject import BigMemoryInjectStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    store = BigMemoryStore()
    await store.init()
    await store.set_profile(key="user.role", value="ML engineer")

    stage = BigMemoryInjectStage(
        store=store, enabled=True, refresh_turns=100, refresh_secs=300,
    )

    def _make_req():
        return PipelineRequest(
            body={"system": "sys", "messages": [_msg("user", "session start")]},
            headers={},
        )

    real_time = _time.time
    fake_now = [real_time()]
    monkeypatch.setattr("tokenq.bigmemory.inject.time.time", lambda: fake_now[0])

    req1 = _make_req()
    await stage.run(req1)
    assert req1.metadata["bigmemory_inject_from_cache"] is False

    # Move forward 10 minutes — past refresh_secs (300s).
    fake_now[0] += 700
    req2 = _make_req()
    await stage.run(req2)
    assert req2.metadata["bigmemory_inject_from_cache"] is False


# ---------- robustness ----------

@pytest.mark.asyncio
async def test_swallows_errors_to_protect_request(tmp_home):
    """A broken store must NEVER raise out of the pipeline stage — the
    proxy keeps serving traffic even when bigmemory is sick."""
    from tokenq.bigmemory.inject import BigMemoryInjectStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    class Broken(BigMemoryStore):
        async def init(self):
            raise RuntimeError("boom")

    stage = BigMemoryInjectStage(store=Broken(), enabled=True)
    body = _request()
    req = PipelineRequest(body=body, headers={})
    result = await stage.run(req)
    assert result is req
    # System untouched — no partial mutation.
    assert body["system"] == "you are helpful"


@pytest.mark.asyncio
async def test_no_messages_no_injection(tmp_home):
    from tokenq.bigmemory.inject import BigMemoryInjectStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    store = BigMemoryStore()
    await store.init()
    await store.set_profile(key="user.role", value="ML engineer")

    stage = BigMemoryInjectStage(store=store, enabled=True)
    body = {"system": "x", "messages": []}
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)
    assert body["system"] == "x"  # untouched


# ---------- budget ----------

@pytest.mark.asyncio
async def test_block_respects_token_budget(tmp_home):
    """A pile of huge memories must not produce an unbounded block — budget
    truncation keeps the injected prefix predictable."""
    from tokenq.bigmemory.inject import BigMemoryInjectStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    store = BigMemoryStore()
    await store.init()
    big = "x" * 500
    for i in range(50):
        await store.add(content=f"fact {i}: {big}", kind="fact")
    # Seed a profile so injection always fires regardless of query overlap.
    await store.set_profile(key="user.role", value="ML engineer")

    stage = BigMemoryInjectStage(store=store, enabled=True, budget_tokens=500)
    body = _request()
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)
    injected = body["system"][-1]["text"]
    # Allow some overhead for header/footer, but don't blow past 2x budget.
    assert len(injected) // 4 < 1000


# ---------- snapshot persistence across restart ----------

@pytest.mark.asyncio
async def test_snapshot_persists_across_stage_recreation(tmp_home):
    """Creating a fresh stage instance against the same DB must reuse the
    persisted snapshot — proxy restart shouldn't blow away cache stability."""
    from tokenq.bigmemory.inject import BigMemoryInjectStage
    from tokenq.bigmemory.store import BigMemoryStore
    from tokenq.pipeline import PipelineRequest

    store = BigMemoryStore()
    await store.init()
    await store.set_profile(key="user.role", value="ML engineer")

    stage1 = BigMemoryInjectStage(store=store, enabled=True)
    body1 = _request()
    await stage1.run(PipelineRequest(body=body1, headers={}))
    block1 = body1["system"][-1]["text"]

    # Simulate a proxy restart — fresh stage, same DB.
    stage2 = BigMemoryInjectStage(store=BigMemoryStore(), enabled=True)
    body2 = _request()  # same system + first message → same session_hash
    req2 = PipelineRequest(body=body2, headers={})
    await stage2.run(req2)
    block2 = body2["system"][-1]["text"]

    assert block1 == block2
    assert req2.metadata["bigmemory_inject_from_cache"] is True
