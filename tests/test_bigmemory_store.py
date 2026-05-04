"""Tests for BigMemoryStore — sqlite + FTS5 backend."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_add_and_get_roundtrip(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    item = await s.add(content="hello world", kind="note", source="test")
    assert item.id is not None
    fetched = await s.get(item.id)
    assert fetched is not None
    assert fetched.content == "hello world"
    assert fetched.kind == "note"


@pytest.mark.asyncio
async def test_dedup_by_content_hash(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    a = await s.add(content="ratelimit handler raises on 429", kind="fact")
    b = await s.add(content="ratelimit handler raises on 429", kind="fact")
    assert a.id == b.id
    st = await s.stats()
    assert st["total_items"] == 1


@pytest.mark.asyncio
async def test_search_bm25_ranks_relevance(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    await s.add(content="dashboard refresh function lives in index.html", kind="fact")
    await s.add(content="ratelimit handler in upstream.py raises on 429", kind="fact")
    await s.add(content="dashboard polls /api/stats every five seconds", kind="fact")

    hits = await s.search("dashboard", limit=10)
    assert len(hits) == 2
    assert all("dashboard" in h.content for h in hits)

    hits = await s.search("upstream ratelimit")
    assert len(hits) == 1
    assert "ratelimit" in hits[0].content


@pytest.mark.asyncio
async def test_search_kind_filter(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    await s.add(content="fact about caching", kind="fact")
    await s.add(content="note about caching", kind="note")
    facts = await s.search("caching", kind="fact")
    assert [h.kind for h in facts] == ["fact"]


@pytest.mark.asyncio
async def test_search_handles_special_chars(tmp_home):
    """FTS5 syntax (-, :, AND, OR) in user input must not throw."""
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    await s.add(content="path /repo/src/auth.py modified", kind="fact")
    # Each of these has FTS5-special chars; should not raise.
    for q in ["/repo/src/auth.py", "AND OR NOT", "foo-bar", "x:y", ""]:
        result = await s.search(q)
        assert isinstance(result, list)


@pytest.mark.asyncio
async def test_recent_orders_by_ts_desc(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    a = await s.add(content="first", kind="note", ts=100.0)
    b = await s.add(content="second", kind="note", ts=200.0)
    c = await s.add(content="third", kind="note", ts=300.0)
    rows = await s.recent(limit=10)
    assert [r.id for r in rows] == [c.id, b.id, a.id]


@pytest.mark.asyncio
async def test_delete(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    a = await s.add(content="ephemeral", kind="note")
    assert await s.delete(a.id) is True
    assert await s.get(a.id) is None
    assert await s.delete(a.id) is False  # already gone


@pytest.mark.asyncio
async def test_search_increments_hit_counter(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    a = await s.add(content="popular memory item", kind="fact")
    await s.search("popular")
    await s.search("popular")
    fetched = await s.get(a.id)
    assert fetched.hits == 2
    assert fetched.last_hit_ts is not None


@pytest.mark.asyncio
async def test_stats_breakdown_by_kind(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    await s.add(content="a", kind="fact")
    await s.add(content="b", kind="fact")
    await s.add(content="c", kind="note")
    st = await s.stats()
    kinds = {row["kind"]: row["n"] for row in st["by_kind"]}
    assert kinds == {"fact": 2, "note": 1}
    assert st["total_items"] == 3


# ---------- supersession / topic_key ----------

@pytest.mark.asyncio
async def test_topic_key_supersedes_prior(tmp_home):
    """Writing two items with the same topic_key marks the older one as
    superseded by the newer one — search/recent default to active rows."""
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    old = await s.add(
        content="user role: data scientist", kind="fact", topic_key="user.role",
    )
    new = await s.add(
        content="user role: ML engineer", kind="fact", topic_key="user.role",
    )
    assert new.id != old.id

    fetched_old = await s.get(old.id)
    assert fetched_old.superseded_by == new.id
    fetched_new = await s.get(new.id)
    assert fetched_new.superseded_by is None

    # Default recent + search hide superseded rows.
    rows = await s.recent()
    assert [r.id for r in rows] == [new.id]
    hits = await s.search("user role")
    assert [h.id for h in hits] == [new.id]

    # include_superseded surfaces the old row for audit.
    all_recent = await s.recent(include_superseded=True)
    assert {r.id for r in all_recent} == {old.id, new.id}


@pytest.mark.asyncio
async def test_topic_key_does_not_supersede_other_keys(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    a = await s.add(content="role: ML engineer", kind="fact", topic_key="user.role")
    b = await s.add(content="lang: Python", kind="fact", topic_key="user.language")
    assert (await s.get(a.id)).superseded_by is None
    assert (await s.get(b.id)).superseded_by is None


@pytest.mark.asyncio
async def test_duplicate_content_increments_strength(tmp_home):
    """Re-saving the same content (same hash) bumps strength — repetition is
    signal that the fact is real and protects it from pruning."""
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    a = await s.add(content="auth uses JWT", kind="fact")
    b = await s.add(content="auth uses JWT", kind="fact")
    c = await s.add(content="auth uses JWT", kind="fact")
    assert a.id == b.id == c.id
    fetched = await s.get(a.id)
    assert fetched.strength == 3


# ---------- profile (stable + recent split) ----------

@pytest.mark.asyncio
async def test_set_profile_persists_with_global_scope(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    item = await s.set_profile(key="user.role", value="data scientist")
    assert item.kind == "profile"
    assert item.scope == "global"
    assert item.topic_key == "user.role"


@pytest.mark.asyncio
async def test_set_profile_supersedes_on_update(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    first = await s.set_profile(key="user.role", value="data scientist")
    second = await s.set_profile(key="user.role", value="ML engineer")
    assert (await s.get(first.id)).superseded_by == second.id
    profile = await s.profile()
    assert [p.id for p in profile] == [second.id]
    assert profile[0].content == "ML engineer"


@pytest.mark.asyncio
async def test_profile_separates_stable_from_recent(tmp_home):
    """profile() returns only stable identity facts; recent() returns the
    rolling context — the split lets a client surface 'who is this user'
    without dredging up every recent tool_result."""
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    await s.set_profile(key="user.role", value="ML engineer")
    await s.set_profile(key="user.lang", value="Python")
    await s.add(content="just ran tests", kind="note")
    await s.add(content="big tool result blob", kind="tool_result")

    profile = await s.profile()
    assert {p.topic_key for p in profile} == {"user.role", "user.lang"}
    assert all(p.kind == "profile" for p in profile)

    recent = await s.recent()
    # Recent surfaces everything active, profile included (it's recent too).
    kinds = {r.kind for r in recent}
    assert {"note", "tool_result", "profile"} <= kinds


# ---------- TTL / decay / expire ----------

@pytest.mark.asyncio
async def test_expire_removes_aged_low_confidence_items(tmp_home):
    """tool_result has a 3-day half-life; after 50 days its decayed
    confidence is well below 0.05 and strength=1 — it should be pruned."""
    import time as _time
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    now = _time.time()
    very_old = await s.add(
        content="ancient tool output",
        kind="tool_result",
        ts=now - 50 * 86400.0,
    )
    fresh = await s.add(content="recent tool output", kind="tool_result")
    pruned = await s.expire(now=now)
    assert pruned == 1
    assert await s.get(very_old.id) is None
    assert await s.get(fresh.id) is not None


@pytest.mark.asyncio
async def test_expire_respects_kind_half_life(tmp_home):
    """A 90-day-old preference (90d half-life) sits at ~0.5 confidence — kept.
    A 90-day-old tool_result (3d half-life) is essentially zero — pruned."""
    import time as _time
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    now = _time.time()
    pref = await s.add(
        content="prefers tabs", kind="preference", ts=now - 90 * 86400.0,
    )
    tr = await s.add(
        content="ls output", kind="tool_result", ts=now - 90 * 86400.0,
    )
    await s.expire(now=now)
    assert await s.get(pref.id) is not None
    assert await s.get(tr.id) is None


@pytest.mark.asyncio
async def test_expire_keeps_strong_items(tmp_home):
    """Strength > 1 protects a row from pruning even when fully decayed —
    repetition is signal that the fact matters."""
    import time as _time
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    now = _time.time()
    # Insert with old ts.
    item = await s.add(
        content="repeated fact", kind="tool_result", ts=now - 50 * 86400.0,
    )
    # Re-add same content twice → strength becomes 3.
    await s.add(content="repeated fact", kind="tool_result")
    await s.add(content="repeated fact", kind="tool_result")
    assert (await s.get(item.id)).strength == 3
    pruned = await s.expire(now=now)
    assert pruned == 0
    assert await s.get(item.id) is not None


@pytest.mark.asyncio
async def test_expire_keeps_profile_items_long_term(tmp_home):
    """Profile items have a ~immortal half-life — stable identity facts
    must not be pruned even years later."""
    import time as _time
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    now = _time.time()
    item = await s.set_profile(key="user.role", value="ML engineer")
    # Force the row's ts to be 5 years ago.
    import aiosqlite as _aios
    async with _aios.connect(s.db_path) as db:
        await db.execute(
            "UPDATE memory_items SET ts = ? WHERE id = ?",
            (now - 5 * 365 * 86400.0, item.id),
        )
        await db.commit()
    await s.expire(now=now)
    assert await s.get(item.id) is not None


@pytest.mark.asyncio
async def test_expire_sweeps_old_superseded_rows(tmp_home):
    """Superseded rows older than 30 days are pruned regardless of confidence —
    they've already served their audit purpose."""
    import time as _time
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    now = _time.time()
    old = await s.add(
        content="old role", kind="fact", topic_key="role", ts=now - 60 * 86400.0,
    )
    await s.add(content="new role", kind="fact", topic_key="role")
    # Old is now superseded *and* >30d old.
    pruned = await s.expire(now=now)
    assert pruned >= 1
    assert await s.get(old.id) is None


@pytest.mark.asyncio
async def test_stats_reports_active_vs_superseded(tmp_home):
    from tokenq.bigmemory.store import BigMemoryStore
    s = BigMemoryStore()
    await s.init()
    await s.add(content="v1", kind="fact", topic_key="x")
    await s.add(content="v2", kind="fact", topic_key="x")
    st = await s.stats()
    assert st["total_items"] == 2
    assert st["active_items"] == 1
    assert st["superseded_items"] == 1
