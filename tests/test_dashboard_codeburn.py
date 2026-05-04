"""Tests for the codeburn-style stats endpoints — sessions, by_project,
by_tool, health.

Mirrors the structure of test_dashboard.py: ASGITransport hits each endpoint
with seeded data so the JSON shapes are locked.
"""
from __future__ import annotations

import json
import time

import httpx
import pytest


@pytest.fixture
async def dashboard_app(tmp_home):
    from tokenq.dashboard.app import app
    from tokenq.storage import init_db
    await init_db()
    return app


async def _seed(
    *,
    session_id: str = "abc123",
    project: str = "tokenq",
    tools: list[str] | None = None,
    spent: float = 0.01,
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read_tokens: int = 0,
    cached_locally: int = 0,
    saved_tokens: int = 0,
    status_code: int = 200,
    model: str = "claude-sonnet-4-6",
    ts: float | None = None,
    activity: str | None = None,
    edit_files: list[str] | None = None,
    bash_verbs: list[str] | None = None,
) -> None:
    from tokenq.storage import log_request
    await log_request(
        ts=ts if ts is not None else time.time(),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=0,
        cache_read_tokens=cache_read_tokens,
        cached_locally=cached_locally,
        saved_tokens=saved_tokens,
        saved_by_cache=saved_tokens,
        latency_ms=120,
        status_code=status_code,
        error=None,
        stream=0,
        estimated_cost_usd=spent,
        session_id=session_id,
        project=project,
        tools_used=json.dumps(tools or []),
        turn_index=1,
        activity=activity,
        edit_files=json.dumps(edit_files) if edit_files else "",
        bash_verbs=json.dumps(bash_verbs) if bash_verbs else "",
    )


# ---------- /api/sessions ----------

@pytest.mark.asyncio
async def test_sessions_empty(dashboard_app):
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/sessions")
    assert r.status_code == 200
    body = r.json()
    assert body["session_count"] == 0
    assert body["top"] == []


@pytest.mark.asyncio
async def test_sessions_groups_by_session_id(dashboard_app):
    await _seed(session_id="s1", spent=0.05)
    await _seed(session_id="s1", spent=0.03)
    await _seed(session_id="s2", spent=0.10)
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/sessions")
    body = r.json()
    assert body["session_count"] == 2
    # Top sorted by spent desc — s2 should be first.
    assert body["top"][0]["session_id"] == "s2"
    # s1 has 2 requests aggregated.
    s1 = next(s for s in body["top"] if s["session_id"] == "s1")
    assert s1["requests"] == 2
    assert abs(s1["spent_usd"] - 0.08) < 1e-9
    # avg cost calculation
    assert abs(body["avg_cost_per_session"] - 0.18 / 2) < 1e-9


# ---------- /api/by_project ----------

@pytest.mark.asyncio
async def test_by_project_groups_and_counts_distinct_sessions(dashboard_app):
    await _seed(project="tokenq", session_id="s1", spent=0.05)
    await _seed(project="tokenq", session_id="s2", spent=0.05)
    await _seed(project="other", session_id="s3", spent=0.20)
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/by_project")
    body = r.json()
    by_name = {p["project"]: p for p in body}
    assert by_name["tokenq"]["sessions"] == 2
    assert by_name["tokenq"]["requests"] == 2
    assert abs(by_name["tokenq"]["spent_usd"] - 0.10) < 1e-9
    assert abs(by_name["tokenq"]["avg_cost_per_session"] - 0.05) < 1e-9
    # Sorted by spent desc
    assert body[0]["project"] == "other"


@pytest.mark.asyncio
async def test_by_project_handles_unknown(dashboard_app):
    """A request without a project (non-Claude-Code client) shows under (unknown)."""
    await _seed(project="", session_id="s1", spent=0.01)
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/by_project")
    body = r.json()
    assert body[0]["project"] == "(unknown)"


# ---------- /api/by_tool ----------

@pytest.mark.asyncio
async def test_by_tool_attributes_request_to_each_tool_used(dashboard_app):
    await _seed(tools=["Read", "Edit"], spent=0.10, input_tokens=1000)
    await _seed(tools=["Read"], spent=0.05, input_tokens=500)
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/by_tool")
    body = r.json()
    by_name = {t["tool"]: t for t in body}
    # Read appeared in both → 2 reqs, 1500 input tokens, $0.15.
    assert by_name["Read"]["requests"] == 2
    assert by_name["Read"]["input_tokens"] == 1500
    assert abs(by_name["Read"]["spent_usd"] - 0.15) < 1e-9
    # Edit only in one.
    assert by_name["Edit"]["requests"] == 1
    assert abs(by_name["Edit"]["spent_usd"] - 0.10) < 1e-9


@pytest.mark.asyncio
async def test_by_tool_skips_rows_with_no_tools(dashboard_app):
    await _seed(tools=[], spent=0.10)
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/by_tool")
    assert r.json() == []


# ---------- /api/health ----------

@pytest.mark.asyncio
async def test_health_grade_no_traffic(dashboard_app):
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/health")
    body = r.json()
    assert body["grade"] == "—"
    assert body["requests"] == 0


@pytest.mark.asyncio
async def test_health_grade_high_when_cache_savings_clean(dashboard_app):
    """Strong upstream cache + good local savings + no errors → A or B."""
    # 10 requests, mostly upstream-cached, no errors.
    for _ in range(10):
        await _seed(
            cache_read_tokens=8000, input_tokens=2000, output_tokens=200,
            saved_tokens=200, cached_locally=1, spent=0.001,
        )
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/health")
    body = r.json()
    assert body["grade"] in ("A", "B")
    assert body["upstream_cache_pct"] >= 70


@pytest.mark.asyncio
async def test_health_grade_high_cache_low_tokenq_savings_is_not_penalized(dashboard_app):
    """Real-world: 100% upstream cache + 0% tokenq savings + 0 errors should
    grade A or B, not C. Anthropic's cache and tokenq's pipeline save the
    same tokens — if cache already caught everything, there's nothing left
    for tokenq to cut, and the user shouldn't get a worse grade for it."""
    for _ in range(10):
        await _seed(
            cache_read_tokens=10000, input_tokens=100, output_tokens=20,
            saved_tokens=0, cached_locally=0, spent=0.001,
        )
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/health")
    body = r.json()
    assert body["grade"] in ("A", "B"), (
        f"got {body['grade']} with 100% upstream cache and 0 errors — "
        f"score={body['score']}, findings={body['findings']}"
    )
    # No "tokenq savings" finding — that's the whole point.
    assert not any("savings are" in f for f in body["findings"])


@pytest.mark.asyncio
async def test_health_grade_flags_low_cache(dashboard_app):
    """Zero upstream cache + zero savings + many errors → low grade with findings."""
    for _ in range(10):
        await _seed(
            cache_read_tokens=0, input_tokens=10000,
            saved_tokens=0, status_code=500, spent=0.05,
        )
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/health")
    body = r.json()
    assert body["grade"] in ("D", "F")
    assert len(body["findings"]) >= 2  # multiple problems flagged
    assert body["error_pct"] == 100.0


# ---------- /api/by_activity ----------

@pytest.mark.asyncio
async def test_by_activity_groups_and_computes_share(dashboard_app):
    await _seed(activity="debugging", spent=0.05)
    await _seed(activity="debugging", spent=0.03)
    await _seed(activity="testing", spent=0.02)
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/by_activity")
    body = r.json()
    rows = body["rows"]
    by_name = {a["activity"]: a for a in rows}
    assert by_name["debugging"]["requests"] == 2
    assert abs(by_name["debugging"]["spent_usd"] - 0.08) < 1e-9
    # 0.08 / 0.10 = 80%
    assert abs(by_name["debugging"]["cost_share_pct"] - 80.0) < 0.01
    # Sorted by spent desc
    assert rows[0]["activity"] == "debugging"
    # No legacy rows in this seed.
    assert body["legacy"]["count"] == 0


@pytest.mark.asyncio
async def test_by_activity_excludes_null_rows_and_reports_them_as_legacy(dashboard_app):
    """A NULL/empty `activity` is pre-classifier traffic — it shouldn't
    pollute the activity table as a giant 'unclassified' bucket. Instead,
    surface the legacy count separately so the UI can flag it once."""
    await _seed(activity=None, spent=0.50)  # legacy
    await _seed(activity="", spent=0.20)    # legacy (empty string, just in case)
    await _seed(activity="debugging", spent=0.05)  # actually classified
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/by_activity")
    body = r.json()
    # Only the classified row appears in rows.
    assert [r["activity"] for r in body["rows"]] == ["debugging"]
    # Legacy bucket carries the rest.
    assert body["legacy"]["count"] == 2
    assert abs(body["legacy"]["spent_usd"] - 0.70) < 1e-9


# ---------- /api/oneshot ----------

@pytest.mark.asyncio
async def test_oneshot_detects_same_file_retry_in_session(dashboard_app):
    """Two Edits on the same file in the same session → first counts as a
    retry (because a later edit touches the file)."""
    now = time.time()
    await _seed(session_id="s1", edit_files=["/a.py"], ts=now - 30)
    await _seed(session_id="s1", edit_files=["/a.py"], ts=now)
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/oneshot")
    body = r.json()
    assert body["edit_turns"] == 2
    assert body["retry_turns"] == 1   # first turn was followed by another edit on same file
    assert body["one_shot_turns"] == 1  # second turn has no follow-up
    assert abs(body["overall_one_shot_pct"] - 50.0) < 0.01


@pytest.mark.asyncio
async def test_oneshot_different_files_count_as_one_shot(dashboard_app):
    now = time.time()
    await _seed(session_id="s1", edit_files=["/a.py"], ts=now - 30)
    await _seed(session_id="s1", edit_files=["/b.py"], ts=now)
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/oneshot")
    body = r.json()
    assert body["retry_turns"] == 0
    assert body["one_shot_turns"] == 2
    assert body["overall_one_shot_pct"] == 100.0


@pytest.mark.asyncio
async def test_oneshot_separate_sessions_dont_count(dashboard_app):
    """Same file edited in two sessions = NOT a retry — retries are scoped
    to a single session/conversation."""
    now = time.time()
    await _seed(session_id="s1", edit_files=["/a.py"], ts=now - 30)
    await _seed(session_id="s2", edit_files=["/a.py"], ts=now)
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/oneshot")
    body = r.json()
    assert body["retry_turns"] == 0
    assert body["overall_one_shot_pct"] == 100.0


@pytest.mark.asyncio
async def test_oneshot_empty_when_no_edit_turns(dashboard_app):
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/oneshot")
    body = r.json()
    assert body["edit_turns"] == 0
    assert body["per_session"] == []


# ---------- /api/by_shell_command ----------

@pytest.mark.asyncio
async def test_by_shell_command_counts_verbs(dashboard_app):
    await _seed(bash_verbs=["git", "ls"], spent=0.05)
    await _seed(bash_verbs=["git"], spent=0.03)
    await _seed(bash_verbs=["npm"], spent=0.10)
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/by_shell_command")
    body = r.json()
    by_cmd = {x["command"]: x for x in body}
    assert by_cmd["git"]["calls"] == 2
    assert abs(by_cmd["git"]["spent_usd"] - 0.08) < 1e-9
    # Sorted by calls desc — git wins.
    assert body[0]["command"] == "git"


@pytest.mark.asyncio
async def test_by_shell_command_skips_rows_without_verbs(dashboard_app):
    await _seed(bash_verbs=None, spent=0.10)
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/by_shell_command")
    assert r.json() == []


# ---------- /api/by_mcp ----------

@pytest.mark.asyncio
async def test_by_mcp_groups_by_server(dashboard_app):
    """`mcp__playwright__navigate` and `mcp__playwright__click` collapse to
    the 'playwright' server; non-MCP tools are ignored."""
    await _seed(
        tools=["mcp__playwright__navigate", "mcp__playwright__click", "Read"],
        spent=0.10,
    )
    await _seed(
        tools=["mcp__sequential-thinking__think", "Edit"],
        spent=0.05,
    )
    await _seed(tools=["Read", "Bash"], spent=1.0)  # no MCP tools — ignored
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/by_mcp")
    body = r.json()
    by_server = {x["server"]: x for x in body}
    assert "playwright" in by_server
    assert "sequential-thinking" in by_server
    # Only one credit per (server, request) — not per function.
    assert by_server["playwright"]["calls"] == 1
    assert abs(by_server["playwright"]["spent_usd"] - 0.10) < 1e-9


@pytest.mark.asyncio
async def test_by_mcp_empty_when_no_mcp_traffic(dashboard_app):
    await _seed(tools=["Read", "Bash", "Edit"], spent=0.05)
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/by_mcp")
    assert r.json() == []


# ---------- /api/by_activity per-row 1-shot ----------

@pytest.mark.asyncio
async def test_by_activity_includes_one_shot_per_activity(dashboard_app):
    """Two debugging edits on the same file → 1 retry, 1 one-shot. One
    refactoring edit (no follow-up) → 1 one-shot. Per-activity rates split."""
    now = time.time()
    await _seed(
        session_id="s1", activity="debugging",
        edit_files=["/auth.py"], ts=now - 60,
    )
    await _seed(
        session_id="s1", activity="debugging",
        edit_files=["/auth.py"], ts=now - 30,
    )
    await _seed(
        session_id="s1", activity="refactoring",
        edit_files=["/billing.py"], ts=now,
    )
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/by_activity")
    body = r.json()
    by_act = {a["activity"]: a for a in body["rows"]}
    # Debugging: 2 edits, 1 retry, 1 one-shot → 50%
    assert by_act["debugging"]["edit_turns"] == 2
    assert abs(by_act["debugging"]["one_shot_pct"] - 50.0) < 0.01
    # Refactoring: 1 edit, 1 one-shot → 100%
    assert by_act["refactoring"]["edit_turns"] == 1
    assert abs(by_act["refactoring"]["one_shot_pct"] - 100.0) < 0.01


@pytest.mark.asyncio
async def test_by_activity_one_shot_null_when_no_edits(dashboard_app):
    """An activity with no edit-bearing turns should report None (rendered
    as '—'), not 0% or 100%."""
    await _seed(activity="exploration", edit_files=None, spent=0.05)
    transport = httpx.ASGITransport(app=dashboard_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/by_activity")
    body = r.json()
    by_act = {a["activity"]: a for a in body["rows"]}
    assert by_act["exploration"]["edit_turns"] == 0
    assert by_act["exploration"]["one_shot_pct"] is None
