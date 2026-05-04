"""Dashboard ASGI app, served on a separate port (default 8090).

JSON endpoints under /api/*; static index.html at /. htmx polls /api/stats and
/api/recent every 2s for live updates.
"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..pricing import PRICING
from ..storage import db_conn, init_db

PRESET_SECONDS = {"1h": 3600, "24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400}

STATIC_DIR = Path(__file__).parent / "static"


async def index(_: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


async def stats(request: Request) -> JSONResponse:
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT
                COUNT(*) AS requests,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                COALESCE(SUM(cache_creation_tokens), 0) AS cache_creation_tokens,
                COALESCE(SUM(saved_tokens), 0) AS saved_tokens,
                COALESCE(SUM(saved_by_cache), 0) AS saved_by_cache,
                COALESCE(SUM(saved_by_dedup), 0) AS saved_by_dedup,
                COALESCE(SUM(saved_by_compress), 0) AS saved_by_compress,
                COALESCE(SUM(saved_by_skills), 0) AS saved_by_skills,
                COALESCE(SUM(saved_by_bandit_usd), 0) AS saved_by_bandit_usd,
                COALESCE(SUM(CASE WHEN saved_by_cache > 0 THEN 1 ELSE 0 END), 0) AS reqs_cache,
                COALESCE(SUM(CASE WHEN saved_by_dedup > 0 THEN 1 ELSE 0 END), 0) AS reqs_dedup,
                COALESCE(SUM(CASE WHEN saved_by_compress > 0 THEN 1 ELSE 0 END), 0) AS reqs_compress,
                COALESCE(SUM(CASE WHEN saved_by_skills > 0 THEN 1 ELSE 0 END), 0) AS reqs_skills,
                COALESCE(SUM(CASE WHEN saved_by_bandit_usd > 0 THEN 1 ELSE 0 END), 0) AS reqs_bandit,
                COALESCE(SUM(estimated_cost_usd), 0) AS spent_usd,
                COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                COALESCE(SUM(cached_locally), 0) AS local_cache_hits,
                COALESCE(SUM(CASE WHEN saved_tokens > 0 THEN 1 ELSE 0 END), 0) AS requests_with_savings
            FROM requests
            WHERE ts >= ? AND ts < ?
            """,
            (start, end),
        )
        row = await cur.fetchone()
        result = dict(row) if row else {}

        # Cumulative skill-compress CLI stats (all-time, since these are
        # offline rewrites — not bound to the 24h request window).
        cur2 = await db.execute(
            """
            SELECT
                COUNT(*) AS skill_compressions,
                COALESCE(SUM(saved_tokens), 0) AS skill_compress_saved_tokens,
                COALESCE(SUM(before_tokens), 0) AS skill_compress_before_tokens,
                COALESCE(SUM(after_tokens), 0) AS skill_compress_after_tokens
            FROM skill_compressions
            """
        )
        sc = await cur2.fetchone()
        if sc:
            result.update(dict(sc))

        # Compaction events in the selected window. Logged only on rollover
        # turns (cache_read==0 + we compacted), so each row is a real, one-off
        # rebuild that strips dropped_tokens from the prefix every subsequent
        # turn until the next rollover.
        cur3 = await db.execute(
            """
            SELECT
                COUNT(*) AS compaction_events,
                COALESCE(SUM(dropped_tokens), 0) AS compaction_dropped_tokens,
                COALESCE(SUM(saved_per_turn), 0) AS compaction_saved_per_turn
            FROM compaction_events
            WHERE ts >= ? AND ts < ?
            """,
            (start, end),
        )
        ce = await cur3.fetchone()
        if ce:
            result.update(dict(ce))

        # Compute saved_usd correctly: cache hits price input + output, others
        # only input. Done per-model because rates differ per family.
        cur4 = await db.execute(
            """
            SELECT
                CASE WHEN model IS NULL OR model = '' THEN '(other)' ELSE model END AS model,
                COALESCE(SUM(CASE WHEN cached_locally = 1 THEN input_tokens ELSE 0 END), 0)
                    AS cache_saved_input,
                COALESCE(SUM(CASE WHEN cached_locally = 1 THEN output_tokens ELSE 0 END), 0)
                    AS cache_saved_output,
                COALESCE(SUM(CASE WHEN cached_locally = 0 THEN saved_tokens ELSE 0 END), 0)
                    AS noncache_saved_input
            FROM requests
            WHERE ts >= ? AND ts < ?
            GROUP BY model
            """,
            (start, end),
        )
        rows = await cur4.fetchall()
        saved_usd = sum(_saved_usd_for_row(dict(r)) for r in rows)
        # Bandit savings are USD-denominated (model-tier price differential),
        # not token-denominated, so they're added directly rather than priced.
        saved_by_bandit_usd = float(result.get("saved_by_bandit_usd") or 0.0)
        result["saved_by_bandit_usd"] = round(saved_by_bandit_usd, 6)
        result["saved_usd"] = round(saved_usd + saved_by_bandit_usd, 6)

        # Shadow-mode counterfactual savings: bandit observed but never routed.
        # Surface separately so it isn't conflated with realized savings.
        cur5 = await db.execute(
            """
            SELECT
                COUNT(*) AS shadow_decisions,
                COALESCE(SUM(est_cost_saved_usd), 0) AS shadow_saved_usd,
                COALESCE(SUM(est_cost_full_usd), 0) AS shadow_cost_full_usd,
                COALESCE(SUM(est_cost_routed_usd), 0) AS shadow_cost_routed_usd
            FROM bandit_shadow_decisions
            WHERE ts >= ? AND ts < ?
            """,
            (start, end),
        )
        sh = await cur5.fetchone()
        if sh:
            sh_d = dict(sh)
            result["shadow_decisions"] = int(sh_d.get("shadow_decisions") or 0)
            result["shadow_saved_usd"] = round(
                float(sh_d.get("shadow_saved_usd") or 0.0), 6
            )
            result["shadow_cost_full_usd"] = round(
                float(sh_d.get("shadow_cost_full_usd") or 0.0), 6
            )
            result["shadow_cost_routed_usd"] = round(
                float(sh_d.get("shadow_cost_routed_usd") or 0.0), 6
            )

        # Bigmemory is capture-only in v1: tool_results written to memory_items,
        # MCP retrievals bump hits + last_hit_ts. tokens_hit is a lower-bound
        # estimate of input tokens that didn't have to be re-fetched.
        try:
            cur6 = await db.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN ts >= ? AND ts < ? THEN 1 ELSE 0 END), 0)
                        AS bigmemory_items_captured,
                    COALESCE(SUM(CASE WHEN ts >= ? AND ts < ? THEN tokens ELSE 0 END), 0)
                        AS bigmemory_tokens_captured,
                    COALESCE(SUM(CASE WHEN last_hit_ts >= ? AND last_hit_ts < ? THEN 1 ELSE 0 END), 0)
                        AS bigmemory_items_hit,
                    COALESCE(SUM(CASE WHEN last_hit_ts >= ? AND last_hit_ts < ? THEN tokens ELSE 0 END), 0)
                        AS bigmemory_tokens_hit
                FROM memory_items
                """,
                (start, end, start, end, start, end, start, end),
            )
            bm = await cur6.fetchone()
            if bm:
                result.update(dict(bm))
        except Exception:
            pass
        return JSONResponse(result)


async def recent(_: Request) -> JSONResponse:
    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT id, ts, model, input_tokens, output_tokens,
                   cache_read_tokens, cache_creation_tokens,
                   latency_ms, cached_locally, status_code,
                   estimated_cost_usd, stream
            FROM requests
            ORDER BY id DESC
            LIMIT 50
            """
        )
        rows = await cur.fetchall()
        return JSONResponse([dict(r) for r in rows])


async def timeseries(request: Request) -> JSONResponse:
    """Time-bucketed series for the chart. Bucket size adapts: hourly for
    windows ≤ 3 days, daily otherwise — keeps the chart legible at any zoom."""
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    span = max(end - start, 1.0)
    bucket_seconds = 3600 if span <= 3 * 86400 else 86400
    async with db_conn() as db:
        cur = await db.execute(
            f"""
            SELECT
                CAST(ts / {bucket_seconds} AS INTEGER) * {bucket_seconds} AS bucket,
                COUNT(*) AS requests,
                COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens,
                COALESCE(SUM(saved_tokens), 0) AS saved_tokens,
                COALESCE(SUM(estimated_cost_usd), 0) AS spent_usd
            FROM requests
            WHERE ts >= ? AND ts < ?
            GROUP BY bucket
            ORDER BY bucket
            """,
            (start, end),
        )
        rows = await cur.fetchall()
        return JSONResponse([dict(r) for r in rows])


async def by_model(request: Request) -> JSONResponse:
    """Per-model rollup over the selected window. Empty model rows (passthrough)
    are grouped under '(other)' so /v1/models style traffic is still visible."""
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT
                CASE WHEN model IS NULL OR model = '' THEN '(other)' ELSE model END AS model,
                COUNT(*) AS requests,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                COALESCE(SUM(cache_creation_tokens), 0) AS cache_creation_tokens,
                COALESCE(SUM(saved_tokens), 0) AS saved_tokens,
                COALESCE(SUM(estimated_cost_usd), 0) AS spent_usd,
                COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                COALESCE(SUM(cached_locally), 0) AS local_cache_hits
            FROM requests
            WHERE ts >= ? AND ts < ?
            GROUP BY model
            ORDER BY spent_usd DESC, requests DESC
            """,
            (start, end),
        )
        rows = await cur.fetchall()
        return JSONResponse([dict(r) for r in rows])


async def skill_compressions(_: Request) -> JSONResponse:
    """Recent CLI `tokenq compress-skill` rewrites — offline, not bound to 24h."""
    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT id, ts, path, output_path, model,
                   before_tokens, after_tokens, saved_tokens
            FROM skill_compressions
            ORDER BY id DESC
            LIMIT 50
            """
        )
        rows = await cur.fetchall()
        return JSONResponse([dict(r) for r in rows])


async def compactions(request: Request) -> JSONResponse:
    """Transcript-compaction rollover events in the selected window. Each row
    is one upstream cache rebuild on a smaller prefix — stripped tokens stop
    being charged on every subsequent turn until the next rollover."""
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT id, ts, model, dropped_messages, dropped_tokens,
                   summary_tokens, saved_per_turn
            FROM compaction_events
            WHERE ts >= ? AND ts < ?
            ORDER BY id DESC
            LIMIT 50
            """,
            (start, end),
        )
        rows = await cur.fetchall()
        return JSONResponse([dict(r) for r in rows])


async def expensive(request: Request) -> JSONResponse:
    """Top 10 most expensive requests in the selected window."""
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT id, ts, model, input_tokens, output_tokens,
                   cache_read_tokens, cache_creation_tokens,
                   latency_ms, status_code, estimated_cost_usd, stream
            FROM requests
            WHERE ts >= ? AND ts < ?
              AND estimated_cost_usd > 0
            ORDER BY estimated_cost_usd DESC
            LIMIT 10
            """,
            (start, end),
        )
        rows = await cur.fetchall()
        return JSONResponse([dict(r) for r in rows])


async def sessions(request: Request) -> JSONResponse:
    """Per-session rollup. A session = stable hash of (system, first user msg).

    Returns the top expensive sessions plus aggregate counts so the dashboard
    can show 'X sessions, $Y avg/session, here are the top 5'.
    """
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT
                COALESCE(NULLIF(session_id, ''), '(unknown)') AS session_id,
                COALESCE(NULLIF(project, ''), '(unknown)') AS project,
                COUNT(*) AS requests,
                MIN(ts) AS started_at,
                MAX(ts) AS ended_at,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                COALESCE(SUM(estimated_cost_usd), 0) AS spent_usd,
                COALESCE(SUM(saved_tokens), 0) AS saved_tokens
            FROM requests
            WHERE ts >= ? AND ts < ?
            GROUP BY session_id
            ORDER BY spent_usd DESC, requests DESC
            LIMIT 50
            """,
            (start, end),
        )
        rows = [dict(r) for r in await cur.fetchall()]

        cur2 = await db.execute(
            """
            SELECT
                COUNT(DISTINCT COALESCE(NULLIF(session_id, ''), '(unknown)')) AS session_count,
                COALESCE(SUM(estimated_cost_usd), 0) AS spent_usd
            FROM requests
            WHERE ts >= ? AND ts < ?
            """,
            (start, end),
        )
        totals = dict(await cur2.fetchone() or {})
    n = max(1, int(totals.get("session_count") or 0))
    return JSONResponse({
        "session_count": int(totals.get("session_count") or 0),
        "spent_usd": float(totals.get("spent_usd") or 0.0),
        "avg_cost_per_session": float(totals.get("spent_usd") or 0.0) / n,
        "top": rows,
    })


async def by_project(request: Request) -> JSONResponse:
    """Per-project rollup keyed by working-directory basename."""
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT
                COALESCE(NULLIF(project, ''), '(unknown)') AS project,
                COUNT(*) AS requests,
                COUNT(DISTINCT NULLIF(session_id, '')) AS sessions,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                COALESCE(SUM(estimated_cost_usd), 0) AS spent_usd,
                COALESCE(SUM(saved_tokens), 0) AS saved_tokens,
                COALESCE(SUM(cached_locally), 0) AS local_cache_hits,
                COALESCE(AVG(latency_ms), 0) AS avg_latency_ms
            FROM requests
            WHERE ts >= ? AND ts < ?
            GROUP BY project
            ORDER BY spent_usd DESC, requests DESC
            """,
            (start, end),
        )
        rows = [dict(r) for r in await cur.fetchall()]
    for r in rows:
        s = max(1, int(r["sessions"] or 0))
        r["avg_cost_per_session"] = float(r["spent_usd"]) / s
    return JSONResponse(rows)


async def by_tool(request: Request) -> JSONResponse:
    """Per-tool rollup. Each request's tools_used JSON is a list of tools
    referenced in the message thread; we attribute the request's input tokens
    and cost to each tool that appears (so a request that used Edit AND Read
    counts under both — this is by design, the model used both).
    """
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT tools_used, input_tokens, output_tokens,
                   estimated_cost_usd, cached_locally
            FROM requests
            WHERE ts >= ? AND ts < ?
              AND tools_used IS NOT NULL AND tools_used != ''
            """,
            (start, end),
        )
        rows = await cur.fetchall()

    import json as _json
    agg: dict[str, dict[str, float]] = {}
    for row in rows:
        try:
            tools = _json.loads(row["tools_used"]) or []
        except (TypeError, ValueError):
            continue
        if not isinstance(tools, list):
            continue
        for t in tools:
            if not isinstance(t, str):
                continue
            slot = agg.setdefault(t, {
                "tool": t, "requests": 0, "input_tokens": 0,
                "output_tokens": 0, "spent_usd": 0.0, "cache_hits": 0,
            })
            slot["requests"] += 1
            slot["input_tokens"] += int(row["input_tokens"] or 0)
            slot["output_tokens"] += int(row["output_tokens"] or 0)
            slot["spent_usd"] += float(row["estimated_cost_usd"] or 0.0)
            slot["cache_hits"] += int(row["cached_locally"] or 0)
    out = sorted(agg.values(), key=lambda r: r["spent_usd"], reverse=True)
    return JSONResponse(out)


async def by_activity(request: Request) -> JSONResponse:
    """Per-activity rollup with per-row 1-shot rate.

    Activity labels come from the codeburn-style classifier run at intercept
    time. Rows logged before the classifier shipped have activity IS NULL —
    they're excluded from the table and surfaced separately as `legacy_count`
    so the dashboard can show "N rows from before the classifier" instead of
    polluting the table with a giant 'unclassified' bucket.

    The 1-shot column counts edit-bearing rows whose Edit-on-file isn't
    followed by another Edit on the same file inside the same session — same
    rule as /api/oneshot, but split per activity so the dashboard can show
    'debugging is 60% one-shot, refactoring is 90% one-shot'.
    """
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT
                activity,
                COUNT(*) AS requests,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(estimated_cost_usd), 0) AS spent_usd,
                COALESCE(SUM(saved_tokens), 0) AS saved_tokens,
                COALESCE(AVG(latency_ms), 0) AS avg_latency_ms
            FROM requests
            WHERE ts >= ? AND ts < ?
              AND activity IS NOT NULL
              AND activity != ''
            GROUP BY activity
            ORDER BY spent_usd DESC, requests DESC
            """,
            (start, end),
        )
        rows = [dict(r) for r in await cur.fetchall()]

        # Count pre-classifier rows in the window so the UI can flag them.
        cur_legacy = await db.execute(
            """
            SELECT COUNT(*) AS n,
                   COALESCE(SUM(estimated_cost_usd), 0) AS spent_usd,
                   datetime(MIN(ts), 'unixepoch', 'localtime') AS first,
                   datetime(MAX(ts), 'unixepoch', 'localtime') AS last
            FROM requests
            WHERE ts >= ? AND ts < ?
              AND (activity IS NULL OR activity = '')
            """,
            (start, end),
        )
        legacy = dict(await cur_legacy.fetchone() or {})

        # 1-shot per activity: pull all edit-bearing rows, group by session,
        # then re-bucket by activity so each activity gets its own counts.
        cur2 = await db.execute(
            """
            SELECT id, session_id, ts, activity, edit_files
            FROM requests
            WHERE ts >= ? AND ts < ?
              AND edit_files IS NOT NULL AND edit_files != ''
              AND session_id IS NOT NULL AND session_id != ''
            ORDER BY session_id, ts
            """,
            (start, end),
        )
        edit_rows = [dict(r) for r in await cur2.fetchall()]

    import json as _json
    by_session: dict[str, list[dict]] = {}
    for r in edit_rows:
        try:
            files = _json.loads(r["edit_files"]) or []
        except (TypeError, ValueError):
            files = []
        if not files:
            continue
        r["_files"] = set(files)
        by_session.setdefault(r["session_id"], []).append(r)

    one_by_act: dict[str, int] = {}
    retry_by_act: dict[str, int] = {}
    for sid, srows in by_session.items():
        for i, row in enumerate(srows):
            is_retry = False
            for nxt in srows[i + 1: i + 4]:
                if row["_files"] & nxt["_files"]:
                    is_retry = True
                    break
            act = row["activity"] or "unclassified"
            if is_retry:
                retry_by_act[act] = retry_by_act.get(act, 0) + 1
            else:
                one_by_act[act] = one_by_act.get(act, 0) + 1

    total_cost = sum(r["spent_usd"] for r in rows) or 1.0
    for r in rows:
        act = r["activity"]
        r["cost_share_pct"] = 100.0 * r["spent_usd"] / total_cost
        one = one_by_act.get(act, 0)
        retry = retry_by_act.get(act, 0)
        edit_total = one + retry
        r["edit_turns"] = edit_total
        r["one_shot_pct"] = (100.0 * one / edit_total) if edit_total else None
    return JSONResponse({
        "rows": rows,
        "legacy": {
            "count": int(legacy.get("n") or 0),
            "spent_usd": float(legacy.get("spent_usd") or 0.0),
            "first": legacy.get("first"),
            "last": legacy.get("last"),
        },
    })


async def by_shell_command(request: Request) -> JSONResponse:
    """Per-bash-verb rollup. `bash_verbs` is a JSON list of leading verbs
    captured from each turn (e.g. ['git','npm']). Same attribution rule as
    by_tool: a turn that ran git AND ls counts under both."""
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT bash_verbs, estimated_cost_usd, input_tokens
            FROM requests
            WHERE ts >= ? AND ts < ?
              AND bash_verbs IS NOT NULL AND bash_verbs != ''
            """,
            (start, end),
        )
        rows = await cur.fetchall()
    import json as _json
    agg: dict[str, dict[str, float]] = {}
    for row in rows:
        try:
            verbs = _json.loads(row["bash_verbs"]) or []
        except (TypeError, ValueError):
            continue
        if not isinstance(verbs, list):
            continue
        for v in verbs:
            if not isinstance(v, str):
                continue
            slot = agg.setdefault(v, {
                "command": v, "calls": 0, "input_tokens": 0, "spent_usd": 0.0,
            })
            slot["calls"] += 1
            slot["input_tokens"] += int(row["input_tokens"] or 0)
            slot["spent_usd"] += float(row["estimated_cost_usd"] or 0.0)
    out = sorted(agg.values(), key=lambda r: r["calls"], reverse=True)
    return JSONResponse(out)


async def by_mcp(request: Request) -> JSONResponse:
    """Per-MCP-server rollup.

    Claude's MCP convention names tools `mcp__<server>__<function>`. We
    extract the middle segment as the server name and group by it, so users
    see 'playwright = 24 calls, sequential-thinking = 8' rather than the
    flat tool list. Non-MCP tools are ignored here (they show up in by_tool).
    """
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT tools_used, estimated_cost_usd, input_tokens
            FROM requests
            WHERE ts >= ? AND ts < ?
              AND tools_used IS NOT NULL AND tools_used != ''
            """,
            (start, end),
        )
        rows = await cur.fetchall()
    import json as _json
    agg: dict[str, dict[str, float]] = {}
    for row in rows:
        try:
            tools = _json.loads(row["tools_used"]) or []
        except (TypeError, ValueError):
            continue
        if not isinstance(tools, list):
            continue
        # One credit per (server, request) pair — collapse multiple functions
        # of the same server into one increment per turn.
        servers_this_row: set[str] = set()
        for t in tools:
            if not isinstance(t, str) or not t.startswith("mcp__"):
                continue
            parts = t.split("__", 2)
            if len(parts) < 2 or not parts[1]:
                continue
            servers_this_row.add(parts[1])
        for server in servers_this_row:
            slot = agg.setdefault(server, {
                "server": server, "calls": 0, "input_tokens": 0, "spent_usd": 0.0,
            })
            slot["calls"] += 1
            slot["input_tokens"] += int(row["input_tokens"] or 0)
            slot["spent_usd"] += float(row["estimated_cost_usd"] or 0.0)
    out = sorted(agg.values(), key=lambda r: r["calls"], reverse=True)
    return JSONResponse(out)


async def oneshot(request: Request) -> JSONResponse:
    """One-shot rate per session.

    Pattern (from codeburn): an Edit on file X is a 'retry' if the same
    session edits the same file again later. We compute per-session via
    LAG-style self-join: for each Edit-bearing request, look at the next
    Edit-bearing request in the session and check for file overlap.

    The endpoint returns a tuple (one_shot_count, retry_count) per session
    plus an overall rate. Editor turns with no follow-up edit count as
    one-shot (best case); turns followed by a same-file edit count as retry.
    """
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT id, session_id, project, ts, edit_files
            FROM requests
            WHERE ts >= ? AND ts < ?
              AND edit_files IS NOT NULL AND edit_files != ''
              AND session_id IS NOT NULL AND session_id != ''
            ORDER BY session_id, ts
            """,
            (start, end),
        )
        edit_rows = [dict(r) for r in await cur.fetchall()]

    import json as _json
    # Group edit-bearing rows by session, ordered by ts.
    by_session: dict[str, list[dict]] = {}
    for r in edit_rows:
        try:
            files = _json.loads(r["edit_files"]) or []
        except (TypeError, ValueError):
            files = []
        if not files:
            continue
        r["_files"] = set(files)
        by_session.setdefault(r["session_id"], []).append(r)

    per_session = []
    overall_one = 0
    overall_retry = 0
    for sid, rows in by_session.items():
        one = retry = 0
        for i, row in enumerate(rows):
            # An edit turn is a retry if any later edit turn in the same
            # session touches an overlapping file. We only need to look at
            # the next few turns — retries usually happen within 1-3 turns.
            is_retry = False
            for nxt in rows[i + 1: i + 4]:
                if row["_files"] & nxt["_files"]:
                    is_retry = True
                    break
            if is_retry:
                retry += 1
            else:
                one += 1
        total = one + retry
        if total == 0:
            continue
        per_session.append({
            "session_id": sid,
            "project": rows[0]["project"] or "",
            "edit_turns": total,
            "one_shot": one,
            "retries": retry,
            "one_shot_pct": 100.0 * one / total,
        })
        overall_one += one
        overall_retry += retry

    overall_total = overall_one + overall_retry
    overall_rate = 100.0 * overall_one / overall_total if overall_total else 0.0
    per_session.sort(key=lambda r: r["edit_turns"], reverse=True)
    return JSONResponse({
        "overall_one_shot_pct": overall_rate,
        "edit_turns": overall_total,
        "one_shot_turns": overall_one,
        "retry_turns": overall_retry,
        "per_session": per_session[:20],
    })


async def health(request: Request) -> JSONResponse:
    """Derive a simple A–F health grade from cache hit rate, savings %, and
    error rate. Surfaces a list of one-line findings the user can act on.
    """
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT
                COUNT(*) AS requests,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                COALESCE(SUM(estimated_cost_usd), 0) AS spent_usd,
                COALESCE(SUM(saved_tokens), 0) AS saved_tokens,
                COALESCE(SUM(cached_locally), 0) AS local_cache_hits,
                COALESCE(SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END), 0) AS errors
            FROM requests
            WHERE ts >= ? AND ts < ?
            """,
            (start, end),
        )
        t = dict(await cur.fetchone() or {})

        cur2 = await db.execute(
            """
            SELECT
                CASE WHEN model IS NULL OR model = '' THEN '(other)' ELSE model END AS model,
                COALESCE(SUM(CASE WHEN cached_locally = 1 THEN input_tokens ELSE 0 END), 0)
                    AS cache_saved_input,
                COALESCE(SUM(CASE WHEN cached_locally = 1 THEN output_tokens ELSE 0 END), 0)
                    AS cache_saved_output,
                COALESCE(SUM(CASE WHEN cached_locally = 0 THEN saved_tokens ELSE 0 END), 0)
                    AS noncache_saved_input
            FROM requests
            WHERE ts >= ? AND ts < ?
            GROUP BY model
            """,
            (start, end),
        )
        saved_usd = sum(_saved_usd_for_row(dict(r)) for r in await cur2.fetchall())

    requests = int(t.get("requests") or 0)
    input_total = int(t.get("input_tokens") or 0) + int(t.get("cache_read_tokens") or 0)
    upstream_cache_pct = (
        100.0 * (t.get("cache_read_tokens") or 0) / input_total
        if input_total else 0.0
    )
    local_cache_pct = (
        100.0 * (t.get("local_cache_hits") or 0) / requests
        if requests else 0.0
    )
    spent = float(t.get("spent_usd") or 0.0)
    counterfactual = spent + saved_usd
    savings_pct = (saved_usd / counterfactual * 100.0) if counterfactual > 0 else 0.0
    error_pct = 100.0 * int(t.get("errors") or 0) / requests if requests else 0.0

    findings: list[str] = []
    score = 0
    # Upstream cache contribution — Anthropic prompt cache. Below 50% is
    # leaving real money on the table on a long Claude Code session.
    if upstream_cache_pct >= 70:
        score += 35
    elif upstream_cache_pct >= 40:
        score += 20
        findings.append(
            f"Upstream cache hit rate is {upstream_cache_pct:.0f}% — "
            "could be higher with stable system prompts."
        )
    else:
        findings.append(
            f"Upstream cache hit rate is only {upstream_cache_pct:.0f}% — "
            "the prefix is likely changing per turn. Check whether something "
            "is mutating system or early messages."
        )

    # Savings band — but don't penalize when upstream cache is already doing
    # the job. Anthropic's prompt cache and tokenq's pipeline stages save
    # the SAME tokens; if cache_read is already high, there's no waste left
    # for tokenq to cut and a low savings number is a feature, not a bug.
    if savings_pct >= 30:
        score += 35
    elif savings_pct >= 10:
        score += 20
    elif upstream_cache_pct >= 70:
        # Cache is doing the work — give full credit, no finding.
        score += 35
    else:
        findings.append(
            f"tokenq savings are {savings_pct:.1f}% with upstream cache at "
            f"{upstream_cache_pct:.0f}% — pipeline stages aren't firing. "
            "Consider enabling bandit routing or compaction."
        )

    if requests > 0 and error_pct < 1:
        score += 30
    elif error_pct < 5:
        score += 15
    else:
        findings.append(
            f"Error rate is {error_pct:.1f}% — investigate upstream 4xx/5xx."
        )

    if requests == 0:
        grade = "—"
    elif score >= 90:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"

    return JSONResponse({
        "grade": grade,
        "score": score,
        "requests": requests,
        "spent_usd": spent,
        "saved_usd": saved_usd,
        "savings_pct": savings_pct,
        "upstream_cache_pct": upstream_cache_pct,
        "local_cache_pct": local_cache_pct,
        "error_pct": error_pct,
        "findings": findings,
    })


def _parse_window(request: Request) -> tuple[float, float] | JSONResponse:
    """Resolve ?from=&to= or ?preset= into a (start, end) unix-seconds tuple.

    Returns a 400 JSONResponse on bad input. Defaults to last 24h if nothing is
    supplied.
    """
    qp = request.query_params
    now = time.time()
    preset = qp.get("preset")
    if preset:
        secs = PRESET_SECONDS.get(preset)
        if secs is None:
            return JSONResponse(
                {"error": f"unknown preset {preset!r}; expected one of {list(PRESET_SECONDS)}"},
                status_code=400,
            )
        return (now - secs, now)

    raw_from = qp.get("from")
    raw_to = qp.get("to")
    if raw_from is None and raw_to is None:
        return (now - 86400, now)
    try:
        start = float(raw_from) if raw_from is not None else now - 86400
        end = float(raw_to) if raw_to is not None else now
    except ValueError:
        return JSONResponse({"error": "from/to must be unix-seconds numbers"}, status_code=400)
    if end <= start:
        return JSONResponse({"error": "to must be greater than from"}, status_code=400)
    return (start, end)


def _input_rate(model: str | None) -> float:
    """USD per token at base input rate. Used to monetize saved input tokens."""
    if not model:
        return 0.0
    key = next((k for k in PRICING if model.startswith(k)), None)
    if not key:
        return 0.0
    return PRICING[key]["input"] / 1_000_000


def _output_rate(model: str | None) -> float:
    """USD per token at output rate. Used to monetize saved output tokens
    on cache hits (where the model would have generated those tokens)."""
    if not model:
        return 0.0
    key = next((k for k in PRICING if model.startswith(k)), None)
    if not key:
        return 0.0
    return PRICING[key]["output"] / 1_000_000


def _saved_usd_for_row(row: dict) -> float:
    """Correctly monetize saved tokens for one aggregated row.

    Cache hits save BOTH the would-have input AND the would-have output (the
    model never ran). Non-cache savings (dedup/compress/skills) only trim
    input. The dashboard previously priced everything at input rate, which
    undercounted cache-hit output savings by ~5×.
    """
    model = row.get("model")
    in_rate = _input_rate(model)
    out_rate = _output_rate(model)
    cache_in = row.get("cache_saved_input", 0) or 0
    cache_out = row.get("cache_saved_output", 0) or 0
    noncache_in = row.get("noncache_saved_input", 0) or 0
    return cache_in * in_rate + cache_out * out_rate + noncache_in * in_rate


async def _collect_report(start: float, end: float) -> dict:
    """Single source of truth for /api/report and /api/report.pdf."""
    span = max(end - start, 1.0)
    # Hourly buckets for windows up to ~3 days, daily otherwise. Keeps the
    # series digestible whether the user picks 1h or 30d.
    bucket_seconds = 3600 if span <= 3 * 86400 else 86400

    async with db_conn() as db:
        cur = await db.execute(
            """
            SELECT
                COUNT(*) AS requests,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                COALESCE(SUM(cache_creation_tokens), 0) AS cache_creation_tokens,
                COALESCE(SUM(saved_tokens), 0) AS saved_tokens,
                COALESCE(SUM(saved_by_cache), 0) AS saved_by_cache,
                COALESCE(SUM(saved_by_dedup), 0) AS saved_by_dedup,
                COALESCE(SUM(saved_by_compress), 0) AS saved_by_compress,
                COALESCE(SUM(saved_by_skills), 0) AS saved_by_skills,
                COALESCE(SUM(saved_by_bandit_usd), 0) AS saved_by_bandit_usd,
                COALESCE(SUM(CASE WHEN saved_by_cache > 0 THEN 1 ELSE 0 END), 0) AS reqs_cache,
                COALESCE(SUM(CASE WHEN saved_by_dedup > 0 THEN 1 ELSE 0 END), 0) AS reqs_dedup,
                COALESCE(SUM(CASE WHEN saved_by_compress > 0 THEN 1 ELSE 0 END), 0) AS reqs_compress,
                COALESCE(SUM(CASE WHEN saved_by_skills > 0 THEN 1 ELSE 0 END), 0) AS reqs_skills,
                COALESCE(SUM(CASE WHEN saved_by_bandit_usd > 0 THEN 1 ELSE 0 END), 0) AS reqs_bandit,
                COALESCE(SUM(estimated_cost_usd), 0) AS spent_usd,
                COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                COALESCE(SUM(cached_locally), 0) AS local_cache_hits,
                COALESCE(SUM(CASE WHEN saved_tokens > 0 THEN 1 ELSE 0 END), 0)
                    AS requests_with_savings
            FROM requests
            WHERE ts >= ? AND ts < ?
            """,
            (start, end),
        )
        totals = dict((await cur.fetchone()) or {})

        cur = await db.execute(
            """
            SELECT
                CASE WHEN model IS NULL OR model = '' THEN '(other)' ELSE model END AS model,
                COUNT(*) AS requests,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                COALESCE(SUM(cache_creation_tokens), 0) AS cache_creation_tokens,
                COALESCE(SUM(saved_tokens), 0) AS saved_tokens,
                COALESCE(SUM(CASE WHEN cached_locally = 1 THEN input_tokens ELSE 0 END), 0)
                    AS cache_saved_input,
                COALESCE(SUM(CASE WHEN cached_locally = 1 THEN output_tokens ELSE 0 END), 0)
                    AS cache_saved_output,
                COALESCE(SUM(CASE WHEN cached_locally = 0 THEN saved_tokens ELSE 0 END), 0)
                    AS noncache_saved_input,
                COALESCE(SUM(saved_by_bandit_usd), 0) AS saved_by_bandit_usd,
                COALESCE(SUM(estimated_cost_usd), 0) AS spent_usd,
                COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                COALESCE(SUM(cached_locally), 0) AS local_cache_hits
            FROM requests
            WHERE ts >= ? AND ts < ?
            GROUP BY model
            ORDER BY spent_usd DESC, requests DESC
            """,
            (start, end),
        )
        by_model_rows = [dict(r) for r in await cur.fetchall()]

        # Monetize savings per model: cache hits save input AT input rate AND
        # output AT output rate (the model would have generated those tokens);
        # non-cache savings (dedup/compress/skills) only trim input. Bandit
        # savings are USD-denominated (price differential between routed and
        # original tier) and live on the routed-to row, so they're added to
        # that model's saved_usd directly.
        saved_usd_by_model: dict[str, float] = {}
        for row in by_model_rows:
            base = _saved_usd_for_row(row)
            bandit_usd = float(row.get("saved_by_bandit_usd") or 0.0)
            row["saved_usd"] = base + bandit_usd
            saved_usd_by_model[row["model"]] = row["saved_usd"]
        saved_usd_total = sum(saved_usd_by_model.values())

        cur = await db.execute(
            """
            SELECT id, ts, model, input_tokens, output_tokens,
                   cache_read_tokens, cache_creation_tokens,
                   latency_ms, status_code, estimated_cost_usd, stream
            FROM requests
            WHERE ts >= ? AND ts < ?
              AND estimated_cost_usd > 0
            ORDER BY estimated_cost_usd DESC
            LIMIT 10
            """,
            (start, end),
        )
        expensive_rows = [dict(r) for r in await cur.fetchall()]

        cur = await db.execute(
            f"""
            SELECT
                CAST(ts / {bucket_seconds} AS INTEGER) * {bucket_seconds} AS bucket,
                COUNT(*) AS requests,
                COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens,
                COALESCE(SUM(saved_tokens), 0) AS saved_tokens,
                COALESCE(SUM(estimated_cost_usd), 0) AS spent_usd
            FROM requests
            WHERE ts >= ? AND ts < ?
            GROUP BY bucket
            ORDER BY bucket
            """,
            (start, end),
        )
        timeseries_rows = [dict(r) for r in await cur.fetchall()]

        cur = await db.execute(
            """
            SELECT id, ts, model, dropped_messages, dropped_tokens,
                   summary_tokens, saved_per_turn
            FROM compaction_events
            WHERE ts >= ? AND ts < ?
            ORDER BY id DESC
            """,
            (start, end),
        )
        compaction_rows = [dict(r) for r in await cur.fetchall()]
        compaction_dropped = sum(r["dropped_tokens"] or 0 for r in compaction_rows)
        compaction_per_turn = sum(r["saved_per_turn"] or 0 for r in compaction_rows)

        cur = await db.execute(
            """
            SELECT id, ts, path, output_path, model,
                   before_tokens, after_tokens, saved_tokens
            FROM skill_compressions
            WHERE ts >= ? AND ts < ?
            ORDER BY id DESC
            """,
            (start, end),
        )
        skill_rows = [dict(r) for r in await cur.fetchall()]

    counterfactual_usd = (totals.get("spent_usd") or 0.0) + saved_usd_total
    actual_usd = totals.get("spent_usd") or 0.0
    savings_pct = (saved_usd_total / counterfactual_usd * 100.0) if counterfactual_usd > 0 else 0.0

    return {
        "window": {
            "from": start,
            "to": end,
            "span_seconds": end - start,
            "bucket_seconds": bucket_seconds,
            "generated_at": time.time(),
        },
        "totals": totals,
        "comparison": {
            "without_tokenq_usd": counterfactual_usd,
            "with_tokenq_usd": actual_usd,
            "saved_usd": saved_usd_total,
            "saved_pct": savings_pct,
        },
        "by_model": by_model_rows,
        "expensive": expensive_rows,
        "timeseries": timeseries_rows,
        "compactions": {
            "events": compaction_rows,
            "dropped_tokens": compaction_dropped,
            "saved_per_turn": compaction_per_turn,
        },
        "skill_compressions": skill_rows,
    }


async def report_json(request: Request) -> Response:
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    payload = await _collect_report(start, end)
    return JSONResponse(payload)


async def report_pdf(request: Request) -> Response:
    window = _parse_window(request)
    if isinstance(window, JSONResponse):
        return window
    start, end = window
    payload = await _collect_report(start, end)

    from .report import render_report_pdf

    pdf_bytes = render_report_pdf(payload)
    filename = f"tokenq-report-{int(start)}-{int(end)}.pdf"
    return Response(
        pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


routes = [
    Route("/", index),
    Route("/api/stats", stats),
    Route("/api/recent", recent),
    Route("/api/timeseries", timeseries),
    Route("/api/by_model", by_model),
    Route("/api/expensive", expensive),
    Route("/api/sessions", sessions),
    Route("/api/by_project", by_project),
    Route("/api/by_tool", by_tool),
    Route("/api/by_activity", by_activity),
    Route("/api/by_shell_command", by_shell_command),
    Route("/api/by_mcp", by_mcp),
    Route("/api/oneshot", oneshot),
    Route("/api/health", health),
    Route("/api/skill_compressions", skill_compressions),
    Route("/api/compactions", compactions),
    Route("/api/report", report_json),
    Route("/api/report.pdf", report_pdf),
    Mount("/static", app=StaticFiles(directory=STATIC_DIR), name="static"),
]


@contextlib.asynccontextmanager
async def lifespan(_: Starlette):
    from ..logging import configure as configure_logging
    from ..logging import get_logger

    configure_logging()
    log = get_logger("dashboard.app")
    await init_db()
    log.info("dashboard_started")
    try:
        yield
    finally:
        log.info("dashboard_stopped")


app = Starlette(routes=routes, lifespan=lifespan)
