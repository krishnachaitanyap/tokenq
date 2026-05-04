"""SQLite storage layer. One file in ~/.tokenq/tokenq.db.

Tables:
  requests       — one row per intercepted request (for dashboard + analytics)
  cache          — exact-match response cache (week 2)
  bandit_state   — serialized bandit parameters per arm (week 3)

Week 1 only writes `requests`. The other tables are created upfront so later
modules don't need migrations.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    model TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cached_locally INTEGER DEFAULT 0,
    saved_tokens INTEGER DEFAULT 0,
    saved_by_cache INTEGER DEFAULT 0,
    saved_by_dedup INTEGER DEFAULT 0,
    saved_by_compress INTEGER DEFAULT 0,
    saved_by_skills INTEGER DEFAULT 0,
    saved_by_bandit_usd REAL DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    status_code INTEGER,
    error TEXT,
    estimated_cost_usd REAL DEFAULT 0,
    stream INTEGER DEFAULT 0,
    session_id TEXT,
    project TEXT,
    tools_used TEXT,           -- JSON-encoded list of tool names invoked
    turn_index INTEGER DEFAULT 0,
    activity TEXT,             -- codeburn-style activity classification
    edit_files TEXT,           -- JSON-encoded list of files Edited/Written this turn
    bash_verbs TEXT            -- JSON-encoded list of leading-verb shell commands
);

CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts DESC);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
-- Indexes on newly-added columns (session_id, project, activity) live in the
-- migration block below — running them here would fail on databases created
-- before those columns existed because executescript runs before ALTER TABLE.

CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    response BLOB NOT NULL,
    model TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    last_hit_at REAL,
    hit_count INTEGER DEFAULT 0,
    is_stream INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cache_created ON cache(created_at);

CREATE TABLE IF NOT EXISTS bandit_state (
    arm TEXT PRIMARY KEY,
    params BLOB NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_compressions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    path TEXT NOT NULL,
    output_path TEXT,
    model TEXT,
    before_tokens INTEGER DEFAULT 0,
    after_tokens INTEGER DEFAULT 0,
    saved_tokens INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_skill_compressions_ts ON skill_compressions(ts DESC);

CREATE TABLE IF NOT EXISTS compaction_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    model TEXT,
    dropped_messages INTEGER DEFAULT 0,
    dropped_tokens INTEGER DEFAULT 0,
    summary_tokens INTEGER DEFAULT 0,
    saved_per_turn INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_compaction_events_ts ON compaction_events(ts DESC);

CREATE TABLE IF NOT EXISTS bandit_shadow_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    bucket TEXT NOT NULL,
    original_arm TEXT NOT NULL,
    recommended_arm TEXT NOT NULL,
    est_success REAL DEFAULT 0,
    est_cost_full_usd REAL DEFAULT 0,
    est_cost_routed_usd REAL DEFAULT 0,
    est_cost_saved_usd REAL DEFAULT 0,
    stop_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_bandit_shadow_ts ON bandit_shadow_decisions(ts DESC);
"""


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.executescript(SCHEMA)
        # Backward-compat ALTERs for older DBs. Each is a no-op on fresh DBs
        # (column already exists in SCHEMA above), so swallow the duplicate-column
        # error.
        for stmt in (
            "ALTER TABLE cache ADD COLUMN is_stream INTEGER DEFAULT 0",
            "ALTER TABLE requests ADD COLUMN saved_by_cache INTEGER DEFAULT 0",
            "ALTER TABLE requests ADD COLUMN saved_by_dedup INTEGER DEFAULT 0",
            "ALTER TABLE requests ADD COLUMN saved_by_compress INTEGER DEFAULT 0",
            "ALTER TABLE requests ADD COLUMN saved_by_skills INTEGER DEFAULT 0",
            "ALTER TABLE requests ADD COLUMN saved_by_bandit_usd REAL DEFAULT 0",
            "ALTER TABLE requests ADD COLUMN session_id TEXT",
            "ALTER TABLE requests ADD COLUMN project TEXT",
            "ALTER TABLE requests ADD COLUMN tools_used TEXT",
            "ALTER TABLE requests ADD COLUMN turn_index INTEGER DEFAULT 0",
            "ALTER TABLE requests ADD COLUMN activity TEXT",
            "ALTER TABLE requests ADD COLUMN edit_files TEXT",
            "ALTER TABLE requests ADD COLUMN bash_verbs TEXT",
            "CREATE INDEX IF NOT EXISTS idx_requests_session ON requests(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_requests_project ON requests(project)",
            "CREATE INDEX IF NOT EXISTS idx_requests_activity ON requests(activity)",
        ):
            try:
                await db.execute(stmt)
            except aiosqlite.OperationalError:
                pass
        await db.commit()


@asynccontextmanager
async def db_conn():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def log_request(**fields: Any) -> None:
    fields.setdefault("ts", time.time())
    cols = ", ".join(fields.keys())
    placeholders = ", ".join("?" * len(fields))
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"INSERT INTO requests ({cols}) VALUES ({placeholders})",
            list(fields.values()),
        )
        await db.commit()


def log_skill_compression_sync(
    *,
    path: str,
    output_path: str | None,
    model: str,
    before_tokens: int,
    after_tokens: int,
    saved_tokens: int,
    ts: float | None = None,
) -> None:
    """Sync logger for the offline `tokenq compress-skill` CLI.

    Uses stdlib sqlite3 (not aiosqlite) since the CLI runs synchronously.
    Creates the table on demand so it works even if the proxy/dashboard never
    booted in this DB.
    """
    import sqlite3

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS skill_compressions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                path TEXT NOT NULL,
                output_path TEXT,
                model TEXT,
                before_tokens INTEGER DEFAULT 0,
                after_tokens INTEGER DEFAULT 0,
                saved_tokens INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_skill_compressions_ts
                ON skill_compressions(ts DESC);
            """
        )
        con.execute(
            """
            INSERT INTO skill_compressions
                (ts, path, output_path, model, before_tokens, after_tokens, saved_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts if ts is not None else time.time(),
                path,
                output_path,
                model,
                before_tokens,
                after_tokens,
                saved_tokens,
            ),
        )
        con.commit()
    finally:
        con.close()
