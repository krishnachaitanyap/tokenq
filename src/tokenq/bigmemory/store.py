"""SQLite + FTS5 backed memory store.

Schema:
  memory_items       — durable rows (id, ts, kind, source, content, hash, tokens,
                       scope, confidence, strength, topic_key, superseded_by)
  memory_items_fts   — FTS5 virtual table mirroring `content`, ranked with bm25

Dedup is by sha256 of normalized content, so re-capturing the same tool_result
across many turns produces one row, not N — and the existing row's `strength`
is incremented (a soft "seen N times" counter that protects it from pruning).
FTS5 stays in sync via triggers.

Forgetting / TTL: `expire()` walks every row, computes a per-kind exponential
decay (`confidence × exp(-age_days / half_life_for_kind)`), and deletes rows
whose decayed confidence falls below 0.05 *and* whose strength is ≤ 1.
Half-lives intentionally vary by kind — corrections persist for a year, raw
tool_results for days. The pruning rule is borrowed from jcode's MEMORY_ARCHITECTURE.

Supersession: when `add()` is called with a `topic_key`, all prior active
items with the same key are marked `superseded_by` the new row, so a refreshed
fact ("user role: data scientist" → "user role: ML engineer") naturally
retires the stale version. Default `search()`/`recent()` filter superseded
rows out; pass `include_superseded=True` to see them for audit.

Profile (stable identity): items written with `kind='profile'` and
`scope='global'` survive nearly forever (10000-day half-life) and are returned
by `profile()` for surfacing as a stable user model alongside the noisier
recent context. `set_profile(key, value)` is the convenience entrypoint —
it wraps add() with topic_key=key so subsequent updates supersede prior values.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import aiosqlite

from ..config import DB_PATH
from . import embedder

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    source TEXT,
    content TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE,
    tokens INTEGER DEFAULT 0,
    hits INTEGER DEFAULT 0,
    last_hit_ts REAL,
    scope TEXT NOT NULL DEFAULT 'session',
    confidence REAL NOT NULL DEFAULT 1.0,
    strength INTEGER NOT NULL DEFAULT 1,
    topic_key TEXT,
    superseded_by INTEGER REFERENCES memory_items(id) ON DELETE SET NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_memory_items_ts ON memory_items(ts DESC);
CREATE INDEX IF NOT EXISTS idx_memory_items_kind ON memory_items(kind);
CREATE INDEX IF NOT EXISTS idx_memory_items_topic ON memory_items(topic_key);
CREATE INDEX IF NOT EXISTS idx_memory_items_scope ON memory_items(scope);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts USING fts5(
    content, source, kind,
    content=memory_items, content_rowid=id,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS memory_items_ai AFTER INSERT ON memory_items BEGIN
    INSERT INTO memory_items_fts(rowid, content, source, kind)
    VALUES (new.id, new.content, new.source, new.kind);
END;

CREATE TRIGGER IF NOT EXISTS memory_items_ad AFTER DELETE ON memory_items BEGIN
    INSERT INTO memory_items_fts(memory_items_fts, rowid, content, source, kind)
    VALUES ('delete', old.id, old.content, old.source, old.kind);
END;

CREATE TRIGGER IF NOT EXISTS memory_items_au AFTER UPDATE OF content ON memory_items BEGIN
    INSERT INTO memory_items_fts(memory_items_fts, rowid, content, source, kind)
    VALUES ('delete', old.id, old.content, old.source, old.kind);
    INSERT INTO memory_items_fts(rowid, content, source, kind)
    VALUES (new.id, new.content, new.source, new.kind);
END;
"""

# Per-kind half-life in days. Numbers ported from jcode/docs/MEMORY_ARCHITECTURE.md.
# `profile` is effectively immortal — stable identity facts shouldn't decay.
HALF_LIFE_DAYS: dict[str, float] = {
    "correction": 365.0,
    "preference": 90.0,
    "procedure": 60.0,
    "fact": 30.0,
    "note": 14.0,
    "turn_summary": 7.0,
    "inferred": 7.0,
    "tool_result": 3.0,
    "profile": 10_000.0,
}
DEFAULT_HALF_LIFE_DAYS = 14.0
PRUNE_CONFIDENCE = 0.05
PRUNE_MAX_STRENGTH = 1


@dataclass
class MemoryItem:
    id: int | None
    ts: float
    kind: str            # tool_result | turn_summary | fact | note | preference | correction | procedure | inferred | profile
    source: str | None   # file path / tool name / session id
    content: str
    hash: str
    tokens: int
    hits: int = 0
    last_hit_ts: float | None = None
    scope: str = "session"           # global | project | session
    confidence: float = 1.0
    strength: int = 1
    topic_key: str | None = None
    superseded_by: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _row_to_item(row: aiosqlite.Row) -> MemoryItem:
    """Build a MemoryItem from a row, stripping the embedding BLOB (the
    dataclass doesn't carry it — pulling 1.5KB through every read would
    bloat list operations)."""
    return MemoryItem(**{k: row[k] for k in row.keys() if k not in ("embedding", "score")})


def _estimate_tokens(content: str) -> int:
    return max(1, len(content) // 4)


def _escape_fts(query: str) -> str:
    """Quote each whitespace-separated token so FTS5 syntax (-, ., :, AND) in
    user input doesn't blow up the parser. Empty input returns ''."""
    parts = [p for p in query.split() if p]
    return " ".join(f'"{p.replace(chr(34), chr(34) * 2)}"' for p in parts)


def _decayed_confidence(
    *, kind: str, ts: float, confidence: float, hits: int, now: float
) -> float:
    """Per-kind exponential decay with a small log-scaled hit bonus.

    confidence × exp(-age_days/half_life) × (1 + 0.1 × log1p(hits))
    """
    half_life = HALF_LIFE_DAYS.get(kind, DEFAULT_HALF_LIFE_DAYS)
    age_days = max(0.0, (now - ts) / 86400.0)
    base = confidence * math.exp(-age_days / half_life)
    hit_boost = 1.0 + 0.1 * math.log1p(max(0, hits))
    return base * hit_boost


# Columns added after v0. SQLite has no IF NOT EXISTS for ADD COLUMN, so we
# introspect PRAGMA table_info and add what's missing. Safe to call repeatedly.
_ADDED_COLUMNS: list[tuple[str, str]] = [
    ("scope", "TEXT NOT NULL DEFAULT 'session'"),
    ("confidence", "REAL NOT NULL DEFAULT 1.0"),
    ("strength", "INTEGER NOT NULL DEFAULT 1"),
    ("topic_key", "TEXT"),
    ("superseded_by", "INTEGER REFERENCES memory_items(id) ON DELETE SET NULL"),
    ("embedding", "BLOB"),
]


class BigMemoryStore:
    """Async wrapper over the FTS5 store. One instance per process is fine —
    aiosqlite opens a fresh connection per call."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = str(db_path) if db_path else str(DB_PATH)

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA foreign_keys=ON")
            await db.executescript(SCHEMA)
            # Migrate older databases: add columns introduced after v0.
            cur = await db.execute("PRAGMA table_info(memory_items)")
            existing = {row[1] for row in await cur.fetchall()}
            for col, decl in _ADDED_COLUMNS:
                if col not in existing:
                    await db.execute(f"ALTER TABLE memory_items ADD COLUMN {col} {decl}")
            await db.commit()

    async def add(
        self,
        *,
        content: str,
        kind: str,
        source: str | None = None,
        ts: float | None = None,
        scope: str = "session",
        confidence: float = 1.0,
        topic_key: str | None = None,
    ) -> MemoryItem:
        """Insert or merge a memory item.

        - If `content` matches an existing row by hash, increment its strength
          (the duplicate is signal that the fact is real) and return that row.
        - If `topic_key` is set, mark every prior *active* row sharing that
          key as superseded by the new row.
        """
        h = _hash(content)
        toks = _estimate_tokens(content)
        ts = ts if ts is not None else time.time()
        # Compute embedding outside the DB connection — fastembed is sync and
        # CPU-bound, ~80ms; we don't want it holding the SQLite connection.
        emb = embedder.embed(content)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            existing = await (await db.execute(
                "SELECT id, embedding FROM memory_items WHERE hash = ?", (h,)
            )).fetchone()
            if existing is not None:
                # Bump strength + opportunistically backfill embedding if missing.
                if existing["embedding"] is None and emb is not None:
                    await db.execute(
                        "UPDATE memory_items SET strength = strength + 1, embedding = ? WHERE id = ?",
                        (emb, existing["id"]),
                    )
                else:
                    await db.execute(
                        "UPDATE memory_items SET strength = strength + 1 WHERE id = ?",
                        (existing["id"],),
                    )
                await db.commit()
                row = await (await db.execute(
                    "SELECT * FROM memory_items WHERE id = ?", (existing["id"],)
                )).fetchone()
                return _row_to_item(row)

            cur = await db.execute(
                """
                INSERT INTO memory_items
                  (ts, kind, source, content, hash, tokens,
                   scope, confidence, strength, topic_key, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (ts, kind, source, content, h, toks, scope, confidence, topic_key, emb),
            )
            new_id = cur.lastrowid
            if topic_key:
                await db.execute(
                    """
                    UPDATE memory_items
                       SET superseded_by = ?
                     WHERE topic_key = ?
                       AND id != ?
                       AND superseded_by IS NULL
                    """,
                    (new_id, topic_key, new_id),
                )
            await db.commit()
            row = await (await db.execute(
                "SELECT * FROM memory_items WHERE id = ?", (new_id,)
            )).fetchone()
        return _row_to_item(row)

    async def set_profile(
        self,
        *,
        key: str,
        value: str,
        source: str | None = None,
        confidence: float = 1.0,
    ) -> MemoryItem:
        """Stable identity fact — supersedes any prior value for the same key."""
        return await self.add(
            content=value,
            kind="profile",
            source=source,
            scope="global",
            confidence=confidence,
            topic_key=key,
        )

    async def profile(self, *, limit: int = 50) -> list[MemoryItem]:
        """Return the active stable profile (kind='profile' or scope='global'),
        ordered by topic_key for stable iteration."""
        sql = """
            SELECT * FROM memory_items
             WHERE superseded_by IS NULL
               AND (kind = 'profile' OR scope = 'global')
             ORDER BY COALESCE(topic_key, ''), ts DESC
             LIMIT ?
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (await db.execute(sql, (limit,))).fetchall()
        return [_row_to_item(r) for r in rows]

    async def get(self, item_id: int) -> MemoryItem | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            row = await (await db.execute(
                "SELECT * FROM memory_items WHERE id = ?", (item_id,)
            )).fetchone()
        return _row_to_item(row) if row else None

    async def delete(self, item_id: int) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "DELETE FROM memory_items WHERE id = ?", (item_id,)
            )
            await db.commit()
            return cur.rowcount > 0

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        kind: str | None = None,
        include_superseded: bool = False,
    ) -> list[MemoryItem]:
        """FTS5 BM25 ranked search. Empty query returns []."""
        fts = _escape_fts(query)
        if not fts:
            return []
        sql = """
            SELECT m.*, bm25(memory_items_fts) AS score
            FROM memory_items_fts
            JOIN memory_items m ON m.id = memory_items_fts.rowid
            WHERE memory_items_fts MATCH ?
        """
        params: list[Any] = [fts]
        if kind:
            sql += " AND m.kind = ?"
            params.append(kind)
        if not include_superseded:
            sql += " AND m.superseded_by IS NULL"
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (await db.execute(sql, params)).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                qmarks = ",".join("?" * len(ids))
                await db.execute(
                    f"UPDATE memory_items SET hits = hits + 1, last_hit_ts = ? "
                    f"WHERE id IN ({qmarks})",
                    [time.time(), *ids],
                )
                await db.commit()
        return [_row_to_item(r) for r in rows]

    async def recent(
        self,
        *,
        limit: int = 20,
        kind: str | None = None,
        include_superseded: bool = False,
    ) -> list[MemoryItem]:
        sql = "SELECT * FROM memory_items WHERE 1=1"
        params: list[Any] = []
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if not include_superseded:
            sql += " AND superseded_by IS NULL"
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (await db.execute(sql, params)).fetchall()
        return [_row_to_item(r) for r in rows]

    async def expire(self, *, now: float | None = None) -> int:
        """Sweep: delete items whose decayed confidence < 0.05 AND strength ≤ 1.

        SQLite has no `exp()` without a math extension, so we score in Python.
        Cost is one SELECT + one bulk DELETE — cheap until the store is huge,
        and even then the dominant cost is iterating the rows once.
        """
        now = now if now is not None else time.time()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (await db.execute(
                "SELECT id, kind, ts, confidence, strength, hits "
                "FROM memory_items WHERE superseded_by IS NULL"
            )).fetchall()
            to_delete: list[int] = []
            for r in rows:
                if r["strength"] > PRUNE_MAX_STRENGTH:
                    continue
                score = _decayed_confidence(
                    kind=r["kind"],
                    ts=r["ts"],
                    confidence=r["confidence"],
                    hits=r["hits"],
                    now=now,
                )
                if score < PRUNE_CONFIDENCE:
                    to_delete.append(r["id"])
            # Also sweep already-superseded rows older than 30 days — they've
            # served their audit purpose.
            cutoff = now - 30 * 86400.0
            stale = await (await db.execute(
                "SELECT id FROM memory_items "
                "WHERE superseded_by IS NOT NULL AND ts < ?",
                (cutoff,),
            )).fetchall()
            to_delete.extend(r["id"] for r in stale)
            if to_delete:
                qmarks = ",".join("?" * len(to_delete))
                await db.execute(
                    f"DELETE FROM memory_items WHERE id IN ({qmarks})", to_delete
                )
                await db.commit()
        return len(to_delete)

    async def semantic_search(
        self,
        query: str,
        *,
        limit: int = 10,
        kind: str | None = None,
        include_superseded: bool = False,
        candidate_pool: int = 500,
    ) -> list[tuple[MemoryItem, float]]:
        """Cosine-similarity search over stored embeddings. Returns
        (item, similarity) pairs sorted descending. Empty result if the
        embedder isn't available.

        We score in Python over a candidate pool (default 500 most recent
        active items). Brute force is fine up to ~50k rows; an ANN index can
        slot in later if the store grows past that. Recency bias of the pool
        is intentional — older noise is unlikely to be relevant and pruning
        the pool keeps cosine work bounded.
        """
        if not query or not query.strip():
            return []
        q_emb = embedder.embed(query)
        if q_emb is None:
            return []

        sql = "SELECT * FROM memory_items WHERE embedding IS NOT NULL"
        params: list[Any] = []
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if not include_superseded:
            sql += " AND superseded_by IS NULL"
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(candidate_pool)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (await db.execute(sql, params)).fetchall()
            scored: list[tuple[MemoryItem, float]] = []
            for r in rows:
                blob = r["embedding"]
                if not blob:
                    continue
                sim = embedder.cosine_bytes(q_emb, blob)
                if sim <= 0.0:
                    continue
                scored.append((_row_to_item(r), sim))
            scored.sort(key=lambda p: p[1], reverse=True)
            top = scored[:limit]
            ids = [it.id for it, _ in top]
            if ids:
                qmarks = ",".join("?" * len(ids))
                await db.execute(
                    f"UPDATE memory_items SET hits = hits + 1, last_hit_ts = ? "
                    f"WHERE id IN ({qmarks})",
                    [time.time(), *ids],
                )
                await db.commit()
        return top

    async def hybrid_search(
        self,
        query: str,
        *,
        limit: int = 10,
        kind: str | None = None,
        include_superseded: bool = False,
        rrf_k: int = 60,
    ) -> list[MemoryItem]:
        """Reciprocal Rank Fusion over BM25 (lexical) and cosine (semantic).

        RRF formula: `score(d) = Σ 1 / (k + rank_in_list(d))` summed over
        each ranked list the doc appears in. k=60 is the canonical default
        from Cormack et al. (2009) — robust across query types and avoids
        the score-normalization pitfalls of weighted-sum fusion.

        Falls back to lexical-only when the embedder is unavailable, so
        callers can switch to hybrid blindly without runtime checks.
        """
        if not embedder.available():
            return await self.search(
                query, limit=limit, kind=kind, include_superseded=include_superseded
            )

        # Pull a wider net from each retriever than `limit` so RRF has signal.
        wide = max(limit * 4, 20)
        lex = await self.search(
            query, limit=wide, kind=kind, include_superseded=include_superseded
        )
        sem = await self.semantic_search(
            query, limit=wide, kind=kind, include_superseded=include_superseded
        )

        scores: dict[int, float] = {}
        items: dict[int, MemoryItem] = {}
        for rank, it in enumerate(lex):
            scores[it.id] = scores.get(it.id, 0.0) + 1.0 / (rrf_k + rank + 1)
            items[it.id] = it
        for rank, (it, _sim) in enumerate(sem):
            scores[it.id] = scores.get(it.id, 0.0) + 1.0 / (rrf_k + rank + 1)
            items.setdefault(it.id, it)

        # Decayed-confidence boost — a fresh, frequently-hit fact wins
        # ties over a stale, never-touched one.
        now = time.time()
        for item_id, item in items.items():
            decay = _decayed_confidence(
                kind=item.kind, ts=item.ts, confidence=item.confidence,
                hits=item.hits, now=now,
            )
            scores[item_id] *= 0.5 + 0.5 * decay  # bound boost in [0.5, 1.0+]

        ordered = sorted(scores.items(), key=lambda p: p[1], reverse=True)[:limit]
        return [items[i] for i, _ in ordered]

    async def backfill_embeddings(self, *, batch: int = 64, max_rows: int = 1024) -> int:
        """Fill `embedding` for rows that have NULL — useful after a fastembed
        install on an existing DB. Returns the count of rows backfilled. Bounded
        by `max_rows` so a one-shot call stays cheap; call repeatedly to drain."""
        if not embedder.available():
            return 0
        filled = 0
        while filled < max_rows:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                rows = await (await db.execute(
                    "SELECT id, content FROM memory_items "
                    "WHERE embedding IS NULL LIMIT ?",
                    (min(batch, max_rows - filled),),
                )).fetchall()
                if not rows:
                    return filled
                texts = [r["content"] for r in rows]
                vecs = embedder.embed_many(texts)
                for r, v in zip(rows, vecs):
                    if v is not None:
                        await db.execute(
                            "UPDATE memory_items SET embedding = ? WHERE id = ?",
                            (v, r["id"]),
                        )
                        filled += 1
                await db.commit()
        return filled

    async def stats(self) -> dict[str, Any]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            totals = await (await db.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(tokens), 0) AS toks, "
                "COALESCE(SUM(hits), 0) AS hits FROM memory_items"
            )).fetchone()
            active = await (await db.execute(
                "SELECT COUNT(*) AS n FROM memory_items WHERE superseded_by IS NULL"
            )).fetchone()
            by_kind = await (await db.execute(
                "SELECT kind, COUNT(*) AS n, COALESCE(SUM(tokens), 0) AS toks "
                "FROM memory_items WHERE superseded_by IS NULL "
                "GROUP BY kind ORDER BY n DESC"
            )).fetchall()
            embedded = await (await db.execute(
                "SELECT COUNT(*) AS n FROM memory_items WHERE embedding IS NOT NULL"
            )).fetchone()
        return {
            "total_items": totals["n"],
            "active_items": active["n"],
            "superseded_items": totals["n"] - active["n"],
            "total_tokens": totals["toks"],
            "total_hits": totals["hits"],
            "embedded_items": embedded["n"],
            "embedder_available": embedder.available(),
            "by_kind": [dict(r) for r in by_kind],
        }
