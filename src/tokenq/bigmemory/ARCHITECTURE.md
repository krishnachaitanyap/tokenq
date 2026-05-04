# bigmemory — Architecture & Cost-Saving Strategies

`bigmemory` is tokenq's local long-term memory layer. It captures large
tool-result payloads passing through the proxy into a local SQLite + FTS5
store, then exposes that store to Claude clients as an MCP server so the
model can recall prior content on demand instead of re-running expensive
tool calls or re-reading large files.

---

## 1. Module layout

```
src/tokenq/bigmemory/
├── __init__.py      Public surface: BigMemoryStore, MemoryItem
├── store.py         SQLite + FTS5 store (async, aiosqlite)
├── pipeline.py      BigMemoryStage — capture-only proxy stage
└── mcp.py           JSON-RPC 2.0 MCP server (Starlette HTTP)
```

Two surfaces, one store:

- **Capture** (`pipeline.BigMemoryStage`) — runs inside the tokenq proxy
  pipeline. Walks every `tool_result` block in an outgoing request and
  writes any whose content exceeds `BIGMEMORY_CAPTURE_MIN_TOKENS` (default
  500) into the store. **Does not mutate the request body** — v1 is
  data-collection only.
- **Recall** (`mcp.create_app`) — exposes the store as an MCP server over
  HTTP/JSON-RPC 2.0. Claude clients call `memory_search`, `memory_save`,
  `memory_recent`, `memory_forget`, `memory_stats`.

---

## 2. Storage schema (`store.py`)

Two tables plus FTS triggers, all in one SQLite file at `DB_PATH`
(`~/tokenq.db` by default), opened in WAL mode.

### `memory_items` — durable rows

| column        | type    | notes                                            |
|---------------|---------|--------------------------------------------------|
| `id`          | INTEGER | PK, autoincrement                                |
| `ts`          | REAL    | unix timestamp at capture                        |
| `kind`        | TEXT    | `tool_result` \| `turn_summary` \| `fact` \| `note` |
| `source`      | TEXT    | tool name + first input arg, or file path        |
| `content`     | TEXT    | raw text                                         |
| `hash`        | TEXT    | sha256 of content — **UNIQUE** (dedup key)       |
| `tokens`      | INTEGER | rough estimate (`len // 4`)                      |
| `hits`        | INTEGER | search-recall counter                            |
| `last_hit_ts` | REAL    | last recall time                                 |

Indexes on `ts DESC` and `kind` for the recent/stats paths.

### `memory_items_fts` — FTS5 virtual table

Mirrors `(content, source, kind)` as a contentless FTS5 table
(`content=memory_items, content_rowid=id`), tokenized with
`unicode61 remove_diacritics 2`. Three triggers (`ai`/`ad`/`au`) keep it
in sync on insert/delete/update — no manual reindexing needed.

### Why this design

- **SQLite + FTS5**: stdlib only on macOS/Linux Python builds. No vector
  DB, no embedding model, no extra process. Search latency is sub-ms for
  hundreds of thousands of rows.
- **BM25 ranking**: good enough for recall over tool-result text where
  the user's query usually shares vocabulary with the stored content.
- **Hash-keyed dedup**: re-capturing the same `cat foo.py` output across
  N turns produces one row, not N. The `ON CONFLICT(hash) DO NOTHING`
  insert is a no-op on collision.

---

## 3. Capture path (`pipeline.py`)

`BigMemoryStage` is a tokenq pipeline stage. For each request body:

1. Walk `body["messages"]` looking for blocks with `type == "tool_result"`.
2. Flatten the block's `content` to a single string (handles both string
   and list-of-text-blocks shapes).
3. Estimate tokens; skip if below `min_tokens`.
4. Walk **backwards** through messages to find the matching `tool_use`
   block (by `tool_use_id`) and build a compact `source` label like
   `Read:/path/to/file.py` or `Bash:npm test`.
5. `store.add(content, kind="tool_result", source=...)` — dedup happens
   at the SQL layer.

Failures are caught and logged; capture **never breaks the proxy
request**. The stage records `req.metadata["bigmemory_captured"] = N`
when it captures anything, which the dashboard can surface.

**Key invariant**: the request body is unchanged. v2 may inject a recall
prefix, but only with a chunk-snapping strategy that preserves Anthropic
prompt-cache boundaries.

---

## 4. Recall path (`mcp.py`)

A Starlette app exposing two routes:

- `GET /healthz` — liveness
- `GET /mcp` — server info
- `POST /mcp` — JSON-RPC 2.0 endpoint

Implements the minimum MCP surface a Claude client uses:

| method                       | purpose                          |
|------------------------------|----------------------------------|
| `initialize`                 | handshake, returns capabilities  |
| `notifications/initialized`  | client ack (no response)         |
| `tools/list`                 | enumerate tools + JSON schemas   |
| `tools/call`                 | invoke a tool                    |

No SSE / streaming — single POST in, single JSON-RPC response out. Keeps
the spec footprint tiny while staying compatible with `mcp-remote`-style
HTTP clients.

### Tools

- **`memory_search(query, limit=10, kind?)`** — FTS5 BM25 search, returns
  top-N items as text blocks. Increments `hits` on returned rows.
- **`memory_save(content, kind="note", source?)`** — explicit write
  surface for facts/decisions the model wants to retain.
- **`memory_recent(limit=20, kind?)`** — most-recent rows.
- **`memory_forget(id)`** — hard delete (FTS row removed via trigger).
- **`memory_stats()`** — totals + per-kind counts/tokens/hits.

Each handler is small; dispatch is a dict lookup. Errors map to JSON-RPC
codes (`-32600..-32603`).

---

## 5. End-to-end flow

```
┌────────────┐  request   ┌────────────────┐  insert   ┌────────────────┐
│ Claude CLI │ ─────────▶ │ tokenq proxy + │ ────────▶ │ SQLite + FTS5  │
└────────────┘            │ BigMemoryStage │           │  (~/tokenq.db) │
      ▲                   └────────────────┘           └────────────────┘
      │                                                         ▲
      │ memory_search / save / recent (JSON-RPC over HTTP)      │
      └─────────────────────────────────────────────────────────┘
                          bigmemory MCP server
```

Capture is silent and side-effect-free for the request. Recall is
explicit — the model decides when to call `memory_search`.

---

## 6. Strategies to cut cost

The token bill on Claude requests is dominated by the **input** side:
every turn re-sends the full transcript, including any large tool-result
blocks that were ever read. bigmemory's job is to keep those blocks out
of the prompt on subsequent turns.

### A. Capture-then-recall (the v1 lever)

A 50KB file Read once and referenced 10 turns later costs ~12k input
tokens **per turn** for those 10 turns. Pattern:

1. First read populates bigmemory (capture stage runs automatically).
2. On later turns the model calls `memory_search("the thing I need")`
   instead of re-running `Read`.
3. The MCP response carries only the matched snippet, not the whole file.

This works **today** with v1, but it's opt-in: the model has to choose
`memory_search` over `Read`. Encouraging it via system prompt or skill
instructions is the lowest-effort win.

### B. Tune `BIGMEMORY_CAPTURE_MIN_TOKENS`

Default is 500. Lower it (e.g. 200) to capture more, raise it (e.g.
1500) to keep the store lean and search noise-free. Trade-off: smaller
threshold → more rows → slower BM25 ranking and more dedup churn.

### C. Active prefix injection (v2, deferred)

The high-leverage move: in the capture stage, **replace** large
tool-result blocks with a short pointer (`<memory id=42 tokens=12000/>`)
once they've been stored, and let the model retrieve on demand. Two
hard problems gate this:

- **Prompt-cache stability** — Anthropic's cache key is a prefix hash.
  Mutating any block in the prefix invalidates the cache for every
  request that shares it. Mitigation: snap mutations to existing cache
  block boundaries, and only rewrite blocks **older** than the last
  cache breakpoint.
- **Tool-result schema** — replacing the body changes what the model
  "sees" as the tool's output. Needs a paired system-prompt instruction
  telling it those pointers are dereferenceable via `memory_search`.

Expected savings: 50–90% of input tokens on long tool-heavy sessions.

### D. Dedup is already a saving

The `hash` UNIQUE constraint means re-capturing identical content costs
zero extra storage and zero extra rows. Sessions that re-read the same
files repeatedly already get this for free.

### E. Hit-counted retention / pruning

`hits` and `last_hit_ts` are populated but **not yet used** for
eviction. Useful policies to add:

- Drop rows where `kind = "tool_result"` AND `hits = 0` AND
  `ts < now - 30d`.
- Cap total tokens; evict by `(hits ASC, ts ASC)`.

This bounds disk and keeps FTS ranking from drifting toward stale
content.

### F. Coarse summarization for recurring large outputs

For tool results that are predictably huge and structured (e.g. full
`git log`, `ls -R`), a summarization stage could store a compact
representation as `kind="turn_summary"` alongside the raw row, and have
`memory_search` prefer summaries when both match. Saves recall tokens,
not capture tokens.

### G. Per-kind search bias

`memory_search` already accepts a `kind` filter. Bias the model's
default queries toward `fact` and `note` (small, high-signal) before
falling back to `tool_result` (large). Reduces the average payload of a
recall hit.

### H. Batch the FTS hit-counter writes

Each search currently issues a synchronous `UPDATE ... SET hits = hits + 1`.
On a hot recall path this is the dominant write. Batching to a queue
flushed every N seconds (or merging into a single statement per search,
which it already does for the result set) keeps SQLite's WAL small.

### I. Right-size the recall payload

`memory_search` returns full `content` per hit. For very large items,
returning a span around the BM25-matched terms (snippet window, ~500
tokens) instead of the whole row would cut recall cost ~10x on file-sized
rows. FTS5's `snippet()` and `highlight()` functions do this natively.

---

## 7. Operational notes

- **DB path**: `~/tokenq.db` (configurable via `DB_PATH` in
  `tokenq/config.py`).
- **Concurrency**: aiosqlite opens a fresh connection per call. WAL
  mode allows concurrent readers + one writer. Single-process is fine;
  multi-process needs care around the FTS triggers (still safe, just
  serialized).
- **Backups**: a single file. `cp ~/tokenq.db ~/tokenq.db.bak` works
  while the proxy runs (WAL).
- **Resetting**: delete the file, or use the `tokenq reset` command via
  the dashboard.

---

## 8. Roadmap signals

- [ ] Active prefix injection (cost lever C)
- [ ] Retention/eviction policy (cost lever E)
- [ ] FTS `snippet()` in `memory_search` results (cost lever I)
- [ ] Optional embedding-backed semantic search alongside BM25
- [ ] Per-session namespaces (so memories from project A don't surface
      in project B)
