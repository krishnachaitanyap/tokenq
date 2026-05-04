import os
from pathlib import Path

ANTHROPIC_API_BASE = os.getenv("TOKENQ_UPSTREAM", "https://api.anthropic.com")

HOME = Path(os.getenv("TOKENQ_HOME", str(Path.home() / ".tokenq")))
HOME.mkdir(parents=True, exist_ok=True)

DB_PATH = HOME / "tokenq.db"
PID_PATH = HOME / "tokenq.pid"        # written by `tokenq start`, read by `stop`
DAEMON_LOG_PATH = HOME / "tokenq.log" # stdout/stderr of detached child

PROXY_HOST = os.getenv("TOKENQ_HOST", "127.0.0.1")
PROXY_PORT = int(os.getenv("TOKENQ_PORT", "8089"))
DASHBOARD_PORT = int(os.getenv("TOKENQ_DASHBOARD_PORT", "8090"))

LOG_LEVEL = os.getenv("TOKENQ_LOG_LEVEL", "info")

CACHE_TTL_SEC = int(os.getenv("TOKENQ_CACHE_TTL_SEC", "86400"))
DEDUP_MIN_CHARS = int(os.getenv("TOKENQ_DEDUP_MIN_CHARS", "300"))
COMPRESS_MAX_LINES = int(os.getenv("TOKENQ_COMPRESS_MAX_LINES", "80"))
COMPRESS_KEEP_LINES = int(os.getenv("TOKENQ_COMPRESS_KEEP_LINES", "25"))

# Sliding-window compaction: when transcript exceeds THRESHOLD tokens, drop
# oldest messages, keeping at least KEEP_RECENT tokens of recent context.
# CHUNK_MESSAGES is the granularity of cut-point movement — bigger chunks =
# fewer rollovers (each rollover costs one upstream cache_creation) but each
# rollover delays the next batch of savings.
COMPACT_THRESHOLD_TOKENS = int(os.getenv("TOKENQ_COMPACT_THRESHOLD_TOKENS", "80000"))
COMPACT_KEEP_RECENT_TOKENS = int(os.getenv("TOKENQ_COMPACT_KEEP_RECENT_TOKENS", "20000"))
COMPACT_CHUNK_MESSAGES = int(os.getenv("TOKENQ_COMPACT_CHUNK_MESSAGES", "20"))

# Smart skill loading: only keep the top-K most relevant skill descriptions in
# the system prompt; trim the rest. Disabled when fewer than MIN_LIST entries
# are present (no point trimming a tiny list).
SKILLS_TOP_K = int(os.getenv("TOKENQ_SKILLS_TOP_K", "3"))
SKILLS_MIN_LIST = int(os.getenv("TOKENQ_SKILLS_MIN_LIST", "4"))

# bigmemory: capture-only pipeline stage stores tool_results above this many
# estimated tokens into a local FTS5 store. The same store is exposed via the
# MCP HTTP server on MCP_PORT so a client can search/save/recall memories.
BIGMEMORY_CAPTURE_MIN_TOKENS = int(os.getenv("TOKENQ_BIGMEMORY_MIN_TOKENS", "500"))
MCP_PORT = int(os.getenv("TOKENQ_MCP_PORT", "8091"))
MCP_ENABLED = os.getenv("TOKENQ_MCP_ENABLED", "1") not in ("0", "false", "False", "")

# bigmemory active prefix injection (Phase 2). Off by default — turning this
# on changes the request body and MUST keep the memory block byte-identical
# within a session window or it destroys the existing prompt cache savings.
# Strategy: hash (system, first_user_msg) → session_id; cache a snapshot per
# session in the snapshots table; refresh every REFRESH_TURNS turns or after
# REFRESH_SECS, whichever comes first.
BIGMEMORY_INJECT_ENABLED = os.getenv("TOKENQ_BIGMEMORY_INJECT", "0") not in ("0", "false", "False", "")
BIGMEMORY_INJECT_BUDGET_TOKENS = int(os.getenv("TOKENQ_BIGMEMORY_INJECT_BUDGET", "2000"))
BIGMEMORY_INJECT_PROFILE_FRACTION = float(os.getenv("TOKENQ_BIGMEMORY_INJECT_PROFILE_FRAC", "0.3"))
BIGMEMORY_INJECT_REFRESH_TURNS = int(os.getenv("TOKENQ_BIGMEMORY_INJECT_REFRESH_TURNS", "8"))
BIGMEMORY_INJECT_REFRESH_SECS = int(os.getenv("TOKENQ_BIGMEMORY_INJECT_REFRESH_SECS", "1800"))
