"""bigmemory — local long-term memory layer for tokenq.

Two surfaces:
  - pipeline.BigMemoryStage captures large tool_results from intercepted
    requests into a local SQLite + FTS5 store.
  - mcp.create_app exposes the same store as a JSON-RPC 2.0 MCP server over
    HTTP so a Claude client can search/save/recall memories directly.

The store is implemented against stdlib sqlite3 only (FTS5 ships with the
Python build on macOS/Linux). No third-party libraries.
"""
from .store import BigMemoryStore, MemoryItem

__all__ = ["BigMemoryStore", "MemoryItem"]
