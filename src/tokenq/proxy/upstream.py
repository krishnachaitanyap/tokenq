"""Shared httpx client for upstream calls to api.anthropic.com.

A single AsyncClient is reused across requests so we keep a connection pool.
HTTP/2 is on so we get multiplexing for streaming responses.
"""
from __future__ import annotations

import httpx

from ..config import ANTHROPIC_API_BASE

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=ANTHROPIC_API_BASE,
            http2=True,
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=30.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _client


def set_client(client: httpx.AsyncClient | None) -> None:
    """Inject a client (used by tests with MockTransport)."""
    global _client
    _client = client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def reset_client() -> None:
    """Drop the shared client so the next get_client() builds a fresh one.

    Use after a transport-level failure (TCP reset, DNS hiccup) where the
    connection pool may be holding broken sockets.
    """
    await close_client()
