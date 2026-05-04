"""Single-process variant of `tokenq start`.

Runs proxy + dashboard + (optional) bigmemory MCP server inside ONE event
loop using `asyncio.gather` over `uvicorn.Server.serve()`. No subprocess
spawning, one PID, one log stream — easier to manage in containers, on
corporate-locked Windows where AppLocker may block child Python spawns,
and anywhere `multiprocessing.spawn` is restricted.

Usage:
    python -m tokenq.serve
    python -m tokenq.serve --no-mcp
    python -m tokenq.serve --port 8089 --dashboard-port 8090 --mcp-port 8091

Conceptually equivalent to running these three in parallel:
    python -m uvicorn tokenq.proxy.app:app      --port 8089
    python -m uvicorn tokenq.dashboard.app:app  --port 8090
    python -m uvicorn tokenq.bigmemory.mcp:app  --port 8091

…but in a single process so there's nothing to detach, nothing to
coordinate across PIDs, and the lifespan shutdown of all three runs in
the same loop.
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys

import uvicorn

from .config import (
    DASHBOARD_PORT,
    LOG_LEVEL,
    MCP_ENABLED,
    MCP_PORT,
    PROXY_HOST,
    PROXY_PORT,
)


async def serve_all(
    *,
    host: str,
    proxy_port: int,
    dashboard_port: int,
    mcp_port: int,
    mcp_on: bool,
    log_level: str,
) -> None:
    """Spin up every uvicorn server concurrently. Returns when all stop."""
    # Pre-initialize the SQLite DB ONCE before any uvicorn lifespan runs.
    # In the multi-process variant (`tokenq start`) each subprocess opens its
    # own connection serially, but here three lifespans race for the same
    # `PRAGMA journal_mode=WAL` and the losers crash with "database is locked".
    # Doing it up front avoids the race entirely; downstream init_db calls
    # are idempotent.
    from .storage import init_db
    await init_db()

    configs = [
        uvicorn.Config(
            "tokenq.proxy.app:app",
            host=host,
            port=proxy_port,
            log_level=log_level,
        ),
        uvicorn.Config(
            "tokenq.dashboard.app:app",
            host=host,
            port=dashboard_port,
            log_level="warning",
        ),
    ]
    if mcp_on:
        configs.append(
            uvicorn.Config(
                "tokenq.bigmemory.mcp:app",
                host=host,
                port=mcp_port,
                log_level="warning",
            )
        )
    servers = [uvicorn.Server(cfg) for cfg in configs]

    # Bridge SIGINT/SIGTERM to every server's `should_exit` flag so a single
    # Ctrl-C stops all three cleanly. On Windows `add_signal_handler` raises
    # NotImplementedError; uvicorn installs its own KeyboardInterrupt path
    # there, so the unix-only branch is best-effort.
    loop = asyncio.get_running_loop()

    def request_shutdown() -> None:
        for srv in servers:
            srv.should_exit = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_shutdown)
        except (NotImplementedError, RuntimeError):
            pass

    await asyncio.gather(*(srv.serve() for srv in servers))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tokenq.serve",
        description="Run tokenq proxy + dashboard + MCP in a single process.",
    )
    parser.add_argument("--host", default=PROXY_HOST, help="Bind address")
    parser.add_argument("--port", type=int, default=PROXY_PORT, help="Proxy port")
    parser.add_argument(
        "--dashboard-port", type=int, default=DASHBOARD_PORT, help="Dashboard port"
    )
    parser.add_argument(
        "--mcp-port", type=int, default=MCP_PORT, help="bigmemory MCP port"
    )
    parser.add_argument(
        "--no-mcp", action="store_true", help="Skip the bigmemory MCP server"
    )
    parser.add_argument(
        "--log-level", default=LOG_LEVEL, help="info | warning | debug"
    )
    args = parser.parse_args(argv)

    mcp_on = MCP_ENABLED and not args.no_mcp

    print()
    print(f"  tokenq proxy     →  http://{args.host}:{args.port}")
    print(f"  tokenq dashboard →  http://{args.host}:{args.dashboard_port}")
    if mcp_on:
        print(f"  bigmemory MCP    →  http://{args.host}:{args.mcp_port}/mcp")
    print()
    print("  Point Claude Code at it:")
    print(f"    export ANTHROPIC_BASE_URL=http://{args.host}:{args.port}")
    print()

    try:
        asyncio.run(
            serve_all(
                host=args.host,
                proxy_port=args.port,
                dashboard_port=args.dashboard_port,
                mcp_port=args.mcp_port,
                mcp_on=mcp_on,
                log_level=args.log_level,
            )
        )
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
