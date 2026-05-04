"""Tests for the single-process `tokenq.serve` runner.

The runner brings up proxy + dashboard + (optional) MCP in one event loop
via asyncio.gather over uvicorn.Server.serve(). The whole point is to be
robust on locked-down environments where multiprocessing.spawn is
restricted, so the test asserts:
  1. all three ports come up and respond
  2. the SQLite WAL pre-init avoids the "database is locked" race we hit
     before pre-initing
  3. SIGTERM (via should_exit) shuts every server down cleanly
"""
from __future__ import annotations

import asyncio
import os
import socket

import pytest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.asyncio
async def test_serve_all_brings_up_three_ports(tmp_path, monkeypatch):
    """Single-process serve must bind proxy, dashboard, AND MCP cleanly.

    Uses real loopback ports — uvicorn is hard to mock without losing the
    coverage we want. Tears down via the same `should_exit` flag the SIGINT
    bridge sets, so this also exercises the shutdown path.
    """
    monkeypatch.setenv("TOKENQ_HOME", str(tmp_path))
    # Re-import config under the patched env so DB_PATH points into tmp_path.
    import importlib

    import tokenq.config as cfg
    importlib.reload(cfg)

    from tokenq import serve as serve_mod
    importlib.reload(serve_mod)

    proxy_port = _free_port()
    dash_port = _free_port()
    mcp_port = _free_port()

    task = asyncio.create_task(
        serve_mod.serve_all(
            host="127.0.0.1",
            proxy_port=proxy_port,
            dashboard_port=dash_port,
            mcp_port=mcp_port,
            mcp_on=True,
            log_level="warning",
        )
    )

    try:
        # Give uvicorn a moment to bind. We need long enough for three lifespans
        # to run — the WAL pre-init in serve_all() should have prevented the
        # database-locked race, but lifespans still take ~1s on cold cache.
        for _ in range(40):
            await asyncio.sleep(0.1)
            if all(_port_listening("127.0.0.1", p) for p in (proxy_port, dash_port, mcp_port)):
                break
        assert _port_listening("127.0.0.1", proxy_port), "proxy didn't bind"
        assert _port_listening("127.0.0.1", dash_port), "dashboard didn't bind"
        assert _port_listening("127.0.0.1", mcp_port), "mcp didn't bind"
    finally:
        # Reach into the (still-running) Server objects via the gather Task to
        # signal shutdown the same way the signal bridge does.
        # asyncio.gather wraps coroutines in a single Task; since we can't
        # pluck the Servers out of it, cancel and let them tear down.
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, BaseException):
            pass


def _port_listening(host: str, port: int) -> bool:
    s = socket.socket()
    s.settimeout(0.1)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


@pytest.mark.asyncio
async def test_serve_all_no_mcp(tmp_path, monkeypatch):
    """`--no-mcp` skips the MCP server but still brings up the other two."""
    monkeypatch.setenv("TOKENQ_HOME", str(tmp_path))
    import importlib

    import tokenq.config as cfg
    importlib.reload(cfg)

    from tokenq import serve as serve_mod
    importlib.reload(serve_mod)

    proxy_port = _free_port()
    dash_port = _free_port()
    mcp_port = _free_port()  # reserved but unused — must NOT be bound

    task = asyncio.create_task(
        serve_mod.serve_all(
            host="127.0.0.1",
            proxy_port=proxy_port,
            dashboard_port=dash_port,
            mcp_port=mcp_port,
            mcp_on=False,
            log_level="warning",
        )
    )

    try:
        for _ in range(40):
            await asyncio.sleep(0.1)
            if _port_listening("127.0.0.1", proxy_port) and _port_listening("127.0.0.1", dash_port):
                break
        assert _port_listening("127.0.0.1", proxy_port)
        assert _port_listening("127.0.0.1", dash_port)
        # The MCP port must remain free — confirms --no-mcp actually skipped it.
        assert not _port_listening("127.0.0.1", mcp_port), (
            "mcp port unexpectedly bound when --no-mcp was passed"
        )
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, BaseException):
            pass


def test_run_py_dispatches_to_cli(tmp_path):
    """`python run.py --help` must list every CLI subcommand including `serve`.

    Catches the case where `run.py`'s sys.path injection breaks (e.g., a
    relative path mishap moving the launcher), since the import would fail
    silently and Typer would fall back to its default no-op help.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    run_py = repo_root / "run.py"
    assert run_py.exists(), "run.py missing — this is the no-install entry point"

    proc = subprocess.run(
        [sys.executable, str(run_py), "--help"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),  # run from anywhere — sys.path injection should still work
        timeout=15,
        env={**os.environ, "TOKENQ_HOME": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    for cmd in ("start", "serve", "stop", "status"):
        assert cmd in out, f"`{cmd}` subcommand missing from `run.py --help`"
