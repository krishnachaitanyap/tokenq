"""tokenq CLI.

Commands:
  tokenq start            — run proxy on 8089 and dashboard on 8090 (foreground)
  tokenq start --detach   — same, but run in background; returns immediately
  tokenq stop             — terminate a running proxy (graceful by default)
  tokenq restart          — stop then start --detach
  tokenq status           — print last-24h stats + liveness
  tokenq logs             — tail the daemon log
  tokenq reset            — wipe the local DB
  tokenq compress-skill   — rewrite a bloated SKILL.md into LLM-optimized form
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
import sys
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from .config import (
    DAEMON_LOG_PATH,
    DASHBOARD_PORT,
    DB_PATH,
    LOG_LEVEL,
    MCP_ENABLED,
    MCP_PORT,
    PROXY_HOST,
    PROXY_PORT,
)
from . import daemon as _daemon

app = typer.Typer(help="tokenq — local proxy that cuts your Claude API bill")


def _run_proxy(host: str, port: int, log_level: str) -> None:
    uvicorn.run("tokenq.proxy.app:app", host=host, port=port, log_level=log_level)


def _run_dashboard(host: str, port: int, log_level: str) -> None:
    uvicorn.run("tokenq.dashboard.app:app", host=host, port=port, log_level=log_level)


def _run_mcp(host: str, port: int, log_level: str) -> None:
    uvicorn.run("tokenq.bigmemory.mcp:app", host=host, port=port, log_level=log_level)


@app.command()
def start(
    host: Annotated[str, typer.Option(help="Bind address.")] = PROXY_HOST,
    port: Annotated[int, typer.Option(help="Proxy port.")] = PROXY_PORT,
    dashboard_port: Annotated[int, typer.Option(help="Dashboard port.")] = DASHBOARD_PORT,
    mcp_port: Annotated[int, typer.Option(help="bigmemory MCP port.")] = MCP_PORT,
    no_mcp: Annotated[bool, typer.Option("--no-mcp", help="Skip the bigmemory MCP server.")] = False,
    log_level: Annotated[str, typer.Option(help="info|warning|debug")] = LOG_LEVEL,
    detach: Annotated[
        bool,
        typer.Option("--detach", "-d", help="Run in the background and return immediately."),
    ] = False,
) -> None:
    """Start the proxy, dashboard, and (by default) the bigmemory MCP server.

    Pass --detach (-d) to spawn it as a background process — useful when you
    want to control tokenq from any shell, or have it survive terminal close.
    The PID is recorded at ~/.tokenq/tokenq.pid so `tokenq stop` can find it.
    """
    mcp_on = MCP_ENABLED and not no_mcp

    # Reject double-start regardless of whether we're foregrounding or detaching.
    # Skip the check when our parent set TOKENQ_DETACHED_CHILD=1 — that means
    # we ARE the freshly-spawned child of `start --detach`, and the PID file
    # already (correctly) points at us; we'd otherwise reject ourselves.
    if not os.environ.pop("TOKENQ_DETACHED_CHILD", ""):
        existing = _daemon.running_pid()
        if existing is not None:
            typer.echo(f"tokenq is already running (PID {existing}). Use `tokenq stop` first.")
            raise typer.Exit(1)

    if detach:
        # Re-invoke ourselves WITHOUT --detach so the child runs the
        # foreground path. Forward every other knob the user set.
        args = ["start", f"--host={host}", f"--port={port}",
                f"--dashboard-port={dashboard_port}",
                f"--mcp-port={mcp_port}", f"--log-level={log_level}"]
        if no_mcp:
            args.append("--no-mcp")
        pid = _daemon.spawn_detached(args)
        typer.echo("")
        typer.echo(f"  tokenq started in background (PID {pid})")
        typer.echo(f"  proxy      http://{host}:{port}")
        typer.echo(f"  dashboard  http://{host}:{dashboard_port}")
        if mcp_on:
            typer.echo(f"  MCP        http://{host}:{mcp_port}/mcp")
        typer.echo(f"  logs       {DAEMON_LOG_PATH}  (`tokenq logs -f` to tail)")
        typer.echo(f"  stop with  `tokenq stop`")
        typer.echo("")
        return

    typer.echo("")
    typer.echo(f"  tokenq proxy     →  http://{host}:{port}")
    typer.echo(f"  tokenq dashboard →  http://{host}:{dashboard_port}")
    if mcp_on:
        typer.echo(f"  bigmemory MCP    →  http://{host}:{mcp_port}/mcp")
    typer.echo("")
    typer.echo("  Point Claude Code at it:")
    typer.echo(f"    export ANTHROPIC_BASE_URL=http://{host}:{port}")
    if mcp_on:
        typer.echo("    claude mcp add --transport http bigmemory "
                   f"http://{host}:{mcp_port}/mcp")
    typer.echo("")

    # Foreground path: write our PID so an external shell can `tokenq stop`
    # us. Cleared in the finally — this is the cleanup path on SIGTERM and
    # on Ctrl-C.
    _daemon.write_pid(os.getpid())

    ctx = mp.get_context("spawn")
    children: list[mp.Process] = []
    children.append(ctx.Process(
        target=_run_dashboard,
        args=(host, dashboard_port, "warning"),
        daemon=True,
    ))
    if mcp_on:
        children.append(ctx.Process(
            target=_run_mcp,
            args=(host, mcp_port, "warning"),
            daemon=True,
        ))
    for p in children:
        p.start()
    try:
        _run_proxy(host, port, log_level)
    finally:
        for p in children:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        _daemon.clear_pid()


@app.command()
def stop(
    timeout: Annotated[
        float, typer.Option(help="Seconds to wait for graceful shutdown before SIGKILL."),
    ] = 10.0,
    force: Annotated[
        bool, typer.Option("--force", help="Send SIGKILL immediately, skip graceful shutdown."),
    ] = False,
) -> None:
    """Stop a running tokenq proxy.

    Sends SIGTERM by default — uvicorn runs its lifespan shutdown so the DB
    closes cleanly and in-flight requests drain. If the process doesn't
    exit within --timeout seconds we escalate to SIGKILL.
    """
    pid = _daemon.running_pid()
    if pid is None:
        typer.echo("tokenq is not running.")
        return
    typer.echo(f"stopping tokenq (PID {pid})…")
    ok = _daemon.stop_pid(pid, timeout_seconds=timeout, force=force)
    _daemon.clear_pid()
    if ok:
        typer.echo("stopped.")
    else:
        typer.echo("warning: could not confirm shutdown — process may still be running.")
        raise typer.Exit(1)


@app.command()
def restart(
    host: Annotated[str, typer.Option(help="Bind address.")] = PROXY_HOST,
    port: Annotated[int, typer.Option(help="Proxy port.")] = PROXY_PORT,
    dashboard_port: Annotated[int, typer.Option(help="Dashboard port.")] = DASHBOARD_PORT,
    mcp_port: Annotated[int, typer.Option(help="bigmemory MCP port.")] = MCP_PORT,
    no_mcp: Annotated[bool, typer.Option("--no-mcp", help="Skip the bigmemory MCP server.")] = False,
    log_level: Annotated[str, typer.Option(help="info|warning|debug")] = LOG_LEVEL,
    timeout: Annotated[float, typer.Option(help="Seconds to wait for stop.")] = 10.0,
) -> None:
    """Stop then start --detach, with a brief pause for the port to free up."""
    import time
    pid = _daemon.running_pid()
    if pid is not None:
        typer.echo(f"stopping tokenq (PID {pid})…")
        _daemon.stop_pid(pid, timeout_seconds=timeout)
        _daemon.clear_pid()
        # Small grace period: SO_REUSEADDR + WAL → port usually frees fast,
        # but a tight restart loop can race the kernel's TIME_WAIT cleanup.
        time.sleep(0.5)
    start(
        host=host, port=port, dashboard_port=dashboard_port,
        mcp_port=mcp_port, no_mcp=no_mcp, log_level=log_level, detach=True,
    )


@app.command()
def logs(
    follow: Annotated[bool, typer.Option("-f", "--follow", help="Stream new lines as they arrive.")] = False,
    lines: Annotated[int, typer.Option("-n", "--lines", help="Last N lines to print.")] = 200,
) -> None:
    """Print (or tail) the detached-daemon log."""
    if not DAEMON_LOG_PATH.exists():
        typer.echo("no daemon log yet — start tokenq with `tokenq start --detach` first.")
        return
    if follow:
        # Defer to `tail -f` rather than reimplementing the 99% case poorly.
        os.execvp("tail", ["tail", "-n", str(lines), "-f", str(DAEMON_LOG_PATH)])
    else:
        os.execvp("tail", ["tail", "-n", str(lines), str(DAEMON_LOG_PATH)])


@app.command()
def status() -> None:
    """Print proxy liveness + last-24h usage stats from the local DB."""
    pid = _daemon.running_pid()
    if pid is not None:
        typer.echo(f"tokenq: running (PID {pid})")
    else:
        typer.echo("tokenq: stopped")

    if not DB_PATH.exists():
        typer.echo("No data yet — run `tokenq start` and make some requests.")
        raise typer.Exit(0)

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute(
        """
        SELECT COUNT(*) AS n,
               COALESCE(SUM(input_tokens), 0) AS in_t,
               COALESCE(SUM(output_tokens), 0) AS out_t,
               COALESCE(SUM(estimated_cost_usd), 0) AS cost,
               COALESCE(SUM(cached_locally), 0) AS cache_hits,
               COALESCE(AVG(latency_ms), 0) AS avg_lat
        FROM requests
        WHERE ts > strftime('%s', 'now') - 86400
        """
    ).fetchone()
    typer.echo(
        f"last 24h: {row['n']} req | {row['in_t']:,} in / {row['out_t']:,} out "
        f"| ~${row['cost']:.2f} | {row['cache_hits']} local-cache hits "
        f"| {row['avg_lat']:.0f}ms avg"
    )


@app.command()
def mcp(
    host: Annotated[str, typer.Option(help="Bind address.")] = PROXY_HOST,
    port: Annotated[int, typer.Option(help="MCP port.")] = MCP_PORT,
    log_level: Annotated[str, typer.Option(help="info|warning|debug")] = LOG_LEVEL,
) -> None:
    """Run only the bigmemory MCP server (no proxy, no dashboard)."""
    typer.echo("")
    typer.echo(f"  bigmemory MCP →  http://{host}:{port}/mcp")
    typer.echo("")
    typer.echo("  Register with Claude Code:")
    typer.echo(f"    claude mcp add --transport http bigmemory http://{host}:{port}/mcp")
    typer.echo("")
    _run_mcp(host, port, log_level)


@app.command()
def reset(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Wipe all stored data."""
    if not DB_PATH.exists():
        typer.echo("Nothing to reset.")
        raise typer.Exit(0)
    if not yes:
        confirm = typer.confirm(f"Delete {DB_PATH}?", default=False)
        if not confirm:
            raise typer.Exit(1)
    DB_PATH.unlink()
    typer.echo(f"Removed {DB_PATH}")


@app.command("compress-skill")
def compress_skill(
    path: Annotated[Path, typer.Argument(
        help="Path to SKILL.md (or any markdown file).",
        exists=True, dir_okay=False, readable=True,
    )],
    model: Annotated[str, typer.Option(help="Anthropic model.")] = "claude-sonnet-4-6",
    output: Annotated[
        Path | None, typer.Option(help="Write result to this path instead of in-place.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print result; do not write.")
    ] = False,
    no_backup: Annotated[
        bool, typer.Option("--no-backup", help="Skip writing .bak when overwriting in place.")
    ] = False,
    timeout: Annotated[float, typer.Option(help="API timeout (seconds).")] = 120.0,
) -> None:
    """Rewrite a SKILL.md into LLM-optimized form (target 50–70% smaller).

    Calls the Claude API. Set `ANTHROPIC_API_KEY` in your environment.
    Frontmatter (the leading `---` block) is preserved unchanged.
    """
    from .skill_compress import CompressionError, compress_file

    try:
        result = compress_file(
            path,
            model=model,
            output=output,
            dry_run=dry_run,
            no_backup=no_backup,
            timeout=timeout,
        )
    except CompressionError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("")
    typer.echo(f"  {path}")
    typer.echo(f"    before:  {result.before_tokens:>6,} tokens")
    typer.echo(f"    after:   {result.after_tokens:>6,} tokens")
    typer.echo(
        f"    saved:   {result.saved_tokens:>6,} tokens ({result.saved_pct:.1f}%)"
    )
    if result.written:
        typer.echo(f"    wrote:   {result.output_path}")
    else:
        typer.echo("    (dry run — not written)")
    typer.echo("")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
