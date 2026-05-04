"""Tests for the PID-file / daemon helpers and the start/stop/restart CLI.

The daemon module is unit-testable without ever forking uvicorn — we use a
trivial sleeper subprocess as a stand-in for the proxy. The CLI commands
that *would* spawn uvicorn (start, restart) are tested at the no-detach
short-circuit level (already-running detection); a real spawn integration
test is intentionally out of scope here.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
from typer.testing import CliRunner


# ---------- pure helpers ----------

def test_read_pid_missing_returns_none(tmp_home):
    from tokenq.daemon import read_pid
    assert read_pid() is None


def test_write_then_read_pid(tmp_home):
    from tokenq.daemon import read_pid, write_pid
    write_pid(12345)
    assert read_pid() == 12345


def test_clear_pid_is_idempotent(tmp_home):
    from tokenq.daemon import clear_pid, read_pid, write_pid
    write_pid(99)
    clear_pid()
    assert read_pid() is None
    clear_pid()  # second call must not raise even though file is gone


def test_read_pid_handles_garbage(tmp_home):
    from tokenq.config import PID_PATH
    from tokenq.daemon import read_pid
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text("not-a-number")
    assert read_pid() is None


def test_is_alive_self_is_true(tmp_home):
    from tokenq.daemon import is_alive
    assert is_alive(os.getpid()) is True


def test_is_alive_unused_pid_is_false(tmp_home):
    from tokenq.daemon import is_alive
    # PID 1 always exists on POSIX; pick something almost certainly free.
    # Sentinel: PID 0 means "send to my process group" — safely treated as
    # not-alive by our helper.
    assert is_alive(0) is False


def test_running_pid_clears_stale(tmp_home):
    """A PID file pointing at a dead process must be cleaned up so a
    subsequent start doesn't refuse to launch."""
    from tokenq.config import PID_PATH
    from tokenq.daemon import running_pid, write_pid
    # Use a definitely-dead PID — fork a child, wait for it to exit, then
    # record its PID. Linux/macOS recycle PIDs only over time, so for a
    # narrow window after exit it's safely unique.
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    write_pid(proc.pid)
    assert running_pid() is None  # stale → returns None
    assert not PID_PATH.exists()  # …and clears the file


# ---------- spawn + stop integration ----------

def _spawn_sleeper() -> subprocess.Popen:
    """Stand-in for the proxy: a subprocess that handles SIGTERM cleanly."""
    return subprocess.Popen([
        sys.executable, "-c",
        "import signal, time; signal.signal(signal.SIGTERM, lambda *_: __import__('sys').exit(0)); time.sleep(60)",
    ])


def test_stop_pid_sigterm_path(tmp_home):
    from tokenq.daemon import is_alive, stop_pid
    proc = _spawn_sleeper()
    try:
        assert is_alive(proc.pid)
        ok = stop_pid(proc.pid, timeout_seconds=3)
        assert ok is True
        assert not is_alive(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_stop_pid_force_sigkill(tmp_home):
    from tokenq.daemon import is_alive, stop_pid
    # Process that ignores SIGTERM — only SIGKILL works.
    proc = subprocess.Popen([
        sys.executable, "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
    ])
    try:
        assert is_alive(proc.pid)
        ok = stop_pid(proc.pid, force=True, timeout_seconds=2)
        assert ok is True
        assert not is_alive(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_stop_pid_escalates_to_sigkill_after_timeout(tmp_home):
    """SIGTERM-ignoring process gets escalated to SIGKILL automatically when
    the timeout elapses. Tight timeout so the test stays fast."""
    from tokenq.daemon import is_alive, stop_pid
    proc = subprocess.Popen([
        sys.executable, "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
    ])
    try:
        ok = stop_pid(proc.pid, timeout_seconds=0.5)
        assert ok is True
        assert not is_alive(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_stop_pid_already_dead_returns_true(tmp_home):
    from tokenq.daemon import stop_pid
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    # PID is gone; stop should report success.
    assert stop_pid(proc.pid, timeout_seconds=0.5) is True


# ---------- CLI integration ----------

@pytest.fixture
def runner():
    return CliRunner()


def test_status_shows_stopped(runner, tmp_home):
    from tokenq.cli import app
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "stopped" in result.stdout


def test_status_shows_running_when_pid_alive(runner, tmp_home):
    from tokenq.cli import app
    from tokenq.daemon import write_pid
    write_pid(os.getpid())  # current test process is "running"
    try:
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "running" in result.stdout
        assert str(os.getpid()) in result.stdout
    finally:
        from tokenq.daemon import clear_pid
        clear_pid()


def test_stop_when_not_running_is_noop(runner, tmp_home):
    from tokenq.cli import app
    result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0
    assert "not running" in result.stdout


def test_stop_terminates_process_recorded_in_pid(runner, tmp_home):
    from tokenq.cli import app
    from tokenq.daemon import is_alive, write_pid
    proc = _spawn_sleeper()
    write_pid(proc.pid)
    try:
        result = runner.invoke(app, ["stop", "--timeout", "3"])
        assert result.exit_code == 0
        assert "stopped" in result.stdout
        assert not is_alive(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_start_refuses_when_already_running(runner, tmp_home):
    """If the PID file points at a live process, start (any mode) refuses
    rather than producing a duplicate proxy on the same port."""
    from tokenq.cli import app
    from tokenq.daemon import write_pid
    write_pid(os.getpid())
    try:
        result = runner.invoke(app, ["start", "--detach"])
        assert result.exit_code == 1
        assert "already running" in result.stdout
    finally:
        from tokenq.daemon import clear_pid
        clear_pid()


def test_logs_command_handles_missing_log(runner, tmp_home):
    """Before any --detach run, the log file doesn't exist. The command
    must not blow up (typer would otherwise let the os.execvp fail)."""
    from tokenq.cli import app
    result = runner.invoke(app, ["logs"])
    assert result.exit_code == 0
    assert "no daemon log" in result.stdout
