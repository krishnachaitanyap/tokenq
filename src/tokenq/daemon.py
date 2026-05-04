"""Process-management primitives — PID file, liveness, detach, stop.

`tokenq start --detach` and `tokenq stop` use these helpers so the proxy can
be controlled from any shell, not just the one that started it. Foreground
`tokenq start` also writes the PID file so an external `tokenq stop` works
even when the process was started in the foreground from another terminal.

Design notes:
  - PID file lives at ~/.tokenq/tokenq.pid (configurable via TOKENQ_HOME).
  - Liveness uses os.kill(pid, 0) — POSIX-only signal-zero probe; raises
    ProcessLookupError if the PID is gone, PermissionError if it exists but
    we don't own it. We treat both as "running" vs "not running" cleanly.
  - Daemonization is simple Popen + start_new_session=True, NOT a full
    POSIX double-fork. The proxy is a long-running uvicorn process, not a
    classic Unix daemon — there's no controlling terminal to detach from
    after start_new_session is set.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import DAEMON_LOG_PATH, PID_PATH


def read_pid(pid_path: Path = PID_PATH) -> int | None:
    """Return the recorded PID, or None if the file is missing/garbage.

    A stale PID file (process no longer running) is *not* cleaned up here —
    that's the caller's job. We just report what we read.
    """
    try:
        with open(pid_path) as f:
            txt = f.read().strip()
    except FileNotFoundError:
        return None
    if not txt:
        return None
    try:
        return int(txt)
    except ValueError:
        return None


def write_pid(pid: int, pid_path: Path = PID_PATH) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n")


def clear_pid(pid_path: Path = PID_PATH) -> None:
    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass


def is_alive(pid: int) -> bool:
    """True iff `pid` refers to a running process.

    Uses signal 0 (the POSIX 'just check' probe). One subtlety: a
    SIGTERM'd subprocess remains in PID-slot limbo as a zombie until its
    parent reaps it, and `os.kill(pid, 0)` on a zombie still reports
    success. To avoid that false positive, we first try waitpid(WNOHANG):
    if the pid is our own child and has exited, this reaps it and we
    correctly report dead. Production callers (`tokenq stop` from a
    different shell) never have this relationship, so the waitpid call
    raises ChildProcessError and we fall through to the kill probe — net
    behavior unchanged for the real path.
    """
    if pid <= 0:
        return False
    try:
        reaped, _status = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False  # zombie successfully reaped; truly gone now
    except ChildProcessError:
        pass  # not our child — kill probe will handle it
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't have permission — still "running".
        return True
    return True


def running_pid(pid_path: Path = PID_PATH) -> int | None:
    """Return the PID if it's recorded AND alive; None otherwise. Side
    effect: clears a stale PID file so subsequent calls don't trip on it.
    """
    pid = read_pid(pid_path)
    if pid is None:
        return None
    if is_alive(pid):
        return pid
    # Stale PID — record left over from a crash. Clean up.
    clear_pid(pid_path)
    return None


def spawn_detached(
    args: list[str],
    *,
    log_path: Path = DAEMON_LOG_PATH,
    python: str | None = None,
) -> int:
    """Spawn a detached child running `python -m tokenq.cli <args>`.

    The child is given a fresh process group (start_new_session=True) so a
    SIGHUP from the parent terminal closing doesn't kill it. stdout/stderr
    redirect to log_path (append) so the user can tail it later. We pass
    TOKENQ_DETACHED_CHILD=1 in the env so the child knows the PID file
    already points at it (written here) and shouldn't refuse to start.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "ab", buffering=0)
    cmd = [python or sys.executable, "-m", "tokenq.cli", *args]
    env = os.environ.copy()
    env["TOKENQ_DETACHED_CHILD"] = "1"
    proc = subprocess.Popen(  # noqa: S603 — args is constructed, not user input
        cmd,
        stdout=log_fh,
        stderr=log_fh,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,  # don't die on parent SIGHUP
        env=env,
    )
    write_pid(proc.pid)
    return proc.pid


def stop_pid(
    pid: int,
    *,
    timeout_seconds: float = 10.0,
    force: bool = False,
) -> bool:
    """Terminate `pid`. Returns True if the process is gone after we're done.

    Sequence:
      1. Send SIGTERM (or SIGKILL if force=True).
      2. Poll for is_alive() at 100ms intervals up to timeout_seconds.
      3. If still alive after timeout, escalate to SIGKILL.

    SIGTERM lets uvicorn run its lifespan shutdown (closes the DB cleanly,
    drains in-flight requests). SIGKILL only as a last resort or when the
    user explicitly asked for force.
    """
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not is_alive(pid):
            return True
        time.sleep(0.1)

    # Timed out on SIGTERM — escalate.
    if not force:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        # Give the kernel a moment to reap.
        time.sleep(0.3)
    return not is_alive(pid)
