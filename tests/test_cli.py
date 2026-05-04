"""CLI tests for `tokenq status` and `tokenq reset`.

We exercise the typer app via `CliRunner` so we don't actually spawn uvicorn.
The `start` command is intentionally not unit-tested here — it spawns long-
running uvicorn processes; integration coverage lives in test_proxy.py.
"""
from __future__ import annotations

import sqlite3
import time

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def test_status_no_db(runner, tmp_home):
    from tokenq.cli import app

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "No data yet" in result.stdout


def test_status_with_data(runner, tmp_home):
    from tokenq.cli import app
    from tokenq.config import DB_PATH

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(
        """
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            model TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cached_locally INTEGER DEFAULT 0,
            latency_ms INTEGER DEFAULT 0,
            estimated_cost_usd REAL DEFAULT 0
        );
        """
    )
    con.execute(
        "INSERT INTO requests (ts, model, input_tokens, output_tokens, "
        "cached_locally, latency_ms, estimated_cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (time.time(), "claude-sonnet-4-6", 100, 20, 1, 150, 0.001),
    )
    con.commit()
    con.close()

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "1 req" in result.stdout
    assert "100" in result.stdout


def test_reset_no_db(runner, tmp_home):
    from tokenq.cli import app

    result = runner.invoke(app, ["reset", "--yes"])
    assert result.exit_code == 0
    assert "Nothing to reset" in result.stdout


def test_reset_with_confirmation_declined(runner, tmp_home):
    from tokenq.cli import app
    from tokenq.config import DB_PATH

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.write_bytes(b"sqlite stub")

    # Decline the confirmation prompt.
    result = runner.invoke(app, ["reset"], input="n\n")
    assert result.exit_code == 1
    assert DB_PATH.exists()


def test_reset_with_yes_flag_deletes_db(runner, tmp_home):
    from tokenq.cli import app
    from tokenq.config import DB_PATH

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.write_bytes(b"sqlite stub")

    result = runner.invoke(app, ["reset", "--yes"])
    assert result.exit_code == 0
    assert "Removed" in result.stdout
    assert not DB_PATH.exists()


def test_help_lists_commands(runner):
    from tokenq.cli import app

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("start", "status", "reset", "compress-skill"):
        assert cmd in result.stdout
