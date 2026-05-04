from __future__ import annotations

import sqlite3

import pytest


async def test_init_db_creates_schema(tmp_home):
    from tokenq.config import DB_PATH
    from tokenq.storage import init_db

    await init_db()
    assert DB_PATH.exists()

    con = sqlite3.connect(DB_PATH)
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"requests", "cache", "bandit_state"}.issubset(tables)


async def test_log_request_inserts_row(tmp_home):
    from tokenq.config import DB_PATH
    from tokenq.storage import init_db, log_request

    await init_db()
    await log_request(
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=50,
        latency_ms=234,
        status_code=200,
        estimated_cost_usd=0.001,
    )

    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT model, input_tokens, output_tokens FROM requests").fetchall()
    assert rows == [("claude-sonnet-4-6", 100, 50)]


def test_pricing_estimate(tmp_home):
    from tokenq.pricing import estimate_cost

    # Sonnet 4.6: $3/M input, $15/M output
    assert estimate_cost("claude-sonnet-4-6", 1_000_000, 0) == pytest.approx(3.0)
    assert estimate_cost("claude-sonnet-4-6", 0, 1_000_000) == pytest.approx(15.0)
    # Unknown model returns 0 rather than crashing
    assert estimate_cost("gpt-4", 1000, 1000) == 0.0
    assert estimate_cost(None, 1000, 1000) == 0.0
