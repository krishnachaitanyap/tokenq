"""Tests for SKILL.md compression."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _api_response(text: str) -> dict:
    return {
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


def test_split_frontmatter_with_yaml():
    from tokenq.skill_compress import split_frontmatter

    text = "---\nname: foo\ndescription: bar\n---\nbody text\nmore body\n"
    fm, body = split_frontmatter(text)
    assert fm == "---\nname: foo\ndescription: bar\n---\n"
    assert body == "body text\nmore body\n"


def test_split_frontmatter_no_yaml():
    from tokenq.skill_compress import split_frontmatter

    fm, body = split_frontmatter("just body text\n")
    assert fm == ""
    assert body == "just body text\n"


def test_split_frontmatter_only_frontmatter():
    from tokenq.skill_compress import split_frontmatter

    fm, body = split_frontmatter("---\nname: x\n---\n")
    assert fm == "---\nname: x\n---\n"
    assert body == ""


def test_strip_outer_code_fence():
    from tokenq.skill_compress import _strip_outer_code_fence

    assert _strip_outer_code_fence("```markdown\nhello\n```") == "hello\n"
    assert _strip_outer_code_fence("```\nhello\n```") == "hello\n"
    # Plain text untouched.
    assert _strip_outer_code_fence("hello world") == "hello world"


def test_compress_file_preserves_frontmatter(tmp_path: Path, httpx_mock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    httpx_mock.add_response(json=_api_response("compressed body"))

    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "---\nname: foo\ndescription: bar\n---\nlong original body content\n",
        encoding="utf-8",
    )

    from tokenq.skill_compress import compress_file

    result = compress_file(skill)

    rewritten = skill.read_text(encoding="utf-8")
    assert rewritten.startswith("---\nname: foo\ndescription: bar\n---\n")
    assert "compressed body" in rewritten
    assert result.written is True
    assert result.output_path == skill
    assert skill.with_suffix(".md.bak").exists()


def test_compress_file_dry_run_does_not_write(tmp_path: Path, httpx_mock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    httpx_mock.add_response(json=_api_response("compressed body"))

    skill = tmp_path / "SKILL.md"
    original = "original body content here\n"
    skill.write_text(original, encoding="utf-8")

    from tokenq.skill_compress import compress_file

    result = compress_file(skill, dry_run=True)

    assert skill.read_text(encoding="utf-8") == original
    assert not skill.with_suffix(".md.bak").exists()
    assert result.written is False
    assert result.output_path is None


def test_compress_file_no_backup(tmp_path: Path, httpx_mock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    httpx_mock.add_response(json=_api_response("compressed body"))

    skill = tmp_path / "SKILL.md"
    skill.write_text("original\n", encoding="utf-8")

    from tokenq.skill_compress import compress_file

    compress_file(skill, no_backup=True)
    assert not skill.with_suffix(".md.bak").exists()


def test_compress_file_output_target_does_not_touch_original(
    tmp_path: Path, httpx_mock, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    httpx_mock.add_response(json=_api_response("compressed"))

    skill = tmp_path / "SKILL.md"
    skill.write_text("original body\n", encoding="utf-8")
    out = tmp_path / "out.md"

    from tokenq.skill_compress import compress_file

    result = compress_file(skill, output=out)
    assert out.exists()
    assert "compressed" in out.read_text(encoding="utf-8")
    # Original is untouched, no backup written when output is set.
    assert skill.read_text(encoding="utf-8") == "original body\n"
    assert not skill.with_suffix(".md.bak").exists()
    assert result.output_path == out


def test_request_shape_matches_api_contract(httpx_mock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-key")
    httpx_mock.add_response(json=_api_response("ok"))

    from tokenq.skill_compress import compress_body

    compress_body("input body")

    req = httpx_mock.get_request()
    assert req.headers["x-api-key"] == "secret-key"
    assert req.headers["anthropic-version"] == "2023-06-01"

    payload = json.loads(req.content)
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload["temperature"] == 0.0
    # Cache control on system block — best-effort prompt caching across runs.
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "input body" in payload["messages"][0]["content"]


def test_missing_api_key_errors(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from tokenq.skill_compress import CompressionError, compress_body

    with pytest.raises(CompressionError, match="ANTHROPIC_API_KEY"):
        compress_body("anything")


def test_rate_limit_error_surfaces_retry_after(httpx_mock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    httpx_mock.add_response(
        status_code=429,
        headers={"retry-after": "30"},
        json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}},
    )

    from tokenq.skill_compress import CompressionError, compress_body

    with pytest.raises(CompressionError, match="rate limited"):
        compress_body("input")


def test_auth_error_message(httpx_mock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad-key")
    httpx_mock.add_response(
        status_code=401,
        json={"type": "error", "error": {"type": "authentication_error", "message": "x"}},
    )

    from tokenq.skill_compress import CompressionError, compress_body

    with pytest.raises(CompressionError, match="authentication"):
        compress_body("input")


def test_compress_file_logs_to_skill_compressions(
    tmp_path: Path, httpx_mock, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("TOKENQ_HOME", str(tmp_path / "tokenq-home"))
    httpx_mock.add_response(json=_api_response("compressed body"))

    # Re-resolve config + storage with the new TOKENQ_HOME so DB_PATH points
    # at the tmp dir.
    import importlib
    import tokenq.config as cfg
    import tokenq.storage as storage
    import tokenq.skill_compress as sc
    importlib.reload(cfg)
    importlib.reload(storage)
    importlib.reload(sc)

    skill = tmp_path / "SKILL.md"
    skill.write_text("---\nname: foo\n---\nbody to compress\n", encoding="utf-8")

    sc.compress_file(skill)

    import sqlite3
    con = sqlite3.connect(cfg.DB_PATH)
    rows = con.execute(
        "SELECT path, model, saved_tokens FROM skill_compressions"
    ).fetchall()
    con.close()
    assert len(rows) == 1
    assert rows[0][0] == str(skill)
    assert rows[0][1] == "claude-sonnet-4-6"


def test_dry_run_does_not_log(tmp_path: Path, httpx_mock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("TOKENQ_HOME", str(tmp_path / "tokenq-home"))
    httpx_mock.add_response(json=_api_response("compressed body"))

    import importlib
    import tokenq.config as cfg
    import tokenq.storage as storage
    import tokenq.skill_compress as sc
    importlib.reload(cfg)
    importlib.reload(storage)
    importlib.reload(sc)

    skill = tmp_path / "SKILL.md"
    skill.write_text("body\n", encoding="utf-8")
    sc.compress_file(skill, dry_run=True)

    import sqlite3
    con = sqlite3.connect(cfg.DB_PATH)
    # Table may not exist yet on a fresh DB — both outcomes count as "not logged".
    try:
        rows = con.execute("SELECT COUNT(*) FROM skill_compressions").fetchone()
        assert rows[0] == 0
    except sqlite3.OperationalError:
        pass
    finally:
        con.close()


def test_strips_outer_fence_in_response(httpx_mock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    httpx_mock.add_response(json=_api_response("```markdown\nrules go here\n```"))

    from tokenq.skill_compress import compress_body

    out = compress_body("input")
    assert "rules go here" in out
    assert "```" not in out
