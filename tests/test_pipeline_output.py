"""Tests for OutputController."""
from __future__ import annotations


def _qa_body(text: str = "what's 2+2?", **extra) -> dict:
    body = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": text}],
    }
    body.update(extra)
    return body


def _tool_followup_body() -> dict:
    return {
        "model": "claude-sonnet-4-6",
        "messages": [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "x", "name": "Bash", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "x", "content": "ok"}
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# classify_turn
# ---------------------------------------------------------------------------


def test_classify_short_question_is_qa():
    from tokenq.pipeline.output import QA, classify_turn

    assert classify_turn(_qa_body("what's the capital of France?")) == QA


def test_classify_short_statement_is_qa():
    from tokenq.pipeline.output import QA, classify_turn

    assert classify_turn(_qa_body("define entropy")) == QA


def test_classify_implementation_request_is_code():
    from tokenq.pipeline.output import CODE, classify_turn

    assert (
        classify_turn(_qa_body("implement a quicksort in Python"))
        == CODE
    )


def test_classify_long_user_message_with_code_block_is_code():
    from tokenq.pipeline.output import CODE, classify_turn

    body = _qa_body("here is the file:\n```\n" + "x\n" * 50 + "```")
    assert classify_turn(body) == CODE


def test_classify_tool_followup_is_tool():
    from tokenq.pipeline.output import TOOL, classify_turn

    assert classify_turn(_tool_followup_body()) == TOOL


def test_classify_with_tools_array_is_tool():
    from tokenq.pipeline.output import TOOL, classify_turn

    body = _qa_body(
        "find auth bugs in this repo, it's a long task",
        tools=[{"name": "Bash"}],
    )
    assert classify_turn(body) == TOOL


def test_classify_unknown_falls_through():
    from tokenq.pipeline.output import UNKNOWN, classify_turn

    # Long-ish, no tools, no question mark, no code signals → leave alone.
    body = _qa_body("ok here's some context " * 50)
    assert classify_turn(body) == UNKNOWN


def test_classify_empty_body_is_unknown():
    from tokenq.pipeline.output import UNKNOWN, classify_turn

    assert classify_turn({}) == UNKNOWN
    assert classify_turn({"messages": []}) == UNKNOWN


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------


async def test_qa_caps_max_tokens_when_unset(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.output import OutputController

    stage = OutputController(caps_enabled=True)
    body = _qa_body("what's 2+2?")
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    assert body["max_tokens"] == stage.qa_max_tokens
    assert req.metadata["output_turn_class"] == "qa"
    assert req.metadata["output_cap_applied"] == stage.qa_max_tokens


async def test_qa_caps_max_tokens_when_above_ceiling(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.output import OutputController

    stage = OutputController(caps_enabled=True, qa_max_tokens=400)
    body = _qa_body("what's 2+2?", max_tokens=8192)
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    assert body["max_tokens"] == 400
    assert req.metadata["output_cap_original"] == 8192


async def test_cap_never_raises_low_value(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.output import OutputController

    stage = OutputController(caps_enabled=True, qa_max_tokens=800)
    body = _qa_body(max_tokens=100)
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    # 100 < 800 → already lower than ceiling, untouched.
    assert body["max_tokens"] == 100
    assert "output_cap_applied" not in req.metadata


async def test_caps_disabled_leaves_body_alone(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.output import OutputController

    stage = OutputController(caps_enabled=False)
    body = _qa_body()
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    assert "max_tokens" not in body


async def test_code_class_untouched(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.output import OutputController

    stage = OutputController(caps_enabled=True, terseness_enabled=True)
    body = _qa_body("implement a binary search tree", max_tokens=4096)
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    assert body["max_tokens"] == 4096
    assert "system" not in body  # terseness not injected


# ---------------------------------------------------------------------------
# Terseness
# ---------------------------------------------------------------------------


async def test_terseness_appends_to_string_system(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.output import TERSENESS_SUFFIX, OutputController

    stage = OutputController(terseness_enabled=True)
    body = _qa_body(system="You are helpful.")
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    assert body["system"].endswith(TERSENESS_SUFFIX)


async def test_terseness_is_idempotent(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.output import TERSENESS_SUFFIX, OutputController

    stage = OutputController(terseness_enabled=True)
    body = _qa_body(system="You are helpful." + TERSENESS_SUFFIX)
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    # Suffix already present — no double-append (cache prefix stays stable).
    assert body["system"].count(TERSENESS_SUFFIX) == 1


async def test_terseness_disabled_by_default(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.output import OutputController

    stage = OutputController()  # reads env, defaults False
    body = _qa_body(system="You are helpful.")
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    assert body["system"] == "You are helpful."


# ---------------------------------------------------------------------------
# Stop sequences
# ---------------------------------------------------------------------------


async def test_stop_sequences_appended_when_enabled(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.output import OutputController

    stage = OutputController(stop_seqs_enabled=True)
    body = _qa_body()
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    seqs = body.get("stop_sequences") or []
    assert any("Let me know" in s for s in seqs)


async def test_stop_sequences_preserve_existing(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.output import OutputController

    stage = OutputController(stop_seqs_enabled=True)
    body = _qa_body(stop_sequences=["</done>"])
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    seqs = body["stop_sequences"]
    assert "</done>" in seqs
    assert any("Let me know" in s for s in seqs)


async def test_stop_sequences_capped_at_four(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.output import OutputController

    stage = OutputController(stop_seqs_enabled=True)
    body = _qa_body(stop_sequences=["a", "b", "c", "d"])
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    # Already at the limit — none added.
    assert body["stop_sequences"] == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


async def test_unknown_class_does_not_mutate_body(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.output import OutputController

    stage = OutputController(
        caps_enabled=True, terseness_enabled=True, stop_seqs_enabled=True
    )
    long_no_signals = "x " * 200  # long, no question, no code, no tools
    body = _qa_body(long_no_signals)
    req = PipelineRequest(body=body, headers={})
    before = dict(body)
    await stage.run(req)

    assert body == before
    assert req.metadata["output_turn_class"] == "unknown"
