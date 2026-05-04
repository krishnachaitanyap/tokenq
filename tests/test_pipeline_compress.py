"""Tests for ToolOutputCompressor."""
from __future__ import annotations


async def test_strips_ansi_escapes():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.compress import ToolOutputCompressor

    text = "\x1b[31mred\x1b[0m \x1b[1mbold\x1b[0m end"
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": text},
                ],
            }
        ]
    }
    req = PipelineRequest(body=body, headers={})
    await ToolOutputCompressor().run(req)
    out = body["messages"][0]["content"][0]["content"]
    assert "\x1b" not in out
    assert "red" in out and "bold" in out
    assert req.metadata["saved_tokens"] > 0


async def test_truncates_long_outputs():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.compress import ToolOutputCompressor

    text = "\n".join(f"line {i}" for i in range(500))
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": text},
                ],
            }
        ]
    }
    req = PipelineRequest(body=body, headers={})
    await ToolOutputCompressor(max_lines=200, keep_lines=50).run(req)
    out = body["messages"][0]["content"][0]["content"]
    assert "lines omitted" in out
    assert "line 0" in out
    assert "line 499" in out
    # Should not still contain a middle line.
    assert "line 250" not in out


async def test_collapses_blank_runs():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.compress import ToolOutputCompressor

    text = "a\n\n\n\n\nb"
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "x", "content": text},
                ],
            }
        ]
    }
    req = PipelineRequest(body=body, headers={})
    await ToolOutputCompressor().run(req)
    out = body["messages"][0]["content"][0]["content"]
    assert out == "a\n\nb"


async def test_credits_only_last_message_tool_results():
    """Same tool_result reappearing as history must not be re-credited each turn.

    Claude Code re-sends the full transcript every turn. The compressor still
    rewrites historical tool_results (so wire bytes match what was cached
    upstream), but only the last message's tool_results represent new content
    that wasn't credited on a previous turn.
    """
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.compress import ToolOutputCompressor

    big = "\n".join(f"line {i}" for i in range(500))
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": big},
                ],
            },
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t2", "content": big},
                ],
            },
        ]
    }
    req = PipelineRequest(body=body, headers={})
    await ToolOutputCompressor(max_lines=200, keep_lines=50).run(req)

    # Both blocks are mutated (cache-key parity with prior turns).
    assert "lines omitted" in body["messages"][0]["content"][0]["content"]
    assert "lines omitted" in body["messages"][2]["content"][0]["content"]

    # But savings are credited for ONE block, not two — single-turn savings.
    single_turn_credit = req.metadata.get("saved_by_compress", 0)

    body2 = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t2", "content": big},
                ],
            }
        ]
    }
    req2 = PipelineRequest(body=body2, headers={})
    await ToolOutputCompressor(max_lines=200, keep_lines=50).run(req2)
    assert req2.metadata.get("saved_by_compress", 0) == single_turn_credit
    assert single_turn_credit > 0


async def test_short_output_unchanged():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.compress import ToolOutputCompressor

    text = "tiny output"
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "x", "content": text},
                ],
            }
        ]
    }
    req = PipelineRequest(body=body, headers={})
    await ToolOutputCompressor().run(req)
    out = body["messages"][0]["content"][0]["content"]
    assert out == text
    assert req.metadata.get("saved_tokens", 0) == 0
