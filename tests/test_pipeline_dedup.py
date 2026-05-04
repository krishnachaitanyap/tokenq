"""Tests for ToolResultDedup."""
from __future__ import annotations


async def test_dedups_identical_large_blocks():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.dedup import STUB, ToolResultDedup

    big = "x" * 1000
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": big},
                    {"type": "tool_result", "tool_use_id": "b", "content": big},
                ],
            }
        ]
    }
    req = PipelineRequest(body=body, headers={})
    await ToolResultDedup().run(req)

    blocks = body["messages"][0]["content"]
    assert blocks[0]["content"] == big
    assert blocks[1]["content"] == STUB
    assert req.metadata["saved_tokens"] > 0


async def test_keeps_uniques():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.dedup import ToolResultDedup

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": "a" * 1000},
                    {"type": "tool_result", "tool_use_id": "b", "content": "b" * 1000},
                ],
            }
        ]
    }
    req = PipelineRequest(body=body, headers={})
    await ToolResultDedup().run(req)
    blocks = body["messages"][0]["content"]
    assert blocks[0]["content"] == "a" * 1000
    assert blocks[1]["content"] == "b" * 1000
    assert req.metadata.get("saved_tokens", 0) == 0


async def test_skips_short_blocks():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.dedup import ToolResultDedup

    short = "small output"
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": short},
                    {"type": "tool_result", "tool_use_id": "b", "content": short},
                ],
            }
        ]
    }
    req = PipelineRequest(body=body, headers={})
    await ToolResultDedup().run(req)
    blocks = body["messages"][0]["content"]
    assert blocks[1]["content"] == short


async def test_handles_list_content_format():
    """tool_result content can be a list of {type:text,text:...} blocks."""
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.dedup import STUB, ToolResultDedup

    big = "y" * 1000
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "a",
                        "content": [{"type": "text", "text": big}],
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "b",
                        "content": [{"type": "text", "text": big}],
                    },
                ],
            }
        ]
    }
    req = PipelineRequest(body=body, headers={})
    await ToolResultDedup().run(req)
    blocks = body["messages"][0]["content"]
    assert blocks[1]["content"] == STUB
