"""Deduplicate identical tool_result blocks within a single request.

When an agent re-reads the same file or re-runs the same command, the prior
tool_result is still in the messages array. We replace later identical copies
with a small reference stub so only one copy is sent upstream.
"""
from __future__ import annotations

import hashlib
from typing import Any

from ..config import DEDUP_MIN_CHARS
from . import PipelineRequest, Stage

STUB = "[duplicate of earlier tool_result, omitted to save tokens]"


def _result_text(block: dict[str, Any]) -> str:
    """Flatten a tool_result content payload into a comparable string."""
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text") or "")
        return "".join(parts)
    return ""


class ToolResultDedup(Stage):
    name = "dedup"

    def __init__(self, min_chars: int = DEDUP_MIN_CHARS) -> None:
        self.min_chars = min_chars

    async def run(self, req: PipelineRequest):
        messages = req.body.get("messages")
        if not isinstance(messages, list):
            return req

        seen: set[str] = set()
        saved_chars = 0

        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for i, block in enumerate(content):
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                text = _result_text(block)
                if len(text) < self.min_chars:
                    continue
                h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if h in seen:
                    saved_chars += len(text) - len(STUB)
                    content[i] = {
                        "type": "tool_result",
                        "tool_use_id": block.get("tool_use_id", ""),
                        "content": STUB,
                    }
                else:
                    seen.add(h)

        if saved_chars > 0:
            saved = saved_chars // 4
            req.metadata["saved_tokens"] = req.metadata.get("saved_tokens", 0) + saved
            req.metadata["saved_by_dedup"] = req.metadata.get("saved_by_dedup", 0) + saved
        return req
