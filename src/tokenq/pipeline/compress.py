"""Compress tool_result content before forwarding upstream.

Three transforms, all conservative:
  1. Strip ANSI escape codes (terminal colors, cursor moves).
  2. Collapse runs of >=3 blank lines into a single blank line.
  3. If a result body exceeds COMPRESS_MAX_LINES, keep the first/last
     COMPRESS_KEEP_LINES and replace the middle with an omission marker.
"""
from __future__ import annotations

import re
from typing import Any

from ..config import COMPRESS_KEEP_LINES, COMPRESS_MAX_LINES
from . import PipelineRequest, Stage

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
BLANK_RUN_RE = re.compile(r"\n\s*\n(\s*\n)+")


def _compress_text(text: str, max_lines: int, keep: int) -> str:
    text = ANSI_RE.sub("", text)
    text = BLANK_RUN_RE.sub("\n\n", text)
    lines = text.split("\n")
    if len(lines) > max_lines:
        omitted = len(lines) - 2 * keep
        head = lines[:keep]
        tail = lines[-keep:]
        text = "\n".join(
            head
            + [f"... [{omitted} lines omitted to save tokens] ..."]
            + tail
        )
    return text


def _set_block_text(block: dict[str, Any], new_text: str) -> None:
    content = block.get("content")
    if isinstance(content, str):
        block["content"] = new_text
    elif isinstance(content, list):
        merged = False
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and not merged:
                item["text"] = new_text
                merged = True
            elif isinstance(item, dict) and item.get("type") == "text":
                item["text"] = ""
        if not merged:
            content.append({"type": "text", "text": new_text})


def _block_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text") or ""
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


class ToolOutputCompressor(Stage):
    name = "compress"

    def __init__(
        self,
        max_lines: int = COMPRESS_MAX_LINES,
        keep_lines: int = COMPRESS_KEEP_LINES,
    ) -> None:
        self.max_lines = max_lines
        self.keep_lines = keep_lines

    async def run(self, req: PipelineRequest):
        messages = req.body.get("messages")
        if not isinstance(messages, list) or not messages:
            return req

        # Mutate every tool_result so the wire bytes stay consistent across
        # turns (changing prior turns would invalidate Anthropic's prompt cache,
        # which would cost more than it saves). But only credit savings for
        # tool_results in the last message — those are the genuinely new bytes
        # this turn. Earlier ones were credited on a previous turn and are now
        # being read from upstream cache.
        last_idx = len(messages) - 1
        new_saved_chars = 0
        for i, msg in enumerate(messages):
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                original = _block_text(block)
                if not original:
                    continue
                compressed = _compress_text(
                    original, self.max_lines, self.keep_lines
                )
                if len(compressed) < len(original):
                    if i == last_idx:
                        new_saved_chars += len(original) - len(compressed)
                    _set_block_text(block, compressed)

        if new_saved_chars > 0:
            saved = new_saved_chars // 4
            req.metadata["saved_tokens"] = req.metadata.get("saved_tokens", 0) + saved
            req.metadata["saved_by_compress"] = req.metadata.get("saved_by_compress", 0) + saved
        return req
