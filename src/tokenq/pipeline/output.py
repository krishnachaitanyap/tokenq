"""Output-side controls.

Anthropic charges 4-5x more per output token than per input token, but nothing
upstream of this module targets output. This stage applies three conservative
levers, gated on a turn classification:

  qa     — short user question, no tool work, brevity is fine.
           Cap max_tokens, optionally inject a terseness suffix and stop
           sequences for trailing fluff.
  tool   — user message expects a tool_use response (tools array present, prior
           assistant turn used tools, or message references files/commands).
           Modest max_tokens cap; no terseness — model needs room for code.
  code   — user wants generation/refactor/long output. Untouched.
  unknown — bias toward not breaking things; untouched.

All three sub-controls are independently flag-gated. Caps default on; terseness
and stop sequences default off until measured.
"""
from __future__ import annotations

import os
from typing import Any

from . import PipelineRequest, Stage

TERSENESS_SUFFIX = (
    "\n\nReply concisely. No preamble, no recap, no follow-up offers."
)

DEFAULT_QA_MAX_TOKENS = 800
DEFAULT_TOOL_MAX_TOKENS = 2000

QA = "qa"
TOOL = "tool"
CODE = "code"
UNKNOWN = "unknown"

# Stop sequences for the qa class — patterns that almost always precede pure
# trailing fluff. Conservative: only patterns the model emits *after* the
# substantive answer, on a paragraph boundary. Anthropic supports up to 4 stop
# sequences per request; client-supplied entries are preserved and we never
# add a duplicate.
_QA_STOP_SEQUENCES: tuple[str, ...] = (
    "\n\nLet me know if",
    "\n\nIs there anything",
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _last_user_text(messages: Any) -> str:
    """Extract text of the most recent user message."""
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(blk.get("text") or "")
            return "".join(parts)
        return ""
    return ""


def _last_message_has_tool_result(messages: Any) -> bool:
    """A tool_result in the last user message means the assistant just ran a tool."""
    if not isinstance(messages, list) or not messages:
        return False
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else None
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def _prior_assistant_used_tools(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            return False
        return any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in content
        )
    return False


def classify_turn(body: dict[str, Any]) -> str:
    """Return one of qa | tool | code | unknown.

    Heuristic, conservative — when in doubt return UNKNOWN so the controller
    leaves the request alone.
    """
    if not isinstance(body, dict):
        return UNKNOWN
    messages = body.get("messages")

    # Tool-result follow-up or prior tool_use → mid-agent loop, treat as tool.
    if _last_message_has_tool_result(messages) or _prior_assistant_used_tools(messages):
        return TOOL

    last = _last_user_text(messages)
    last_stripped = last.strip()
    if not last_stripped:
        return UNKNOWN

    chars = len(last_stripped)
    has_tools = bool(body.get("tools"))
    lower = last_stripped.lower()

    code_signals = (
        "implement", "refactor", "write a", "generate", "build a",
        "scaffold", "create a function", "create the", "translate this",
        "port this", "rewrite",
    )
    if any(s in lower for s in code_signals):
        return CODE

    # Code-ish content in the user message itself → leave room.
    if "```" in last_stripped or last_stripped.count("\n") > 30:
        return CODE

    # Short conversational message + no tools → likely qa.
    if not has_tools and chars <= 600 and "?" in last_stripped:
        return QA
    if not has_tools and chars <= 240:
        return QA

    if has_tools:
        return TOOL

    return UNKNOWN


def _existing_stop_sequences(body: dict[str, Any]) -> list[str]:
    raw = body.get("stop_sequences")
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, str)]
    return []


class OutputController(Stage):
    name = "output"

    def __init__(
        self,
        qa_max_tokens: int | None = None,
        tool_max_tokens: int | None = None,
        terseness_enabled: bool | None = None,
        stop_seqs_enabled: bool | None = None,
        caps_enabled: bool | None = None,
    ) -> None:
        self._qa_max_tokens = qa_max_tokens
        self._tool_max_tokens = tool_max_tokens
        self._terseness_enabled = terseness_enabled
        self._stop_seqs_enabled = stop_seqs_enabled
        self._caps_enabled = caps_enabled

    @property
    def qa_max_tokens(self) -> int:
        return self._qa_max_tokens if self._qa_max_tokens is not None else _env_int(
            "TOKENQ_QA_MAX_TOKENS", DEFAULT_QA_MAX_TOKENS
        )

    @property
    def tool_max_tokens(self) -> int:
        return self._tool_max_tokens if self._tool_max_tokens is not None else _env_int(
            "TOKENQ_TOOL_MAX_TOKENS", DEFAULT_TOOL_MAX_TOKENS
        )

    @property
    def caps_enabled(self) -> bool:
        return self._caps_enabled if self._caps_enabled is not None else _env_bool(
            "TOKENQ_OUTPUT_CAPS_ENABLED", True
        )

    @property
    def terseness_enabled(self) -> bool:
        return self._terseness_enabled if self._terseness_enabled is not None else _env_bool(
            "TOKENQ_TERSE_ENABLED", False
        )

    @property
    def stop_seqs_enabled(self) -> bool:
        return self._stop_seqs_enabled if self._stop_seqs_enabled is not None else _env_bool(
            "TOKENQ_STOP_SEQS_ENABLED", False
        )

    async def run(self, req: PipelineRequest):
        body = req.body
        if not isinstance(body, dict):
            return req

        kind = classify_turn(body)
        req.metadata["output_turn_class"] = kind
        if kind == UNKNOWN or kind == CODE:
            return req

        ceiling: int | None = None
        if kind == QA:
            ceiling = self.qa_max_tokens
        elif kind == TOOL:
            ceiling = self.tool_max_tokens

        if ceiling is not None and self.caps_enabled:
            self._cap_max_tokens(req, body, ceiling)

        if kind == QA and self.terseness_enabled:
            self._inject_terseness(body)

        if kind == QA and self.stop_seqs_enabled:
            self._add_stop_sequences(body, _QA_STOP_SEQUENCES)

        return req

    @staticmethod
    def _cap_max_tokens(
        req: PipelineRequest, body: dict[str, Any], ceiling: int
    ) -> None:
        original = body.get("max_tokens")
        if isinstance(original, int) and original > 0 and original <= ceiling:
            return  # already at or below ceiling — leave alone.
        if isinstance(original, int) and original > 0:
            req.metadata["output_cap_original"] = original
        body["max_tokens"] = ceiling
        req.metadata["output_cap_applied"] = ceiling

    @staticmethod
    def _inject_terseness(body: dict[str, Any]) -> None:
        """Append a stable suffix to the system prompt.

        Stable bytes: the same suffix is always added in the same place, so
        Anthropic's prompt cache prefix doesn't churn between requests. Idempotent:
        if the suffix is already present, do nothing.
        """
        sys = body.get("system")
        if isinstance(sys, str):
            if TERSENESS_SUFFIX in sys:
                return
            body["system"] = sys + TERSENESS_SUFFIX
            return
        if isinstance(sys, list):
            for blk in sys:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    text = blk.get("text") or ""
                    if TERSENESS_SUFFIX in text:
                        return
            sys.append({"type": "text", "text": TERSENESS_SUFFIX.lstrip("\n")})
            return
        # No system prompt at all — set one with just the suffix (without leading
        # newlines, since there's nothing to separate from).
        body["system"] = TERSENESS_SUFFIX.lstrip("\n")

    @staticmethod
    def _add_stop_sequences(
        body: dict[str, Any], to_add: tuple[str, ...]
    ) -> None:
        existing = _existing_stop_sequences(body)
        merged = list(existing)
        for seq in to_add:
            if seq in merged:
                continue
            if len(merged) >= 4:  # Anthropic limit
                break
            merged.append(seq)
        if merged != existing:
            body["stop_sequences"] = merged
