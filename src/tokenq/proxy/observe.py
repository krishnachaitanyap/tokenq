"""Per-request observation helpers.

Pulls three dimensions out of an Anthropic Messages request body so the
dashboard can pivot beyond model/cost:

  - session_id   stable hash of (system, first user message). Same logical
                 conversation gets the same id even after compaction.
  - project      working directory inferred from the system prompt. Claude
                 Code injects 'Primary working directory: <path>'; we surface
                 the basename so the dashboard groups by 'tokenq' rather than
                 the full path. Falls back to '' when undetectable.
  - tools_used   list of tool_use names referenced in this turn's messages.
                 Stored as JSON so a single request can carry multiple tools.

These are deliberately cheap regex-based extractions — the proxy hot path
must not pay an LLM call for each request.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

# Match what Claude Code injects: "Primary working directory: /path/to/repo"
# Some clients use 'cwd:' or 'Working directory:' instead; cover the common shapes.
_CWD_PATTERNS = [
    re.compile(r"Primary working directory:\s*([^\n\r]+)", re.IGNORECASE),
    re.compile(r"Working directory:\s*([^\n\r]+)", re.IGNORECASE),
    re.compile(r"^\s*cwd:\s*([^\n\r]+)", re.IGNORECASE | re.MULTILINE),
]


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for blk in system:
            if isinstance(blk, dict) and blk.get("type") == "text":
                t = blk.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


def _first_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(blk.get("text") or "")
            return "\n".join(parts)
    return ""


def session_id(body: dict[str, Any]) -> str:
    """sha256-hex truncated to 16 chars — short enough for human eyes,
    long enough that collisions don't matter."""
    payload = json.dumps(
        {
            "sys": _system_text(body.get("system")),
            "msg": _first_user_text(body.get("messages")),
        },
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def project_label(body: dict[str, Any]) -> str:
    """Extract the working-directory basename from the system prompt.

    Returns '' when no cwd is detectable (e.g. non-Claude-Code clients). We
    use the basename rather than full path so the same project is grouped
    consistently across machines/users with different home dirs.
    """
    system = _system_text(body.get("system"))
    if not system:
        return ""
    for pat in _CWD_PATTERNS:
        m = pat.search(system)
        if not m:
            continue
        path = m.group(1).strip().strip('"').strip("'")
        if not path:
            continue
        # Use the last non-empty segment as the project name.
        return os.path.basename(path.rstrip("/")) or path
    return ""


def tools_used(body: dict[str, Any]) -> list[str]:
    """Collect every tool_use name referenced in the message thread.

    We capture from the assistant turns (the source of tool_use blocks). The
    list is deduped but preserves first-seen order — useful for "this turn
    used Read then Edit then Bash" reasoning later.
    """
    out: list[str] = []
    seen: set[str] = set()
    messages = body.get("messages")
    if not isinstance(messages, list):
        return out
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            name = block.get("name")
            if not isinstance(name, str) or not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


def turn_index(body: dict[str, Any]) -> int:
    """Approximate turn count = number of user messages in the thread."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
    )


def extract(body: dict[str, Any]) -> dict[str, Any]:
    """One-shot helper — returns the dimensions written to log_request.

    `tools_used` and `edit_files` are JSON-encoded for storage; empty lists
    serialize as empty strings so the dashboard can filter them cheaply with
    a `!= ''` check rather than parsing.

    Activity classification runs here too — codeburn-style deterministic
    rules over tool patterns and the latest user message. See
    proxy.classify.classify_turn for the rule set.
    """
    from . import classify
    tools = tools_used(body)
    bash_text = classify.collect_bash_text(body)
    user_text = classify.latest_user_text(body)
    edit_files = classify.collect_edit_files(body)
    bash_verbs = classify.collect_bash_verbs(body)
    activity = classify.classify_turn(
        tools_used=tools, bash_text=bash_text, user_text=user_text,
    )
    return {
        "session_id": session_id(body),
        "project": project_label(body),
        "tools_used": json.dumps(tools) if tools else "",
        "turn_index": turn_index(body),
        "activity": activity,
        "edit_files": json.dumps(edit_files) if edit_files else "",
        "bash_verbs": json.dumps(bash_verbs) if bash_verbs else "",
    }
