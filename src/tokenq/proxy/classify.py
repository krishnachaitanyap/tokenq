"""Per-turn activity classifier — what is the user doing?

Ported from codeburn (getagentseal/codeburn, src/classifier.ts), MIT license.
Deterministic keyword + tool-pattern rules, no LLM call. The result is one of
13 categories that the dashboard pivots on so the user can see "60% of my
spend this week was debugging."

# Priority order (matters when multiple categories match)

Tool patterns first, then keyword refinement. Within tool patterns:
  bash → testing > git > build > install
  edit → coding (refined by debug/refactor/feature keywords)
  read-only → exploration
  task tools → planning
  agent → delegation
  no tools → conversation/brainstorming/general by user-text keywords

The priority order is the codeburn original — Testing wins over Coding when
both match because someone running `pytest` while the model also Edits a
fixture file is *primarily* testing, not coding.

# Inputs

- `tools_used`: deduped list of tool_use names from this turn's messages
  (already extracted by observe.tools_used).
- `bash_text`: concatenated content of bash command tool inputs. Used for
  testing/git/build keyword detection.
- `user_text`: the latest user message text. Used for debug/feature/refactor
  keyword refinement.
"""
from __future__ import annotations

import re
from typing import Any

# Tool name sets — match Claude Code, Cursor, and a few generic shapes.
EDIT_TOOLS = {
    "Edit", "Write", "MultiEdit", "FileEditTool", "FileWriteTool",
    "NotebookEdit", "cursor:edit",
}
READ_TOOLS = {
    "Read", "Grep", "Glob", "FileReadTool", "GrepTool", "GlobTool", "LS",
}
BASH_TOOLS = {"Bash", "BashTool", "PowerShellTool"}
TASK_TOOLS = {
    "TaskCreate", "TaskUpdate", "TaskGet", "TaskList", "TaskOutput",
    "TaskStop", "TodoWrite", "EnterPlanMode", "ExitPlanMode",
}
AGENT_TOOLS = {"Agent", "Task"}  # Task in some Anthropic tool catalogs
SEARCH_TOOLS = {"WebSearch", "WebFetch", "ToolSearch"}

# Bash command patterns. Order matters — testing checked before git etc.
_TESTING_RE = re.compile(
    r"\b(test|pytest|vitest|jest|mocha|spec|coverage|"
    r"npm\s+test|npx\s+vitest|npx\s+jest|go\s+test|cargo\s+test)\b",
    re.IGNORECASE,
)
_GIT_RE = re.compile(
    r"\bgit\s+(push|pull|commit|merge|rebase|checkout|branch|stash|"
    r"log|diff|status|add|reset|cherry-pick|tag)\b",
    re.IGNORECASE,
)
_BUILD_RE = re.compile(
    r"\b(npm\s+run\s+build|npm\s+publish|docker|deploy|make\s+build|"
    r"npm\s+run\s+dev|npm\s+start|pm2|systemctl|brew\s+services|"
    r"cargo\s+build|cargo\s+run)\b",
    re.IGNORECASE,
)
_INSTALL_RE = re.compile(
    r"\b(npm\s+install|pip\s+install|brew\s+install|apt\s+install|"
    r"cargo\s+add|yarn\s+add|poetry\s+add)\b",
    re.IGNORECASE,
)

# User-message keyword patterns (lower-cased text).
_DEBUG_RE = re.compile(
    r"\b(fix|bug|error|broken|failing|crash|issue|debug|traceback|"
    r"exception|stack\s+trace|not\s+working|wrong|unexpected|"
    r"status\s+code|404|500|401|403)\b",
    re.IGNORECASE,
)
_FEATURE_RE = re.compile(
    r"\b(add|create|implement|new|build|feature|introduce|set\s+up|"
    r"scaffold|generate)\b",
    re.IGNORECASE,
)
_REFACTOR_RE = re.compile(
    r"\b(refactor|clean\s+up|rename|reorganize|simplify|extract|"
    r"restructure|migrate|split)\b",
    re.IGNORECASE,
)
_BRAINSTORM_RE = re.compile(
    r"\b(brainstorm|idea|what\s+if|explore|think\s+about|approach|"
    r"strategy|design|consider)\b",
    re.IGNORECASE,
)
_EXPLORE_RE = re.compile(
    r"\b(research|investigate|look\s+into|find\s+out|search|analyze|"
    r"review|understand|explain)\b",
    re.IGNORECASE,
)


# All thirteen labels. Stable strings — the dashboard groups on these.
ACTIVITIES = (
    "coding", "debugging", "feature_dev", "refactoring", "testing",
    "exploration", "planning", "delegation", "git_ops", "build_deploy",
    "brainstorming", "conversation", "general",
)


def classify_turn(
    *,
    tools_used: list[str],
    bash_text: str = "",
    user_text: str = "",
) -> str:
    """Return one of ACTIVITIES. Pure function — same inputs always yield
    the same label, no global state, no I/O.
    """
    tool_set = set(tools_used or [])

    # No tools used → it's a pure conversation. Refine by intent keywords.
    if not tool_set:
        return _classify_conversation(user_text)

    # Tool-pattern phase. Bash patterns first (codeburn priority).
    if tool_set & BASH_TOOLS and bash_text:
        if _TESTING_RE.search(bash_text):
            return "testing"
        if _GIT_RE.search(bash_text):
            return "git_ops"
        if _BUILD_RE.search(bash_text):
            return "build_deploy"
        if _INSTALL_RE.search(bash_text):
            return "build_deploy"  # codeburn folds install under build/deploy

    # Edit/Write present → coding, then refine by user intent.
    if tool_set & EDIT_TOOLS:
        return _refine_coding(user_text)

    # Task management tools → planning. Checked before delegation/exploration
    # because a TodoWrite call while also reading is still planning.
    if tool_set & TASK_TOOLS:
        return "planning"

    # Agent spawn → delegation.
    if tool_set & AGENT_TOOLS:
        return "delegation"

    # Search/research tools.
    if tool_set & SEARCH_TOOLS:
        return "exploration"

    # Bash without an identified pattern, or just reads → exploration.
    if (tool_set & BASH_TOOLS) or (tool_set & READ_TOOLS):
        return "exploration"

    return "general"


def _refine_coding(user_text: str) -> str:
    """When edits were made, the *intent* tells us the flavor."""
    text = user_text or ""
    if _DEBUG_RE.search(text):
        return "debugging"
    if _REFACTOR_RE.search(text):
        return "refactoring"
    if _FEATURE_RE.search(text):
        return "feature_dev"
    return "coding"


def _classify_conversation(user_text: str) -> str:
    text = user_text or ""
    if _BRAINSTORM_RE.search(text):
        return "brainstorming"
    if _EXPLORE_RE.search(text):
        return "exploration"
    if _DEBUG_RE.search(text):
        return "debugging"
    return "conversation"


def collect_bash_text(body: dict[str, Any], *, max_chars: int = 4000) -> str:
    """Concatenate every bash tool_use's `command` input from the message
    thread — that's the haystack the testing/git/build patterns search.

    Truncated at max_chars to bound work on long sessions; testing/git/build
    keywords are usually right at the start of the command anyway.
    """
    out: list[str] = []
    total = 0
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("name") not in BASH_TOOLS:
                continue
            inp = block.get("input")
            if not isinstance(inp, dict):
                continue
            cmd = inp.get("command")
            if isinstance(cmd, str):
                out.append(cmd)
                total += len(cmd)
                if total >= max_chars:
                    return "\n".join(out)[:max_chars]
    return "\n".join(out)


def collect_bash_verbs(body: dict[str, Any], *, limit: int = 20) -> list[str]:
    """Extract the leading verb from each Bash tool_use input.

    The "verb" is the first whitespace-separated token of the command,
    minus its path prefix — so `/usr/bin/git push` becomes `git`. Common
    shell prologues (`sudo`, `time`, `cd && ...`) are stripped so the
    semantic command (the second token after them) wins.

    Why basename rather than the full command: the dashboard groups by
    verb to show "ran git 672x, npm 50x" — keeping `/usr/local/bin/git`
    distinct from `git` would just balkanize the same tool.
    """
    SHELL_PRELUDE = {"sudo", "time", "nice", "env", "exec"}
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
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("name") not in BASH_TOOLS:
                continue
            inp = block.get("input")
            if not isinstance(inp, dict):
                continue
            cmd = inp.get("command")
            if not isinstance(cmd, str) or not cmd.strip():
                continue
            # Take the first command on the line (split on shell separators).
            head = re.split(r"[;&|]", cmd, maxsplit=1)[0].strip()
            tokens = head.split()
            # Strip leading env-style prefix (FOO=bar python ...).
            while tokens and "=" in tokens[0] and not tokens[0].startswith("="):
                tokens = tokens[1:]
            # Strip shell-prelude verbs (sudo, time, etc.).
            while tokens and tokens[0] in SHELL_PRELUDE:
                tokens = tokens[1:]
            if not tokens:
                continue
            verb = tokens[0].rsplit("/", 1)[-1]  # basename
            if not verb or verb in seen:
                continue
            seen.add(verb)
            out.append(verb)
            if len(out) >= limit:
                return out
    return out


def collect_edit_files(body: dict[str, Any], *, limit: int = 20) -> list[str]:
    """Return the list of file paths edited/written in this turn — feeds the
    one-shot retry detector. Deduped, first-seen order.
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
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("name") not in EDIT_TOOLS:
                continue
            inp = block.get("input")
            if not isinstance(inp, dict):
                continue
            for k in ("file_path", "path", "filename"):
                v = inp.get(k)
                if isinstance(v, str) and v and v not in seen:
                    seen.add(v)
                    out.append(v)
                    if len(out) >= limit:
                        return out
                    break
    return out


def latest_user_text(body: dict[str, Any], *, max_chars: int = 1000) -> str:
    """Return the most recent user message's text content, truncated. Used
    by the classifier and stored on the request row for post-hoc retraining
    of the keyword regexes."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content[:max_chars]
        if isinstance(content, list):
            parts = []
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(blk.get("text") or "")
            return ("\n".join(parts))[:max_chars]
    return ""
