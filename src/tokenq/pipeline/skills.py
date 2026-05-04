"""Smart skill loading.

Many clients send a system prompt that enumerates a long library of skills,
each with a paragraph-long description. Most are irrelevant to the current
turn. This stage parses that list, scores each entry against the user's latest
message, keeps only the top K, and replaces the rest with a one-line stub so
the model still knows the names exist.

Scoring is deliberately cheap: token overlap between the query and each skill's
(name + description), plus an explicit override for skills the user invoked by
slash (e.g. `/review`).
"""
from __future__ import annotations

import re
from typing import Any

from ..config import SKILLS_MIN_LIST, SKILLS_TOP_K
from . import PipelineRequest, Stage

# Match "- name: description". The name is non-greedy so the FIRST `: `
# terminates it — descriptions may themselves contain colons.
_SKILL_LINE = re.compile(r"^[ \t]*-[ \t]+([\w][\w:.\-]*?):[ \t]+(.+)$")
_SLASH = re.compile(r"/([\w][\w:\-]*)")
_WORD = re.compile(r"[a-z0-9]+")

# Aggressive stopword list — these dominate any natural-language query but
# carry no signal for skill selection.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do",
    "does", "for", "from", "get", "had", "has", "have", "how", "i", "if",
    "in", "into", "is", "it", "its", "just", "me", "my", "no", "not", "of",
    "on", "or", "our", "out", "should", "so", "some", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "to", "use",
    "want", "was", "we", "were", "what", "when", "where", "which", "will",
    "with", "would", "you", "your",
})


def _tokenize(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}


def _last_user_text(messages: Any) -> str:
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
            return "\n".join(parts)
    return ""


def _explicit_slash_skills(query: str) -> set[str]:
    return {m.lower() for m in _SLASH.findall(query)}


def _score(name: str, description: str, query_terms: set[str]) -> int:
    if not query_terms:
        return 0
    skill_terms = set(_WORD.findall(f"{name} {description}".lower()))
    return len(query_terms & skill_terms)


def _parse_block(text: str) -> tuple[int, int, list[tuple[str, str, str]]] | None:
    """Locate the contiguous skill listing inside `text`.

    A skill bullet may span multiple lines: a `- name: description` line
    followed by zero or more continuation lines (non-blank, not starting with
    `-`). This shape is common in real harnesses where a skill description
    carries TRIGGER/SKIP guidance on its own lines. The block ends at:
    end-of-text, a markdown heading (`#`), a foreign bullet, or a blank line
    followed by a non-continuation line.

    Returns (start_line, end_line, [(name, desc, raw_block), ...]) or None
    when no list is found. `raw_block` carries the full multi-line text so
    rebuild preserves continuations for kept skills. `desc` includes
    continuation lines (joined with spaces) so they contribute to scoring.
    """
    lines = text.split("\n")
    first: int | None = None
    last: int | None = None
    names: list[str] = []
    descs: list[list[str]] = []
    raws: list[list[str]] = []
    saw_blank = False
    for i, line in enumerate(lines):
        m = _SKILL_LINE.match(line)
        if m:
            if first is None:
                first = i
            last = i
            names.append(m.group(1))
            descs.append([m.group(2)])
            raws.append([line])
            saw_blank = False
            continue
        if first is None:
            continue
        stripped = line.strip()
        if stripped == "":
            saw_blank = True
            continue
        if stripped.startswith("#"):
            break
        if line.lstrip().startswith("-"):
            break
        if saw_blank:
            break
        descs[-1].append(stripped)
        raws[-1].append(line)
        last = i
    if first is None or last is None or not names:
        return None
    skills = [
        (name, " ".join(d), "\n".join(r))
        for name, d, r in zip(names, descs, raws)
    ]
    return first, last, skills


def _rebuild(
    lines: list[str], start: int, end: int,
    kept_raw: list[str], trimmed_count: int,
) -> str:
    placeholder = (
        f"- ({trimmed_count} additional skills hidden by tokenq — "
        f"mention by name to load full description)"
    )
    new_lines = lines[:start] + kept_raw
    if trimmed_count > 0:
        new_lines.append(placeholder)
    new_lines += lines[end + 1:]
    return "\n".join(new_lines)


class SkillLoader(Stage):
    name = "skills"

    def __init__(
        self,
        top_k: int = SKILLS_TOP_K,
        min_list: int = SKILLS_MIN_LIST,
    ) -> None:
        self.top_k = top_k
        self.min_list = min_list

    async def run(self, req: PipelineRequest):
        body = req.body
        if not isinstance(body, dict):
            return req

        query = _last_user_text(body.get("messages"))
        query_terms = _tokenize(query)
        explicit = _explicit_slash_skills(query)

        system = body.get("system")
        if isinstance(system, str):
            new_text, saved_chars = self._process(system, query_terms, explicit)
            if new_text is not None:
                body["system"] = new_text
                self._record(req, saved_chars)
            return req

        if isinstance(system, list):
            for blk in system:
                if not isinstance(blk, dict) or blk.get("type") != "text":
                    continue
                text = blk.get("text") or ""
                new_text, saved_chars = self._process(text, query_terms, explicit)
                if new_text is not None:
                    blk["text"] = new_text
                    self._record(req, saved_chars)
        return req

    def _process(
        self, text: str, query_terms: set[str], explicit: set[str],
    ) -> tuple[str | None, int]:
        parsed = _parse_block(text)
        if parsed is None:
            return None, 0
        start, end, skills = parsed
        if len(skills) < self.min_list:
            return None, 0
        if len(skills) <= self.top_k and not explicit:
            return None, 0

        # Always-keep set: skills the user explicitly invoked.
        scored: list[tuple[int, int, tuple[str, str, str]]] = []
        for i, (name, desc, raw) in enumerate(skills):
            if name.lower() in explicit:
                priority = 10**6
            else:
                priority = _score(name, desc, query_terms)
            scored.append((priority, i, (name, desc, raw)))

        # Sort by score desc, original index asc (stable for ties).
        scored.sort(key=lambda t: (-t[0], t[1]))
        top = scored[: max(self.top_k, sum(1 for s, _, _ in scored if s >= 10**6))]
        # Restore original order so any documented sequencing survives.
        top.sort(key=lambda t: t[1])
        kept_raw = [t[2][2] for t in top]
        trimmed_count = len(skills) - len(kept_raw)
        if trimmed_count <= 0:
            return None, 0

        lines = text.split("\n")
        new_text = _rebuild(lines, start, end, kept_raw, trimmed_count)
        saved_chars = len(text) - len(new_text)
        if saved_chars <= 0:
            return None, 0
        return new_text, saved_chars

    def _record(self, req: PipelineRequest, saved_chars: int) -> None:
        saved = saved_chars // 4
        req.metadata["saved_tokens"] = req.metadata.get("saved_tokens", 0) + saved
        req.metadata["saved_by_skills"] = (
            req.metadata.get("saved_by_skills", 0) + saved
        )
