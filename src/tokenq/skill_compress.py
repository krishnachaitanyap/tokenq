"""One-shot SKILL.md compression via the Claude API.

Rewrites a SKILL.md body into a token-optimized form, preserving every
behavioral instruction. Frontmatter is split off and re-attached unchanged.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import ANTHROPIC_API_BASE

DEFAULT_MODEL = "claude-sonnet-4-6"
ANTHROPIC_VERSION = "2023-06-01"
ENDPOINT = "/v1/messages"

SYSTEM_PROMPT = """\
You rewrite SKILL.md files for use as LLM context, producing the most token-\
efficient form that preserves every behavioral instruction. Target 50-70% \
size reduction.

PRESERVE EXACTLY:
- Every directive (must/should/never/don't/always/when X do Y).
- Every conditional, exception, and edge case.
- Every code block, shell snippet, and table that encodes rules — verbatim.
- Every concrete value: paths, URLs, model IDs, version numbers, header names, \
parameter names, regex patterns.
- Every example's behavioral meaning. You may shorten prose around examples; \
do not change inputs or outputs.

CUT:
- Hedging and motivational filler ("you should", "please", "remember to").
- Restatements of the same rule across paragraphs (keep one canonical statement).
- Decorative section headers that don't structure rules.
- Multi-paragraph rationale — keep at most one short "why" clause per rule.
- Background or context sections that don't gate behavior.

OUTPUT:
- Markdown. Keep meaningful headings; drop decorative ones.
- Never invent new rules or change the meaning of existing ones.
- If unsure whether a sentence is load-bearing, KEEP it.
- Output ONLY the rewritten body. No preamble, no commentary, no surrounding \
code fences.
"""

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


def split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter_block, body).

    `frontmatter_block` is empty when the file has no `---`-delimited YAML; it
    is otherwise the full `---\\n...\\n---\\n` prefix so callers can re-prepend
    it verbatim.
    """
    m = _FRONTMATTER.match(text)
    if not m:
        return "", text
    return f"---\n{m.group(1)}\n---\n", m.group(2)


def count_tokens(text: str) -> int:
    """Approximate token count via tiktoken's o200k_base. Falls back to chars/4."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


@dataclass
class CompressionResult:
    path: Path
    before_tokens: int
    after_tokens: int
    saved_tokens: int
    saved_pct: float
    written: bool
    output_path: Path | None


class CompressionError(Exception):
    pass


def _build_payload(body: str, model: str) -> dict[str, Any]:
    user_message = (
        "Compress the following SKILL.md body. Output the rewritten body only.\n\n"
        "<input>\n"
        f"{body}\n"
        "</input>"
    )
    return {
        "model": model,
        "max_tokens": 16000,
        "temperature": 0.0,
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": user_message}],
    }


def _extract_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for blk in response.get("content") or []:
        if isinstance(blk, dict) and blk.get("type") == "text":
            parts.append(blk.get("text") or "")
    return "".join(parts).strip()


def _strip_outer_code_fence(text: str) -> str:
    """If the model wrapped its output in ```...```, peel that off."""
    s = text.strip()
    if not s.startswith("```") or not s.endswith("```"):
        return text
    nl = s.find("\n")
    if nl == -1:
        return text
    return s[nl + 1 : -3].rstrip() + "\n"


def _call_api(
    payload: dict[str, Any], api_key: str, timeout: float, base_url: str,
) -> dict[str, Any]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    try:
        resp = httpx.post(
            f"{base_url.rstrip('/')}{ENDPOINT}",
            headers=headers,
            content=json.dumps(payload).encode("utf-8"),
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        raise CompressionError(f"network error: {e}") from e

    if resp.status_code == 200:
        return resp.json()

    try:
        err = (resp.json().get("error") or {}).get("message") or resp.text
    except Exception:
        err = resp.text

    code = resp.status_code
    if code == 401:
        raise CompressionError("authentication failed: check ANTHROPIC_API_KEY")
    if code == 429:
        retry_after = resp.headers.get("retry-after", "?")
        raise CompressionError(f"rate limited (retry after {retry_after}s): {err}")
    if 500 <= code < 600:
        raise CompressionError(f"server error {code}: {err}")
    raise CompressionError(f"API error {code}: {err}")


def compress_body(
    body: str,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: float = 120.0,
    base_url: str = ANTHROPIC_API_BASE,
) -> str:
    """Compress a SKILL.md body via the Claude API. Returns the rewritten body."""
    api_key = (api_key or os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise CompressionError("ANTHROPIC_API_KEY is not set")

    payload = _build_payload(body, model)
    response = _call_api(payload, api_key, timeout, base_url)
    text = _extract_text(response)
    if not text:
        raise CompressionError("model returned empty output")
    return _strip_outer_code_fence(text)


def compress_file(
    path: Path,
    *,
    model: str = DEFAULT_MODEL,
    output: Path | None = None,
    dry_run: bool = False,
    no_backup: bool = False,
    api_key: str | None = None,
    timeout: float = 120.0,
    base_url: str = ANTHROPIC_API_BASE,
) -> CompressionResult:
    """Compress a SKILL.md file. Frontmatter is preserved unchanged.

    By default writes back in place after copying the original to `<path>.bak`.
    Use `output` to write elsewhere (no backup made), or `dry_run=True` to skip
    the write entirely.
    """
    original = path.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(original)

    compressed_body = compress_body(
        body, model=model, api_key=api_key, timeout=timeout, base_url=base_url,
    )
    new_text = frontmatter + compressed_body
    if not new_text.endswith("\n"):
        new_text += "\n"

    before = count_tokens(original)
    after = count_tokens(new_text)
    saved = before - after
    saved_pct = (saved / before * 100.0) if before else 0.0

    target = output if output is not None else path
    written = False
    if not dry_run:
        if output is None and not no_backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        target.write_text(new_text, encoding="utf-8")
        written = True

    if written:
        try:
            from .storage import log_skill_compression_sync

            log_skill_compression_sync(
                path=str(path),
                output_path=str(target),
                model=model,
                before_tokens=before,
                after_tokens=after,
                saved_tokens=saved,
            )
        except Exception:
            # Logging is best-effort; never fail the rewrite over it.
            pass

    return CompressionResult(
        path=path,
        before_tokens=before,
        after_tokens=after,
        saved_tokens=saved,
        saved_pct=saved_pct,
        written=written,
        output_path=target if written else None,
    )
