"""Structured logging for tokenq.

Emits one JSON object per log line on stderr. Use `get_logger(name)` from any
module; the formatter is configured once at process startup via `configure()`.

Honors TOKENQ_LOG_LEVEL (default "info"). When TOKENQ_LOG_FORMAT=text, falls
back to plain text — useful for local debugging.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

from .config import LOG_LEVEL

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
}


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_configured = False


def configure(level: str | None = None) -> None:
    """Install handlers on the root tokenq logger. Idempotent."""
    global _configured
    if _configured:
        return
    _configured = True

    lvl_name = (level or LOG_LEVEL or "info").upper()
    lvl = getattr(logging, lvl_name, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    fmt = os.getenv("TOKENQ_LOG_FORMAT", "json")
    if fmt == "text":
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
        )
    else:
        handler.setFormatter(_JSONFormatter())

    logger = logging.getLogger("tokenq")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(lvl)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the `tokenq` namespace. Auto-configures on first call."""
    if not _configured:
        configure()
    if not name.startswith("tokenq"):
        name = f"tokenq.{name}"
    return logging.getLogger(name)


def now_ms() -> int:
    return int(time.time() * 1000)
