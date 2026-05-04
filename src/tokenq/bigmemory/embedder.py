"""Optional semantic-embedding layer for bigmemory.

Wraps `fastembed` (Qdrant's ONNX-based embedder) so the rest of the store
can call `embed()` without caring whether the package is installed. When
fastembed is missing the module behaves as `available=False` and every call
returns None — bigmemory then runs in lexical-only mode (FTS5 / BM25).

Why fastembed: pure-ONNX, no torch dependency, ~50MB footprint, syncronous
API that fits aiosqlite's per-call connection model. Default model
`BAAI/bge-small-en-v1.5` is 384-dimensional, ~30MB on disk, tops MTEB at
that size class. Override via `TOKENQ_EMBED_MODEL`.

Vectors are stored as raw float32 little-endian bytes (1.5KB per row at
384d) — keeps the column compact and avoids any pickle/dtype surprises.
"""
from __future__ import annotations

import math
import os
import struct
import threading
from typing import Iterable

DEFAULT_MODEL = os.getenv("TOKENQ_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384  # bge-small / MiniLM family. Override DIM env if changing models.
EMBED_DIM = int(os.getenv("TOKENQ_EMBED_DIM", str(EMBED_DIM)))

# Off-switch — set TOKENQ_EMBED_ENABLED=0 to force lexical-only mode even when
# fastembed is installed. Used by the test suite to keep iteration fast (the
# ONNX model takes ~80ms/embed and reloads per test due to module-cache resets).
_DISABLED = os.getenv("TOKENQ_EMBED_ENABLED", "1") in ("0", "false", "False", "")


_lock = threading.Lock()
_model = None  # lazily initialized fastembed.TextEmbedding
_load_failed = False


def _load_model():
    """Lazy-load the embedding model. Returns None if fastembed is missing,
    initialization fails, or `TOKENQ_EMBED_ENABLED=0`. Callers MUST handle None."""
    global _model, _load_failed
    if _DISABLED:
        return None
    if _model is not None or _load_failed:
        return _model
    with _lock:
        if _model is not None or _load_failed:
            return _model
        try:
            from fastembed import TextEmbedding  # type: ignore
            _model = TextEmbedding(DEFAULT_MODEL)
        except Exception:
            _load_failed = True
            _model = None
        return _model


def available() -> bool:
    """True iff fastembed loaded successfully. Cheap after first call."""
    return _load_model() is not None


def embed(text: str) -> bytes | None:
    """Return the embedding of `text` as packed float32 LE bytes, or None
    if the embedder isn't available. Empty text returns None."""
    if not text or not text.strip():
        return None
    m = _load_model()
    if m is None:
        return None
    vec = next(iter(m.embed([text])))
    # fastembed returns a numpy array; serialize to raw float32 LE.
    return _pack(list(vec))


def embed_many(texts: list[str]) -> list[bytes | None]:
    """Batched variant — much faster than calling embed() in a loop."""
    m = _load_model()
    if m is None:
        return [None] * len(texts)
    indexed = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    out: list[bytes | None] = [None] * len(texts)
    if not indexed:
        return out
    inputs = [t for _, t in indexed]
    for (i, _), vec in zip(indexed, m.embed(inputs)):
        out[i] = _pack(list(vec))
    return out


def _pack(vec: Iterable[float]) -> bytes:
    arr = list(vec)
    return struct.pack(f"<{len(arr)}f", *arr)


def unpack(blob: bytes) -> list[float]:
    """Inverse of _pack. Reads raw float32 LE bytes back to a Python list."""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def cosine_bytes(a: bytes, b: bytes) -> float:
    """Cosine similarity between two packed-float32-LE blobs.

    Pure-Python (no numpy dep) — for ≤50k rows this is ~80ms/query on a
    modern laptop, fast enough that we don't need an ANN index yet.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    n = len(a) // 4
    av = struct.unpack(f"<{n}f", a)
    bv = struct.unpack(f"<{n}f", b)
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(av, bv):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
