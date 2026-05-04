"""Optimization pipeline.

Each Stage transforms a PipelineRequest. A stage may also short-circuit by
returning a PipelineShortCircuit (e.g. cache hit) — the proxy then serves
that response directly without an upstream call.

A Stage may also implement an `after(req, response_data)` hook that runs
post-upstream for non-streaming responses — used by the cache stage to
populate the cache table once the real response is in hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..logging import get_logger

_log = get_logger("pipeline")


@dataclass
class PipelineRequest:
    body: dict[str, Any]
    headers: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineShortCircuit:
    """A stage may return this to bypass the upstream call entirely.

    For non-stream hits, fill `response` with the JSON body to return.
    For stream hits, fill `stream_response` with the raw SSE bytes to replay
    and leave `response` as None.
    """

    response: dict[str, Any] | None = None
    stream_response: bytes | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    saved_tokens: int = 0
    source: str = ""  # e.g. "cache", "rule"


class Stage:
    name: str = ""

    async def run(
        self, req: PipelineRequest
    ) -> PipelineRequest | PipelineShortCircuit:
        return req

    async def after(self, req: PipelineRequest, response: dict[str, Any]) -> None:
        """Optional post-upstream hook for non-stream responses. Default no-op."""
        return None

    async def after_stream(
        self, req: PipelineRequest, raw: bytes, captured: dict[str, Any]
    ) -> None:
        """Optional post-upstream hook for streamed responses. `raw` is the full
        concatenated SSE byte stream; `captured` carries parsed usage and flags
        (input_tokens, output_tokens, stop_reason, tool_use, error). Default no-op.
        """
        return None


class Pipeline:
    def __init__(self, stages: list[Stage] | None = None) -> None:
        self.stages: list[Stage] = stages or []

    async def process(
        self, req: PipelineRequest
    ) -> PipelineRequest | PipelineShortCircuit:
        for stage in self.stages:
            try:
                result = await stage.run(req)
            except Exception:
                # A misbehaving stage must not crash the request. Skip it and
                # continue with the unmutated request — visibility comes from
                # the structured log.
                _log.exception("pipeline_stage_failed", extra={"stage": stage.name})
                continue
            if isinstance(result, PipelineShortCircuit):
                return result
            req = result
        return req

    async def after(self, req: PipelineRequest, response: dict[str, Any]) -> None:
        for stage in reversed(self.stages):
            try:
                await stage.after(req, response)
            except Exception:
                # Best-effort. A misbehaving after-hook must never break the
                # response we already returned to the caller — but we DO need
                # to surface it so silent cache/bandit-stage failures stop
                # eating savings unobserved.
                _log.exception("pipeline_after_failed", extra={"stage": stage.name})

    async def after_stream(
        self, req: PipelineRequest, raw: bytes, captured: dict[str, Any]
    ) -> None:
        for stage in reversed(self.stages):
            try:
                await stage.after_stream(req, raw, captured)
            except Exception:
                _log.exception(
                    "pipeline_after_stream_failed", extra={"stage": stage.name}
                )


def _build_default_pipeline() -> Pipeline:
    from ..bigmemory.inject import BigMemoryInjectStage
    from ..bigmemory.pipeline import BigMemoryStage
    from .bandit import BanditRouter
    from .cache import ExactMatchCache
    from .compaction import TranscriptCompactor
    from .compress import ToolOutputCompressor
    from .dedup import ToolResultDedup
    from .output import OutputController
    from .skills import SkillLoader

    # Bandit runs first so cache keys reflect the routed model. OutputController
    # runs next so max_tokens / stop_sequences / system suffix become part of
    # the cache key. Compaction runs before SkillLoader/cache so they all see
    # the trimmed body.
    #
    # BigMemoryInjectStage is OFF by default (TOKENQ_BIGMEMORY_INJECT=1 to
    # enable). Placed BEFORE ExactMatchCache so the cache key reflects the
    # injected memory block — otherwise a cache hit would serve a response
    # that didn't see the latest profile, which is wrong. The injection's
    # snapshot machinery keeps the block byte-identical within a session
    # window, so cache hits are still possible across consecutive turns.
    #
    # BigMemoryStage runs last (capture-only, no mutation): it observes the
    # final body and stores large tool_results into the FTS5 store, which the
    # inject stage and MCP server then read.
    return Pipeline([
        BanditRouter(),
        OutputController(),
        TranscriptCompactor(),
        SkillLoader(),
        BigMemoryInjectStage(),
        ExactMatchCache(),
        ToolResultDedup(),
        ToolOutputCompressor(),
        BigMemoryStage(),
    ])


_default_pipeline_cached: Pipeline | None = None


def __getattr__(name: str):
    """Lazy: build `default_pipeline` on first access. This avoids a circular
    import — bigmemory.pipeline imports PipelineRequest/Stage from this module,
    so we cannot construct a pipeline that imports BigMemoryStage at this
    module's eval time."""
    global _default_pipeline_cached
    if name == "default_pipeline":
        if _default_pipeline_cached is None:
            _default_pipeline_cached = _build_default_pipeline()
        return _default_pipeline_cached
    raise AttributeError(name)
