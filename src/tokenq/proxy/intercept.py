"""POST /v1/messages handler.

Two paths: streaming (SSE passthrough) and non-streaming (JSON). Both extract
usage info to log into SQLite, then forward upstream verbatim. The pipeline is
applied here so future stages (cache, dedup, compress, route, compile) plug in
at one place.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from ..logging import get_logger
from ..pipeline import Pipeline, PipelineRequest, PipelineShortCircuit, default_pipeline
from ..pricing import estimate_cost
from ..storage import log_request
from .observe import extract as observe_extract
from .upstream import get_client, reset_client

_log = get_logger("proxy.intercept")

# Hop-by-hop headers and ones we should never blindly forward.
HOP_BY_HOP = {
    "host",
    "content-length",
    "connection",
    "transfer-encoding",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
}


def filter_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


# httpx exception classes that signal a transport-level failure where the
# connection pool may be holding a broken socket. Recovered by resetting the
# shared client. Distinguished from logical 4xx/5xx upstream responses, which
# come back as a normal Response object and should be forwarded verbatim.
_TRANSPORT_FAILURES = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.PoolTimeout,
)


async def _upstream_failure_response(
    *,
    exc: Exception,
    request_id: str,
    model: str,
    started: float,
    body_bytes: bytes,
    saved_tokens: int,
    saved_by_dedup: int,
    saved_by_compress: int,
    saved_by_skills: int,
    is_stream: bool,
) -> Response:
    """Convert a transport-level upstream failure into a clean 502 to the
    client and log the attempt so the dashboard sees the failure."""
    latency_ms = int((time.monotonic() - started) * 1000)
    err = f"{type(exc).__name__}: {exc}"[:500]
    _log.error(
        "upstream_transport_failure",
        extra={
            "request_id": request_id,
            "model": model,
            "latency_ms": latency_ms,
            "error": err,
        },
    )
    # Drop the connection pool so subsequent requests don't reuse a broken socket.
    await reset_client()

    try:
        await log_request(
            ts=time.time(),
            model=model,
            input_tokens=0,
            output_tokens=0,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            saved_tokens=saved_tokens,
            saved_by_dedup=saved_by_dedup,
            saved_by_compress=saved_by_compress,
            saved_by_skills=saved_by_skills,
            latency_ms=latency_ms,
            status_code=502,
            error=err,
            stream=1 if is_stream else 0,
            estimated_cost_usd=0.0,
        )
    except Exception:
        _log.exception("log_request_failed", extra={"request_id": request_id})

    return JSONResponse(
        {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": f"Upstream request failed: {err}",
            },
        },
        status_code=502,
    )


async def handle_messages(request: Request, pipeline: Pipeline | None = None) -> Response:
    pipeline = pipeline or default_pipeline
    started = time.monotonic()
    request_id = uuid.uuid4().hex[:12]
    body_bytes = await request.body()

    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        # Malformed JSON — return 400 explicitly rather than silently treating
        # it as an empty request and forwarding nonsense upstream.
        _log.warning("malformed_request_body", extra={"request_id": request_id})
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Request body is not valid JSON.",
                },
            },
            status_code=400,
        )

    headers = filter_headers(dict(request.headers))
    # Force identity encoding upstream so the SSE parser can read plain bytes.
    # httpx auto-decodes JSON responses regardless, so this only matters for the
    # streaming path, but applying it uniformly keeps both paths symmetric.
    headers["accept-encoding"] = "identity"
    pipe_req = PipelineRequest(body=body, headers=headers)
    pipe_req.metadata["request_id"] = request_id
    # Observe session/project/tools BEFORE the pipeline runs — pipeline stages
    # may rewrite system or tool_use blocks (e.g. bigmemory inject), but the
    # session identity should reflect what the user actually sent.
    pipe_req.metadata["observe"] = observe_extract(body)

    try:
        result = await pipeline.process(pipe_req)
    except Exception:
        _log.exception(
            "pipeline_process_failed",
            extra={"request_id": request_id, "model": body.get("model", "")},
        )
        # Pipeline failure must not drop the request. Fall back to forwarding
        # the original body upstream untouched so the user still gets a real
        # response — they just lose this request's optimization opportunity.
        result = pipe_req

    if isinstance(result, PipelineShortCircuit):
        # Cache hit (or other early-return). Serve directly without upstream call.
        is_stream_req = bool(body.get("stream", False))
        if isinstance(result.response, dict):
            usage = result.response.get("usage", {}) or {}
        else:
            usage = {}
        input_tokens = result.input_tokens or usage.get("input_tokens", 0)
        output_tokens = result.output_tokens or usage.get("output_tokens", 0)

        obs = pipe_req.metadata.get("observe") or {}
        await log_request(
            ts=time.time(),
            model=body.get("model", ""),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_locally=1,
            saved_tokens=result.saved_tokens,
            saved_by_cache=result.saved_tokens,
            latency_ms=int((time.monotonic() - started) * 1000),
            status_code=200,
            stream=int(is_stream_req),
            estimated_cost_usd=0.0,
            **obs,
        )

        if result.stream_response is not None:
            payload = result.stream_response

            async def replay():
                yield payload

            return StreamingResponse(
                replay(),
                status_code=200,
                media_type="text/event-stream",
            )

        return Response(
            content=json.dumps(result.response or {}).encode("utf-8"),
            status_code=200,
            media_type="application/json",
        )

    # Pipeline returned a (possibly mutated) request. Forward upstream.
    new_body_bytes = json.dumps(result.body).encode("utf-8")
    new_headers = result.headers
    # Refresh content-length implicitly by letting httpx set it.
    new_headers.pop("content-length", None)

    is_stream = bool(result.body.get("stream", False))
    model = result.body.get("model", "")
    saved_tokens = int(result.metadata.get("saved_tokens", 0))
    saved_by_dedup = int(result.metadata.get("saved_by_dedup", 0))
    saved_by_compress = int(result.metadata.get("saved_by_compress", 0))
    saved_by_skills = int(result.metadata.get("saved_by_skills", 0))

    if is_stream:
        return await _stream(
            request=request,
            body_bytes=new_body_bytes,
            headers=new_headers,
            model=model,
            started=started,
            request_id=request_id,
            saved_tokens=saved_tokens,
            saved_by_dedup=saved_by_dedup,
            saved_by_compress=saved_by_compress,
            saved_by_skills=saved_by_skills,
            pipeline=pipeline,
            pipe_req=result,
        )
    return await _non_stream(
        request=request,
        body_bytes=new_body_bytes,
        headers=new_headers,
        model=model,
        started=started,
        request_id=request_id,
        pipeline=pipeline,
        pipe_req=result,
        saved_tokens=saved_tokens,
        saved_by_dedup=saved_by_dedup,
        saved_by_compress=saved_by_compress,
        saved_by_skills=saved_by_skills,
    )


async def _non_stream(
    request: Request,
    body_bytes: bytes,
    headers: dict[str, str],
    model: str,
    started: float,
    request_id: str = "",
    pipeline: Pipeline | None = None,
    pipe_req: PipelineRequest | None = None,
    saved_tokens: int = 0,
    saved_by_dedup: int = 0,
    saved_by_compress: int = 0,
    saved_by_skills: int = 0,
) -> Response:
    client = get_client()
    try:
        upstream = await client.request(
            request.method,
            request.url.path,
            content=body_bytes,
            headers=headers,
            params=request.query_params,
        )
    except _TRANSPORT_FAILURES as exc:
        return await _upstream_failure_response(
            exc=exc,
            request_id=request_id,
            model=model,
            started=started,
            body_bytes=body_bytes,
            saved_tokens=saved_tokens,
            saved_by_dedup=saved_by_dedup,
            saved_by_compress=saved_by_compress,
            saved_by_skills=saved_by_skills,
            is_stream=False,
        )
    latency_ms = int((time.monotonic() - started) * 1000)

    input_tokens = output_tokens = cache_creation = cache_read = 0
    error: str | None = None
    data: dict[str, Any] | None = None
    try:
        data = upstream.json()
        usage = data.get("usage", {}) or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        if upstream.status_code >= 400:
            error = (data.get("error") or {}).get("message") or str(data)[:500]
    except Exception:
        if upstream.status_code >= 400:
            error = upstream.text[:500]

    if (
        pipeline is not None
        and pipe_req is not None
        and upstream.status_code == 200
        and isinstance(data, dict)
    ):
        try:
            await pipeline.after(pipe_req, data)
        except Exception:
            _log.exception("after_failed", extra={"request_id": request_id})

    saved_by_bandit_usd = (
        float(pipe_req.metadata.get("saved_by_bandit_usd", 0.0))
        if pipe_req is not None
        else 0.0
    )
    obs = (pipe_req.metadata.get("observe") if pipe_req is not None else None) or {}
    try:
        await log_request(
            ts=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            saved_tokens=saved_tokens,
            saved_by_dedup=saved_by_dedup,
            saved_by_compress=saved_by_compress,
            saved_by_skills=saved_by_skills,
            saved_by_bandit_usd=saved_by_bandit_usd,
            latency_ms=latency_ms,
            status_code=upstream.status_code,
            error=error,
            stream=0,
            estimated_cost_usd=estimate_cost(
                model, input_tokens, output_tokens, cache_creation, cache_read
            ),
            **obs,
        )
    except Exception:
        _log.exception("log_request_failed", extra={"request_id": request_id})

    resp_headers = filter_headers(dict(upstream.headers))
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
    )


async def _stream(
    request: Request,
    body_bytes: bytes,
    headers: dict[str, str],
    model: str,
    started: float,
    request_id: str = "",
    saved_tokens: int = 0,
    saved_by_dedup: int = 0,
    saved_by_compress: int = 0,
    saved_by_skills: int = 0,
    pipeline: Pipeline | None = None,
    pipe_req: PipelineRequest | None = None,
) -> Response:
    """Streaming SSE passthrough.

    Yield raw bytes as they arrive (no buffering of the response to the client).
    Concurrently buffer the bytes for the cache stage and parse SSE events to
    capture usage / stop_reason / tool_use so we can both log accurate token
    counts and decide whether to populate the cache.
    """
    client = get_client()
    req = client.build_request(
        request.method,
        request.url.path,
        content=body_bytes,
        headers=headers,
        params=request.query_params,
    )
    try:
        upstream = await client.send(req, stream=True)
    except _TRANSPORT_FAILURES as exc:
        return await _upstream_failure_response(
            exc=exc,
            request_id=request_id,
            model=model,
            started=started,
            body_bytes=body_bytes,
            saved_tokens=saved_tokens,
            saved_by_dedup=saved_by_dedup,
            saved_by_compress=saved_by_compress,
            saved_by_skills=saved_by_skills,
            is_stream=True,
        )

    captured: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation": 0,
        "cache_read": 0,
        "stop_reason": None,
        "tool_use": False,
        "error": None,
    }

    async def passthrough():
        residual = b""
        full_buf = bytearray()
        stream_completed = False
        try:
            async for chunk in upstream.aiter_raw():
                # Forward immediately — never buffer the response to the client.
                yield chunk
                # Concurrently parse for usage info, frame-by-frame, and
                # accumulate for the cache stage.
                full_buf.extend(chunk)
                residual += chunk
                while b"\n\n" in residual:
                    event, residual = residual.split(b"\n\n", 1)
                    _capture_usage(event, captured)
            stream_completed = True
        except _TRANSPORT_FAILURES as exc:
            # Mid-stream transport failure. The client has already received
            # partial bytes; we can't synthesize a 502 now. Emit an explicit
            # SSE error frame so consumers see the failure rather than a
            # silent truncation, log it, and let the finally block log the
            # request and close the upstream.
            err = f"{type(exc).__name__}: {exc}"[:500]
            captured["error"] = err
            _log.error(
                "upstream_stream_interrupted",
                extra={"request_id": request_id, "model": model, "error": err},
            )
            try:
                yield (
                    b'event: error\ndata: {"type":"error","error":'
                    b'{"type":"api_error","message":"upstream stream interrupted"}}\n\n'
                )
            except Exception:
                pass
        except Exception:
            _log.exception(
                "stream_passthrough_failed",
                extra={"request_id": request_id, "model": model},
            )
            captured["error"] = "stream_passthrough_failed"
        finally:
            try:
                await upstream.aclose()
            except Exception:
                _log.exception(
                    "upstream_aclose_failed",
                    extra={"request_id": request_id},
                )
            latency_ms = int((time.monotonic() - started) * 1000)
            if (
                stream_completed
                and pipeline is not None
                and pipe_req is not None
                and upstream.status_code == 200
                and not captured["error"]
            ):
                # after_stream populates the cache and lets stages (e.g. bandit)
                # record post-hoc savings into pipe_req.metadata. Run it before
                # log_request so those savings land on the same row. Only run
                # when the stream completed cleanly — caching a truncated
                # response would poison subsequent identical requests.
                try:
                    await pipeline.after_stream(pipe_req, bytes(full_buf), captured)
                except Exception:
                    _log.exception(
                        "after_stream_failed", extra={"request_id": request_id}
                    )
            saved_by_bandit_usd = (
                float(pipe_req.metadata.get("saved_by_bandit_usd", 0.0))
                if pipe_req is not None
                else 0.0
            )
            obs = (
                pipe_req.metadata.get("observe") if pipe_req is not None else None
            ) or {}
            try:
                await log_request(
                    ts=time.time(),
                    model=model,
                    input_tokens=captured["input_tokens"],
                    output_tokens=captured["output_tokens"],
                    cache_creation_tokens=captured["cache_creation"],
                    cache_read_tokens=captured["cache_read"],
                    saved_tokens=saved_tokens,
                    saved_by_dedup=saved_by_dedup,
                    saved_by_compress=saved_by_compress,
                    saved_by_skills=saved_by_skills,
                    saved_by_bandit_usd=saved_by_bandit_usd,
                    latency_ms=latency_ms,
                    status_code=upstream.status_code,
                    error=captured["error"],
                    stream=1,
                    estimated_cost_usd=estimate_cost(
                        model,
                        captured["input_tokens"],
                        captured["output_tokens"],
                        captured["cache_creation"],
                        captured["cache_read"],
                    ),
                    **obs,
                )
            except Exception:
                _log.exception(
                    "log_request_failed", extra={"request_id": request_id}
                )

    resp_headers = filter_headers(dict(upstream.headers))
    return StreamingResponse(
        passthrough(),
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
    )


def _capture_usage(event_bytes: bytes, captured: dict[str, Any]) -> None:
    """Parse an SSE event frame and update the captured usage dict in-place."""
    try:
        text = event_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return
    for line in text.split("\n"):
        if not line.startswith("data:"):
            continue
        payload = line[5:].lstrip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        # message_start emits the initial usage on `data.message.usage`
        # message_delta emits incremental usage on `data.usage`
        usage = (data.get("message") or {}).get("usage") or data.get("usage") or {}
        if "input_tokens" in usage:
            captured["input_tokens"] = usage["input_tokens"]
        if "output_tokens" in usage:
            captured["output_tokens"] = usage["output_tokens"]
        if "cache_creation_input_tokens" in usage:
            captured["cache_creation"] = usage["cache_creation_input_tokens"]
        if "cache_read_input_tokens" in usage:
            captured["cache_read"] = usage["cache_read_input_tokens"]

        evt = data.get("type")
        if evt == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                captured["tool_use"] = True
        elif evt == "message_delta":
            stop = (data.get("delta") or {}).get("stop_reason")
            if stop:
                captured["stop_reason"] = stop
        elif evt == "error":
            err = (data.get("error") or {}).get("message")
            if err:
                captured["error"] = err
