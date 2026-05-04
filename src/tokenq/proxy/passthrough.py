"""Generic passthrough for non-/v1/messages endpoints.

Used for /v1/messages/count_tokens, /v1/models, etc. We don't try to extract
usage info from these — just forward and log status + latency.
"""
from __future__ import annotations

import time

from starlette.requests import Request
from starlette.responses import Response

from ..storage import log_request
from .intercept import filter_headers
from .upstream import get_client


async def handle_passthrough(request: Request) -> Response:
    started = time.monotonic()
    body = await request.body()
    headers = filter_headers(dict(request.headers))
    client = get_client()
    upstream = await client.request(
        request.method,
        request.url.path,
        content=body,
        headers=headers,
        params=request.query_params,
    )
    await log_request(
        ts=time.time(),
        model="",
        latency_ms=int((time.monotonic() - started) * 1000),
        status_code=upstream.status_code,
    )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=filter_headers(dict(upstream.headers)),
        media_type=upstream.headers.get("content-type"),
    )
