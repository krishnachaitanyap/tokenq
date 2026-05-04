"""Starlette ASGI app for the proxy. Routes /v1/messages through the pipeline,
forwards everything else under /v1/* as a passthrough.
"""
from __future__ import annotations

import contextlib

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ..logging import configure as configure_logging
from ..logging import get_logger
from ..storage import init_db
from .intercept import handle_messages
from .passthrough import handle_passthrough
from .upstream import close_client

_log = get_logger("proxy.app")


async def health(_: Request) -> JSONResponse:
    """Liveness + readiness check.

    Validates the SQLite DB (read-only ping) and that the upstream httpx client
    can be obtained. We do NOT make a live network call to Anthropic — that
    would tie our health to theirs and create thundering-herd risk on probes.
    Returns 200 when both local checks pass, 503 otherwise. The body always
    carries `checks` so probes can distinguish DB vs upstream failure.
    """
    from ..storage import db_conn
    from .upstream import get_client

    checks: dict[str, str] = {}
    overall_ok = True

    try:
        async with db_conn() as db:
            await db.execute("SELECT 1")
        checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"fail: {type(exc).__name__}"
        overall_ok = False

    try:
        client = get_client()
        if client is None:
            raise RuntimeError("client is None")
        checks["upstream_client"] = "ok"
    except Exception as exc:
        checks["upstream_client"] = f"fail: {type(exc).__name__}"
        overall_ok = False

    return JSONResponse(
        {
            "status": "ok" if overall_ok else "degraded",
            "service": "tokenq",
            "checks": checks,
        },
        status_code=200 if overall_ok else 503,
    )


async def messages_route(request: Request):
    return await handle_messages(request)


async def fallback(request: Request):
    return await handle_passthrough(request)


routes = [
    Route("/healthz", health, methods=["GET"]),
    Route("/v1/messages", messages_route, methods=["POST"]),
    Route("/v1/{path:path}", fallback, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
]


@contextlib.asynccontextmanager
async def lifespan(_: Starlette):
    configure_logging()
    await init_db()
    _log.info("proxy_started")
    try:
        yield
    finally:
        _log.info("proxy_shutting_down")
        await close_client()
        _log.info("proxy_stopped")


app = Starlette(routes=routes, lifespan=lifespan)
