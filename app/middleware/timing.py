# -*- coding: utf-8 -*-
"""
Performance Timing Middleware
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
from app.core.logging import logger


class TimingMiddleware(BaseHTTPMiddleware):
    """Records processing latency and attaches X-Process-Time-Ms header."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Process-Time-Ms"] = str(latency_ms)

        request_id = getattr(request.state, "request_id", "unknown")
        logger.debug(
            f"{request.method} {request.url.path} - status={response.status_code} latency={latency_ms}ms (request_id={request_id})"
        )
        return response

