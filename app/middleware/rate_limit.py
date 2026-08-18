# -*- coding: utf-8 -*-
"""
Lightweight In-Memory Sliding-Window Rate Limiter
"""

import time
import threading
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from app.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window rate limiter per client IP."""

    def __init__(self, app):
        super().__init__(app)
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
        self.window_seconds = 60
        self.max_requests = settings.RATE_LIMIT_REQUESTS_PER_MINUTE

    async def dispatch(self, request: Request, call_next) -> Response:
        if not settings.RATE_LIMIT_ENABLED or request.url.path in ["/health", "/ready", "/docs", "/openapi.json"]:
            return await call_next(request)

        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()

        with self.lock:
            # Purge timestamps older than 60s
            self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < self.window_seconds]
            if len(self.requests[client_ip]) >= self.max_requests:
                request_id = getattr(request.state, "request_id", "unknown")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "status": "error",
                        "error_code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Maximum {self.max_requests} requests per minute allowed.",
                        "request_id": request_id
                    },
                    headers={"Retry-After": "60"}
                )
            self.requests[client_ip].append(now)

        return await call_next(request)

