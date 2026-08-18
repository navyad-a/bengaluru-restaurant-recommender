# -*- coding: utf-8 -*-
"""
Request ID Correlation Middleware
"""

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attaches a unique X-Request-ID to every request and response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

