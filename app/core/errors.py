# -*- coding: utf-8 -*-
"""
Standardized API Error Handling and Formatting
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.logging import logger


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Formats standard HTTPExceptions with structured JSON response."""
    request_id = getattr(request.state, "request_id", "unknown")
    error_code = "HTTP_ERROR"
    if exc.status_code == status.HTTP_404_NOT_FOUND:
        error_code = "RESOURCE_NOT_FOUND"
    elif exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        error_code = "SERVICE_UNAVAILABLE"
    elif exc.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY:
        error_code = "VALIDATION_ERROR"
    elif exc.status_code == status.HTTP_400_BAD_REQUEST:
        error_code = "BAD_REQUEST"

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": error_code,
            "message": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
            "detail": exc.detail,
            "request_id": request_id,
            "path": request.url.path
        },
        headers=getattr(exc, "headers", None)
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Formats Pydantic request validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    errors = exc.errors()
    simplified_errors = []
    for err in errors:
        loc = " -> ".join([str(l) for l in err.get("loc", [])])
        simplified_errors.append(f"{loc}: {err.get('msg')}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_code": "VALIDATION_ERROR",
            "message": "Request body or query parameter validation failed.",
            "request_id": request_id,
            "path": request.url.path,
            "details": simplified_errors
        }
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catches unhandled exceptions and returns a sanitized 500 error response."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unhandled exception on {request.method} {request.url.path} (request_id={request_id}): {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred while processing your recommendation request.",
            "request_id": request_id,
            "path": request.url.path
        }
    )

