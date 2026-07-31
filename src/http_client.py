"""Centralized HTTP client construction and request retry logic.

This module is the single place in the project for creating
httpx.AsyncClient instances (``create_client``) and for executing HTTP
requests with retries and exponential backoff (``fetch_with_retry``).
Proxy environment variables (ALL_PROXY, HTTPS_PROXY, HTTP_PROXY) are
honored because httpx defaults ``trust_env`` to True.
"""

import asyncio
from typing import Any

import httpx

from src.config import logger

DEFAULT_TIMEOUT: float = 30.0
LLM_TIMEOUT: float = 600.0
DEFAULT_LIMITS: httpx.Limits = httpx.Limits(
    max_connections=50, max_keepalive_connections=20
)


def create_client(
    *,
    timeout: float = DEFAULT_TIMEOUT,
    limits: httpx.Limits = DEFAULT_LIMITS,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    """Create a configured httpx.AsyncClient.

    Args:
        timeout: Request timeout in seconds.
        limits: Connection pool limits.
        base_url: Optional base URL; request paths are resolved against it.
        headers: Optional default headers sent with every request.

    Returns:
        A ready-to-use httpx.AsyncClient.

    Note:
        ``trust_env`` is left at httpx's default of True, so the client
        honors the ALL_PROXY, HTTPS_PROXY, and HTTP_PROXY environment
        variables. SOCKS proxies require the optional ``socksio``
        package; without it, constructing the client raises ImportError.
    """
    client_kwargs: dict[str, Any] = {
        "headers": headers,
        "timeout": timeout,
        "limits": limits,
    }
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    return httpx.AsyncClient(**client_kwargs)


class RetryExhaustedError(RuntimeError):
    """Raised when all retry attempts of fetch_with_retry fail."""


async def fetch_with_retry(
    method: str,
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    headers: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
    timeout: float | None = None,
    retries: int = 5,
    backoff_base: float = 1.0,
) -> httpx.Response:
    """Execute an HTTP request with retries on transient failures.

    Retries on timeouts, transport errors, and any non-2xx response
    (including HTTP 429) with exponential backoff. If no client is
    provided, one is created with the given timeout and closed before
    returning; an injected client is never closed by this function.

    Args:
        method: HTTP method (e.g. "GET", "POST").
        url: Request URL (absolute, or relative to the client's base_url).
        client: Optional pre-configured client. If None, an owned client
            is created and closed automatically.
        headers: Optional request headers.
        json: Optional JSON request body.
        timeout: Optional per-request timeout in seconds; when None, the
            client's default timeout is used. Also used for the owned
            client when *client* is None.
        retries: Maximum number of attempts.
        backoff_base: Base backoff in seconds; the delay before attempt
            *n* is ``backoff_base * 2 ** n``.

    Returns:
        The 2xx response of the successful attempt.

    Raises:
        RetryExhaustedError: If all attempts fail with retryable errors;
            the last error is chained as the cause.
    """
    if client is None:
        client = create_client(timeout=timeout or DEFAULT_TIMEOUT)
        owns_client = True
    else:
        owns_client = False

    last_error: Exception | None = None
    try:
        for attempt in range(retries):
            try:
                if timeout is not None:
                    response = await client.request(
                        method, url, headers=headers, json=json, timeout=timeout
                    )
                else:
                    response = await client.request(
                        method, url, headers=headers, json=json
                    )
                response.raise_for_status()
                return response
            except (
                httpx.TimeoutException,
                httpx.TransportError,
                httpx.HTTPStatusError,
            ) as e:
                if attempt < retries - 1:
                    logger.warning(
                        f"Request {method} {url} attempt {attempt + 1}/{retries} "
                        f"failed: {type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(backoff_base * (2**attempt))
                else:
                    last_error = e
    finally:
        if owns_client:
            await client.aclose()

    raise RetryExhaustedError(
        f"Request {method} {url} failed after {retries} retries: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error
