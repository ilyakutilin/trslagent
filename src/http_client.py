"""Centralized HTTP client construction and request retry logic.

This module is the single place in the project for creating
httpx.AsyncClient instances (``create_client``) and for executing HTTP
requests with retries and exponential backoff (``fetch_with_retry``).
Proxy behavior is resolved by ``resolve_proxy_kwargs`` from the
``[proxy]`` settings: explicit proxy configuration takes precedence over
proxy environment variables (ALL_PROXY, HTTPS_PROXY, HTTP_PROXY), a
disabled proxy forces direct connections, and otherwise the environment
variables are honored (httpx ``trust_env=True``).
"""

import asyncio
import os
from typing import Any
from urllib.parse import quote

import httpx

from src.config import ProxySettings, logger

DEFAULT_TIMEOUT: float = 30.0
LLM_TIMEOUT: float = 600.0
DEFAULT_LIMITS: httpx.Limits = httpx.Limits(
    max_connections=50, max_keepalive_connections=20
)

SUPPORTED_PROXY_PROTOCOLS: tuple[str, ...] = (
    "http",
    "https",
    "socks5",
    "socks5h",
    "socks4",
    "socks4a",
)
PROXY_ENV_VARS: tuple[str, ...] = (
    "ALL_PROXY",
    "all_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
)


class ProxyConfigError(ValueError):
    """Raised when proxy settings cannot be resolved into client kwargs."""


def resolve_proxy_kwargs(proxy_settings: ProxySettings | None) -> dict[str, Any]:
    """Resolve proxy settings into httpx.AsyncClient constructor kwargs.

    Applies the following resolution rules:

    1. ``proxy_settings is None`` (caller didn't provide) — legacy
       behavior: no kwargs are returned, so httpx defaults to
       ``trust_env=True`` (proxy env vars honored).
    2. ``enabled`` is False — ``{"trust_env": False}`` (direct
       connections, env vars ignored).
    3. ``protocol`` is set (non-empty after strip) — validated against
       ``SUPPORTED_PROXY_PROTOCOLS``, then a proxy URL is built as
       ``{protocol}://[{quote(user)}[:{quote(password)}]@]{host}:{port}``
       and ``{"proxy": url, "trust_env": False}`` is returned (explicit
       config takes precedence over env vars).
    4. ``protocol`` unset — any of ``PROXY_ENV_VARS`` set and non-empty
       after strip yields ``{"trust_env": True}``; otherwise
       ``ProxyConfigError`` is raised.

    Args:
        proxy_settings: Optional proxy settings; None triggers the legacy
            behavior of rule 1.

    Returns:
        A dict of keyword arguments to merge into the client constructor
        call.

    Raises:
        ProxyConfigError: If the protocol is unsupported or no proxy
            configuration can be established (rule 4).
    """
    if proxy_settings is None:
        return {}
    if not proxy_settings.enabled:
        logger.debug("Proxy disabled — requests go direct")
        return {"trust_env": False}
    protocol = proxy_settings.protocol.strip() if proxy_settings.protocol else ""
    if protocol:
        if protocol not in SUPPORTED_PROXY_PROTOCOLS:
            raise ProxyConfigError(
                f"Unsupported proxy protocol '{protocol}'. "
                f"Supported protocols: {', '.join(SUPPORTED_PROXY_PROTOCOLS)}"
            )
        userinfo = ""
        if proxy_settings.username:
            userinfo = quote(proxy_settings.username)
            if proxy_settings.password is not None:
                userinfo += f":{quote(proxy_settings.password.get_secret_value())}"
        userinfo = f"{userinfo}@" if userinfo else ""
        proxy_url = (
            f"{protocol}://{userinfo}{proxy_settings.host}:{proxy_settings.port}"
        )
        logger.info(
            "Proxy: using configured {} proxy at {}:{}",
            protocol,
            proxy_settings.host,
            proxy_settings.port,
        )
        return {"proxy": proxy_url, "trust_env": False}
    if any(os.getenv(var, "").strip() for var in PROXY_ENV_VARS):
        logger.info("Proxy: using proxy settings from environment variables")
        return {"trust_env": True}
    raise ProxyConfigError(
        "No proxy configured: set [proxy] settings "
        "(PROXY__PROTOCOL/PROXY__HOST/PROXY__PORT), or export "
        "ALL_PROXY/HTTPS_PROXY/HTTP_PROXY env vars, or set "
        "PROXY__ENABLED=false for direct connections."
    )


def create_client(
    *,
    timeout: float = DEFAULT_TIMEOUT,
    limits: httpx.Limits = DEFAULT_LIMITS,
    base_url: str | None = None,
    headers: dict[str, str] | None = None,
    proxy_settings: ProxySettings | None = None,
) -> httpx.AsyncClient:
    """Create a configured httpx.AsyncClient.

    Args:
        timeout: Request timeout in seconds.
        limits: Connection pool limits.
        base_url: Optional base URL; request paths are resolved against it.
        headers: Optional default headers sent with every request.
        proxy_settings: Optional proxy settings resolved via
            :func:`resolve_proxy_kwargs`; when None, the legacy behavior
            is kept (httpx default ``trust_env=True``).

    Returns:
        A ready-to-use httpx.AsyncClient.

    Note:
        Proxy behavior follows :func:`resolve_proxy_kwargs`: explicit
        ``[proxy]`` configuration takes precedence and forces
        ``trust_env=False``, a disabled proxy forces direct connections
        (env vars ignored), and otherwise the ALL_PROXY, HTTPS_PROXY, and
        HTTP_PROXY environment variables are honored (httpx
        ``trust_env=True``). SOCKS proxies require the optional
        ``socksio`` package; without it, constructing the client raises
        ImportError.
    """
    client_kwargs: dict[str, Any] = {
        "headers": headers,
        "timeout": timeout,
        "limits": limits,
    }
    client_kwargs.update(resolve_proxy_kwargs(proxy_settings))
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
