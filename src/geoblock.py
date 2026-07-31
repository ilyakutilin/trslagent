"""Geoblocking: verification of the outgoing IP country against a blocklist.

The outgoing IP is the one an LLM provider would see, i.e. the proxied
IP when a proxy is configured. The verification queries three free,
keyless IP-geo checkers concurrently and fails closed: the request is
allowed only when at least ``MIN_CLEAN_CONFIRMATIONS`` checkers confirm
the IP is not located in any blocked country.
"""

import asyncio
from typing import NamedTuple

import httpx

from iso3166 import countries_by_alpha2

from src.config import ProxySettings, logger
from src.http_client import create_client

CHECKER_TIMEOUT: float = 5.0
MIN_CLEAN_CONFIRMATIONS: int = 2


class GeoCheckerSpec(NamedTuple):
    """Specification of a single IP-geo checker endpoint.

    Attributes:
        name: Human-readable checker name used in logs and error messages.
        url: Endpoint URL returning JSON with the country of the outgoing IP.
        country_key: JSON key holding the alpha2 country code.
    """

    name: str
    url: str
    country_key: str


CHECKERS: tuple[GeoCheckerSpec, ...] = (
    GeoCheckerSpec(
        name="ip-api.com",
        url="http://ip-api.com/json/",
        country_key="countryCode",
    ),
    GeoCheckerSpec(
        name="ipinfo.io",
        url="https://ipinfo.io/json",
        country_key="country",
    ),
    GeoCheckerSpec(
        name="api.country.is",
        url="https://api.country.is/",
        country_key="country",
    ),
)


class GeoblockError(RuntimeError):
    """Raised when the outgoing IP is in a blocked country or its geography cannot be verified."""


async def _query_checker(
    client: httpx.AsyncClient,
    spec: GeoCheckerSpec,
    timeout: float,
) -> str | None:
    """Query a single geo checker for the alpha2 code of the outgoing IP country.

    Args:
        client: HTTP client to use for the request.
        spec: Checker endpoint specification.
        timeout: Per-request timeout in seconds.

    Returns:
        The uppercase alpha2 country code if the checker responded with a
        valid one, or None on any failure (timeout, HTTP error, malformed
        or unrecognized response).
    """
    try:
        response = await client.get(
            spec.url,
            timeout=timeout,
            headers={"User-Agent": "trslagent/0.1"},
        )
        response.raise_for_status()
        data = response.json()
        code = data.get(spec.country_key)
    except Exception as e:
        logger.warning("Geo checker {} failed: {}: {}", spec.name, type(e).__name__, e)
        return None
    if not isinstance(code, str):
        logger.warning(
            "Geo checker {} returned a non-string or missing country value: {!r}",
            spec.name,
            code,
        )
        return None
    code = code.strip().upper()
    if code not in countries_by_alpha2:
        logger.warning(
            "Geo checker {} returned an unrecognized country code: {!r}",
            spec.name,
            code,
        )
        return None
    return code


async def verify_ip_not_geoblocked(
    blocked_countries: list[str],
    *,
    proxy_settings: ProxySettings | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = CHECKER_TIMEOUT,
) -> None:
    """Verify that the outgoing IP is not located in any blocked country.

    The outgoing IP is the one the LLM provider would see, i.e. the
    proxied IP when a proxy is configured. All checkers from
    ``CHECKERS`` are queried concurrently; the check fails closed, so
    the request proceeds only when at least ``MIN_CLEAN_CONFIRMATIONS``
    checkers confirm a country that is not in the blocked set.

    Args:
        blocked_countries: Canonical uppercase alpha2 country codes to
            block requests from. An empty list disables the check
            (returns immediately).
        proxy_settings: Proxy settings used when creating the owned
            client; an injected *client* is unaffected.
        client: Optional injected HTTP client; when None, an owned client
            is created (with *proxy_settings*) and closed before
            returning, even when the check raises.
        timeout: Per-checker request timeout in seconds.

    Returns:
        None when the outgoing IP is confirmed to be outside all blocked
        countries, or immediately when *blocked_countries* is empty.

    Raises:
        GeoblockError: If the outgoing IP is in a blocked country or its
            geography cannot be confirmed by enough checkers.
    """
    blocked = {code.upper() for code in blocked_countries}
    if not blocked_countries:
        logger.debug("Geoblocking disabled: no blocked countries configured")
        return None
    logger.info(
        "Verifying outgoing IP is not in blocked countries: {}",
        ", ".join(sorted(blocked)),
    )

    active_client = client
    owned_client: httpx.AsyncClient | None = None
    if active_client is None:
        owned_client = create_client(proxy_settings=proxy_settings)
        active_client = owned_client

    try:
        results: list[str | None] = await asyncio.gather(
            *(_query_checker(active_client, spec, timeout) for spec in CHECKERS)
        )

        codes: list[str] = []
        clean = 0
        for spec, code in zip(CHECKERS, results, strict=True):
            logger.debug(
                "Geo checker {} reports country: {}", spec.name, code or "unknown"
            )
            if code is None:
                continue
            codes.append(code)
            if code in blocked:
                logger.error(
                    "Request blocked: the outgoing IP is in the blocked country "
                    "{} (per {} checker)",
                    code,
                    spec.name,
                )
                raise GeoblockError(
                    f"Request blocked: the outgoing IP is in the blocked country "
                    f"{code} (per {spec.name} checker)"
                )
            clean += 1

        if clean >= MIN_CLEAN_CONFIRMATIONS:
            logger.info(
                "IP geolocation confirmed clean by {} of {} checkers: {}",
                clean,
                len(CHECKERS),
                ", ".join(codes),
            )
            return None

        logger.error(
            "Request blocked: unable to verify the outgoing IP is not in blocked "
            "countries ({} of {} checkers confirmed)",
            clean,
            len(CHECKERS),
        )
        raise GeoblockError(
            f"Request blocked: unable to verify the outgoing IP is not in blocked "
            f"countries ({clean} of {len(CHECKERS)} checkers confirmed)"
        )
    finally:
        if owned_client is not None:
            try:
                await owned_client.aclose()
            except Exception as e:
                logger.warning("Failed to close geoblock HTTP client: {}", e)
