from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.config import ProxySettings, parse_geoblock_countries
from src.geoblock import (
    CHECKERS,
    GeoCheckerSpec,
    GeoblockError,
    _query_checker,
    verify_ip_not_geoblocked,
)

spec = GeoCheckerSpec(name="test", url="https://test.example/", country_key="country")


def _client_with_response(resp: MagicMock) -> AsyncMock:
    """Build a fake HTTP client returning the given response from get()."""
    client = AsyncMock()
    client.get.return_value = resp
    return client


def _make_client(responses: list[str | None]) -> AsyncMock:
    """Build a fake HTTP client answering each checker with the given codes."""

    async def get(url, timeout=None, headers=None):
        index = [spec.url for spec in CHECKERS].index(url)
        code = responses[index]
        if code is None:
            raise TimeoutError(f"timeout {url}")
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"countryCode": code, "country": code}
        return resp

    client = AsyncMock()
    client.get.side_effect = get
    return client


class TestParseGeoblockCountries:
    def test_alpha2_normalized_to_upper(self):
        assert parse_geoblock_countries(["us"]) == ["US"]

    def test_alpha3_code(self):
        assert parse_geoblock_countries(["RUS"]) == ["RU"]

    def test_numeric_code(self):
        assert parse_geoblock_countries(["643"]) == ["RU"]

    def test_full_name(self):
        assert parse_geoblock_countries(["Russian Federation"]) == ["RU"]

    def test_full_name_case_insensitive(self):
        assert parse_geoblock_countries(["russian federation"]) == ["RU"]

    def test_deduplicates_preserving_order(self):
        assert parse_geoblock_countries(["RU", "ru", "RUS"]) == ["RU"]

    def test_mixed_forms_deduplicated(self):
        assert parse_geoblock_countries(["ru", "BY", "643"]) == ["RU", "BY"]

    def test_empty_list(self):
        assert parse_geoblock_countries([]) == []

    def test_unknown_values_raise_listing_all(self):
        with pytest.raises(ValueError, match="Unknown country") as exc_info:
            parse_geoblock_countries(["NotACountry", "Atlantis"])
        message = str(exc_info.value)
        assert "NotACountry" in message
        assert "Atlantis" in message


class TestVerifyIpNotGeoblocked:
    @pytest.mark.asyncio
    async def test_all_clean_returns_none(self):
        client = _make_client(["US", "US", "US"])
        result = await verify_ip_not_geoblocked(["RU"], client=client)
        assert result is None

    @pytest.mark.asyncio
    async def test_blocked_country_raises(self):
        client = _make_client(["RU", "US", "US"])
        with pytest.raises(GeoblockError, match="blocked country RU"):
            await verify_ip_not_geoblocked(["RU"], client=client)

    @pytest.mark.asyncio
    async def test_blocked_from_last_checker_beats_clean_majority(self):
        client = _make_client(["US", "US", "RU"])
        with pytest.raises(GeoblockError, match="blocked country RU"):
            await verify_ip_not_geoblocked(["RU"], client=client)

    @pytest.mark.asyncio
    async def test_empty_blocked_list_returns_none_without_queries(self):
        client = AsyncMock()
        result = await verify_ip_not_geoblocked([], client=client)
        assert result is None
        client.get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_blocked_list_skips_owned_client_creation(self, mocker):
        mock_create = mocker.patch("src.geoblock.create_client")
        result = await verify_ip_not_geoblocked([])
        assert result is None
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_one_checker_failure_still_passes(self):
        client = _make_client(["US", None, "US"])
        result = await verify_ip_not_geoblocked(["RU"], client=client)
        assert result is None

    @pytest.mark.asyncio
    async def test_insufficient_confirmations_raises(self):
        client = _make_client(["US", None, None])
        with pytest.raises(GeoblockError, match="unable to verify"):
            await verify_ip_not_geoblocked(["RU"], client=client)

    @pytest.mark.asyncio
    async def test_all_checkers_fail_raises(self):
        client = _make_client([None, None, None])
        with pytest.raises(GeoblockError, match="unable to verify"):
            await verify_ip_not_geoblocked(["RU"], client=client)

    @pytest.mark.asyncio
    async def test_invalid_country_code_treated_as_failure(self):
        client = _make_client(["US", "XX", "XX"])
        with pytest.raises(GeoblockError, match="unable to verify"):
            await verify_ip_not_geoblocked(["RU"], client=client)

    @pytest.mark.asyncio
    async def test_lowercase_code_normalized_to_clean(self):
        client = _make_client(["us", "US", "US"])
        result = await verify_ip_not_geoblocked(["RU"], client=client)
        assert result is None


class TestVerifyOwnedClient:
    @pytest.mark.asyncio
    async def test_owned_client_created_with_proxy_settings_and_closed(self, mocker):
        proxy = ProxySettings(enabled=False)
        client = _make_client(["US", "US", "US"])
        mock_create = mocker.patch("src.geoblock.create_client", return_value=client)

        await verify_ip_not_geoblocked(["RU"], proxy_settings=proxy)

        mock_create.assert_called_once_with(proxy_settings=proxy)
        client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_owned_client_closed_on_geoblock_error(self, mocker):
        client = MagicMock()
        client.get.side_effect = TimeoutError("timeout")
        client.aclose = AsyncMock()
        mocker.patch("src.geoblock.create_client", return_value=client)

        with pytest.raises(GeoblockError):
            await verify_ip_not_geoblocked(["RU"])

        client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_injected_client_not_closed(self, mocker):
        client = _make_client(["US", "US", "US"])
        mock_create = mocker.patch("src.geoblock.create_client")

        await verify_ip_not_geoblocked(["RU"], client=client)

        mock_create.assert_not_called()
        client.aclose.assert_not_awaited()


class TestQueryChecker:
    @pytest.mark.asyncio
    async def test_http_error_status_returns_none(self):
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error",
            request=httpx.Request("GET", spec.url),
            response=httpx.Response(500, request=httpx.Request("GET", spec.url)),
        )
        resp.json.return_value = {}

        result = await _query_checker(_client_with_response(resp), spec, 5.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_non_dict_json_returns_none(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = [1, 2, 3]

        result = await _query_checker(_client_with_response(resp), spec, 5.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_non_string_country_value_returns_none(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"country": 42}

        result = await _query_checker(_client_with_response(resp), spec, 5.0)
        assert result is None
