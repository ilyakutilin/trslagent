import json
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from pydantic import SecretStr

from src.config import ProxySettings
from src.http_client import (
    ProxyConfigError,
    RetryExhaustedError,
    create_client,
    fetch_with_retry,
    resolve_proxy_kwargs,
)


class TestCreateClient:
    @pytest.mark.asyncio
    async def test_default_timeout_is_30(self):
        client = create_client()
        try:
            assert client.timeout.connect == 30.0
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_custom_timeout_passed_through(self):
        client = create_client(timeout=10.5)
        try:
            assert client.timeout.connect == 10.5
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_base_url_none_works_with_absolute_urls(self):
        client = create_client()
        try:
            assert str(client.base_url) == ""
            with respx.mock() as mock:
                route = mock.get("https://api.example.com/items").respond(
                    json={"ok": True}
                )
                response = await client.get("https://api.example.com/items")
            assert response.json() == {"ok": True}
            assert route.called
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_headers_passed_through(self):
        client = create_client(headers={"X-Custom": "value"})
        try:
            assert client.headers["X-Custom"] == "value"
        finally:
            await client.aclose()


class TestFetchWithRetry:
    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        with respx.mock() as mock:
            route = mock.get("https://api.example.com/ok").respond(
                json={"status": "ok"}
            )
            response = await fetch_with_retry("GET", "https://api.example.com/ok")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert len(route.calls) == 1

    @pytest.mark.asyncio
    async def test_recovers_after_429(self, mocker):
        mocker.patch("asyncio.sleep")
        with respx.mock() as mock:
            route = mock.get("https://api.example.com/rate-limited")
            route.side_effect = [
                httpx.Response(429),
                httpx.Response(200, json={"ok": True}),
            ]
            response = await fetch_with_retry(
                "GET", "https://api.example.com/rate-limited"
            )
        assert response.json() == {"ok": True}
        assert len(route.calls) == 2

    @pytest.mark.asyncio
    async def test_recovers_after_timeout(self, mocker):
        mocker.patch("asyncio.sleep")
        with respx.mock() as mock:
            route = mock.get("https://api.example.com/slow")
            route.side_effect = [
                httpx.TimeoutException("timed out"),
                httpx.Response(200, json={"ok": True}),
            ]
            response = await fetch_with_retry("GET", "https://api.example.com/slow")
        assert response.json() == {"ok": True}
        assert len(route.calls) == 2

    @pytest.mark.asyncio
    async def test_recovers_after_connect_error(self, mocker):
        mocker.patch("asyncio.sleep")
        with respx.mock() as mock:
            route = mock.get("https://api.example.com/unreachable")
            route.side_effect = [
                httpx.ConnectError("connection refused"),
                httpx.Response(200, json={"ok": True}),
            ]
            response = await fetch_with_retry(
                "GET", "https://api.example.com/unreachable"
            )
        assert response.json() == {"ok": True}
        assert len(route.calls) == 2

    @pytest.mark.asyncio
    async def test_retries_http_500(self, mocker):
        mocker.patch("asyncio.sleep")
        with respx.mock() as mock:
            route = mock.get("https://api.example.com/broken")
            route.side_effect = [
                httpx.Response(500),
                httpx.Response(200, json={"ok": True}),
            ]
            response = await fetch_with_retry("GET", "https://api.example.com/broken")
        assert response.json() == {"ok": True}
        assert len(route.calls) == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_and_raises_with_cause(self, mocker):
        mock_sleep = mocker.patch("asyncio.sleep")
        with respx.mock() as mock:
            route = mock.get("https://api.example.com/down")
            route.side_effect = httpx.ConnectError("connection refused")

            with pytest.raises(RetryExhaustedError) as exc_info:
                await fetch_with_retry("GET", "https://api.example.com/down")
        assert len(route.calls) == 5
        assert mock_sleep.call_args_list == [
            mocker.call(1.0),
            mocker.call(2.0),
            mocker.call(4.0),
            mocker.call(8.0),
        ]
        assert isinstance(exc_info.value.__cause__, httpx.ConnectError)

    @pytest.mark.asyncio
    async def test_injected_client_is_not_closed(self):
        client = create_client()
        try:
            with respx.mock() as mock:
                mock.get("https://api.example.com/ok").respond(json={"status": "ok"})
                response = await fetch_with_retry(
                    "GET", "https://api.example.com/ok", client=client
                )
            assert response.status_code == 200
            assert not client.is_closed
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_owns_and_closes_client_when_none_injected(self, mocker):
        client = AsyncMock()
        client.request.return_value = httpx.Response(
            200,
            json={"status": "ok"},
            request=httpx.Request("GET", "https://api.example.com/ok"),
        )
        mocker.patch("src.http_client.create_client", return_value=client)

        response = await fetch_with_retry("GET", "https://api.example.com/ok")

        assert response.json() == {"status": "ok"}
        client.request.assert_awaited_once()
        client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_json_body_passed_to_request(self):
        with respx.mock() as mock:
            route = mock.post("https://api.example.com/items").respond(json={"id": 1})
            await fetch_with_retry(
                "POST", "https://api.example.com/items", json={"name": "widget"}
            )
        body = json.loads(route.calls.last.request.content)
        assert body == {"name": "widget"}


class TestProxyResolution:
    def test_none_settings_legacy_behavior(self):
        assert resolve_proxy_kwargs(None) == {}

    def test_disabled_forces_direct_connections(self):
        assert resolve_proxy_kwargs(ProxySettings(enabled=False)) == {
            "trust_env": False
        }

    def test_explicit_protocol_builds_proxy_url(self):
        assert resolve_proxy_kwargs(
            ProxySettings(protocol="http", host="10.0.0.1", port=3128)
        ) == {"proxy": "http://10.0.0.1:3128", "trust_env": False}

    def test_socks_protocols_accepted(self):
        for proto in ("socks5", "socks5h", "socks4", "socks4a"):
            kwargs = resolve_proxy_kwargs(
                ProxySettings(protocol=proto, host="127.0.0.1", port=1080)
            )
            assert kwargs == {"proxy": f"{proto}://127.0.0.1:1080", "trust_env": False}

    def test_protocol_with_username_only(self):
        kwargs = resolve_proxy_kwargs(ProxySettings(protocol="http", username="user"))
        assert kwargs["proxy"] == "http://user@127.0.0.1:1080"

    def test_protocol_with_username_and_password_encoded(self):
        kwargs = resolve_proxy_kwargs(
            ProxySettings(
                protocol="http",
                username="user",
                password=SecretStr("p@ss:w"),
            )
        )
        assert kwargs["proxy"] == "http://user:p%40ss%3Aw@127.0.0.1:1080"

    def test_password_without_username_not_included(self):
        kwargs = resolve_proxy_kwargs(
            ProxySettings(
                protocol="http",
                password=SecretStr("secret"),
            )
        )
        assert kwargs["proxy"] == "http://127.0.0.1:1080"

    def test_invalid_protocol_raises(self):
        with pytest.raises(ProxyConfigError) as exc_info:
            resolve_proxy_kwargs(ProxySettings(protocol="ftp"))
        assert "ftp" in str(exc_info.value)
        assert "http" in str(exc_info.value)
        assert "socks5" in str(exc_info.value)

    def test_whitespace_protocol_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("ALL_PROXY", "http://proxy:3128")
        assert resolve_proxy_kwargs(ProxySettings(protocol="  ")) == {"trust_env": True}

    def test_env_var_upper_case_used(self, monkeypatch):
        monkeypatch.setenv("ALL_PROXY", "http://proxy:3128")
        assert resolve_proxy_kwargs(ProxySettings()) == {"trust_env": True}

    def test_env_var_lower_case_used(self, monkeypatch):
        monkeypatch.setenv("https_proxy", "http://proxy:3128")
        assert resolve_proxy_kwargs(ProxySettings()) == {"trust_env": True}

    def test_empty_env_var_not_used(self, monkeypatch):
        monkeypatch.setenv("ALL_PROXY", "")
        with pytest.raises(ProxyConfigError):
            resolve_proxy_kwargs(ProxySettings())

    def test_no_settings_and_no_env_raises(self, monkeypatch):
        monkeypatch.delenv("ALL_PROXY", raising=False)
        monkeypatch.delenv("all_proxy", raising=False)
        with pytest.raises(ProxyConfigError) as exc_info:
            resolve_proxy_kwargs(ProxySettings())
        assert "PROXY__ENABLED=false" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_client_disabled_has_trust_env_false(self):
        client = create_client(proxy_settings=ProxySettings(enabled=False))
        try:
            assert client.trust_env is False
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_create_client_legacy_has_trust_env_true(self):
        client = create_client()
        try:
            assert client.trust_env is True
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_create_client_explicit_proxy_passed_to_constructor(self, mocker):
        mock_async = mocker.patch("httpx.AsyncClient")
        create_client(proxy_settings=ProxySettings(protocol="socks5h", port=1080))
        kwargs = mock_async.call_args.kwargs
        assert kwargs["proxy"] == "socks5h://127.0.0.1:1080"
        assert kwargs["trust_env"] is False
