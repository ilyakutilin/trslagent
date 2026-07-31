import json
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from src.http_client import RetryExhaustedError, create_client, fetch_with_retry


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
