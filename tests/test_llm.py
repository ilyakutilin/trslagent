import json
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from iso639 import Lang

from src.config import CostSettings, InputData, ProxySettings, Settings
from src.http_client import LLM_TIMEOUT, create_client
from src.llm import LLM, _find_key, fetch_cost, resolve_and_log_cost


class TestFindKey:
    def test_flat_dict(self):
        assert _find_key({"a": 1, "b": 2}, "b") == 2

    def test_nested_dict(self):
        data = {"outer": {"inner": 42}}
        assert _find_key(data, "inner") == 42

    def test_list_of_dicts(self):
        data = [{"x": 1}, {"y": 2}, {"z": 3}]
        assert _find_key(data, "y") == 2

    def test_key_absent(self):
        data = {"a": 1, "b": {"c": 3}}
        assert _find_key(data, "missing") is None

    def test_none_input(self):
        assert _find_key(None, "any") is None

    def test_empty_dict(self):
        assert _find_key({}, "any") is None

    def test_empty_list(self):
        assert _find_key([], "any") is None

    def test_duplicate_keys_shallowest_wins(self):
        data = {"a": 1, "b": {"a": 2, "c": {"a": 3}}}
        assert _find_key(data, "a") == 1


class TestFetchCost:
    @pytest.fixture
    def cost_settings(self) -> CostSettings:
        return CostSettings(
            generation_info_url="https://api.example.com/generation",
            cost_key="total_cost",
        )

    @pytest.mark.asyncio
    async def test_success(self, cost_settings: CostSettings):
        url = f"{cost_settings.generation_info_url}?id=test-123"
        with respx.mock() as mock:
            mock.get(url).respond(json={"total_cost": 0.0042})
            result = await fetch_cost("test-123", "fake-key", cost_settings)
        assert result == 0.0042

    @pytest.mark.asyncio
    async def test_key_not_found(self, cost_settings: CostSettings):
        url = f"{cost_settings.generation_info_url}?id=test-123"
        with respx.mock() as mock:
            mock.get(url).respond(json={"other_field": "value"})
            result = await fetch_cost("test-123", "fake-key", cost_settings)
        assert result is None

    @pytest.mark.asyncio
    async def test_non_numeric_value(self, cost_settings: CostSettings):
        url = f"{cost_settings.generation_info_url}?id=test-123"
        with respx.mock() as mock:
            mock.get(url).respond(json={"total_cost": "not-a-number"})
            result = await fetch_cost("test-123", "fake-key", cost_settings)
        assert result is None

    @pytest.mark.asyncio
    async def test_http_error(self, cost_settings: CostSettings, mocker):
        mocker.patch("asyncio.sleep")
        url = f"{cost_settings.generation_info_url}?id=test-123"
        with respx.mock() as mock:
            mock.get(url).respond(status_code=500)
            result = await fetch_cost("test-123", "fake-key", cost_settings)
        assert result is None

    @pytest.mark.asyncio
    async def test_zero_cost_value(self, cost_settings: CostSettings):
        url = f"{cost_settings.generation_info_url}?id=test-123"
        with respx.mock() as mock:
            mock.get(url).respond(json={"total_cost": 0})
            result = await fetch_cost("test-123", "fake-key", cost_settings)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_empty_api_key(self, cost_settings: CostSettings):
        url = f"{cost_settings.generation_info_url}?id=test-123"
        with respx.mock() as mock:
            route = mock.get(url).respond(json={"total_cost": 0.001})
            result = await fetch_cost("test-123", "", cost_settings)
        assert result == 0.001
        assert "Authorization" not in route.calls.last.request.headers

    @pytest.mark.asyncio
    async def test_connection_error(self, cost_settings: CostSettings, mocker):
        mocker.patch("asyncio.sleep")
        url = f"{cost_settings.generation_info_url}?id=test-123"
        with respx.mock() as mock:
            mock.get(url).mock(side_effect=httpx.ConnectError("Connection refused"))
            result = await fetch_cost("test-123", "fake-key", cost_settings)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_generation_info_url(self):
        settings = CostSettings(generation_info_url=None, cost_key="total_cost")
        result = await fetch_cost("test-123", "fake-key", settings)
        assert result is None

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(
        self, cost_settings: CostSettings, mocker
    ):
        mocker.patch("asyncio.sleep")
        url = f"{cost_settings.generation_info_url}?id=test-123"
        with respx.mock() as mock:
            route = mock.get(url)
            route.side_effect = [
                httpx.Response(500),
                httpx.Response(200, json={"total_cost": 0.01}),
            ]
            result = await fetch_cost("test-123", "fake-key", cost_settings)
        assert result == 0.01

    @pytest.mark.asyncio
    async def test_retry_exhausted_logs_error_type(
        self, cost_settings: CostSettings, mocker
    ):
        mocker.patch("asyncio.sleep")
        url = f"{cost_settings.generation_info_url}?id=test-123"
        with respx.mock() as mock:
            mock.get(url).mock(side_effect=httpx.ConnectError("Connection refused"))
            mock_logger = mocker.patch("src.llm.logger")
            result = await fetch_cost("test-123", "fake-key", cost_settings)
        assert result is None
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "ConnectError" in warning_msg
        assert "Connection refused" in warning_msg
        assert "after 5 retries" in warning_msg

    @pytest.mark.asyncio
    async def test_injected_client_used_and_not_closed(
        self, cost_settings: CostSettings, mocker
    ):
        mock_create = mocker.patch("src.llm.create_client")
        url = f"{cost_settings.generation_info_url}?id=test-123"
        client = create_client()
        try:
            with respx.mock() as mock:
                mock.get(url).respond(json={"total_cost": 0.007})
                result = await fetch_cost(
                    "test-123", "fake-key", cost_settings, client=client
                )
            assert result == 0.007
            assert not client.is_closed
        finally:
            await client.aclose()
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_proxy_settings_forwarded_to_owned_client(
        self, cost_settings: CostSettings, mocker
    ):
        proxy = ProxySettings(enabled=False)
        mock_client = AsyncMock()
        mock_client.request.return_value = httpx.Response(
            200,
            json={"total_cost": 0.01},
            request=httpx.Request("GET", "https://api.example.com/generation"),
        )
        mock_create = mocker.patch("src.llm.create_client", return_value=mock_client)
        result = await fetch_cost(
            "test-123", "fake-key", cost_settings, proxy_settings=proxy
        )
        assert result == 0.01
        mock_create.assert_called_once_with(timeout=30, proxy_settings=proxy)


def _make_completion_response(
    content: str | None, completion_id: str = "test-id"
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "choices": [{"message": {"content": content}}],
        "created": 1,
        "model": "test-model",
        "object": "chat.completion",
    }


class TestLLM:
    @pytest.fixture
    def llm(self) -> LLM:
        return LLM(
            base_url="https://test.api",
            api_key="test-key",
            model="test-model",
            temperature=0.3,
        )

    @pytest.mark.asyncio
    async def test_get_reply_success(self, llm: LLM):
        with respx.mock() as mock:
            route = mock.post("https://test.api/chat/completions").respond(
                json=_make_completion_response("Translated text")
            )
            text, cid = await llm.get_reply_async("system", "user")
        assert text == "Translated text"
        assert cid == "test-id"
        assert route.calls.last.request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(route.calls.last.request.content)
        assert payload["model"] == "test-model"
        assert payload["messages"] == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
        assert payload["temperature"] == 0.3
        assert "reasoning_effort" not in payload

    @pytest.mark.asyncio
    async def test_get_reply_sends_reasoning_effort(self):
        llm = LLM(
            base_url="https://test.api",
            api_key="test-key",
            model="test-model",
            temperature=0.3,
            reasoning_effort="high",
        )
        with respx.mock() as mock:
            route = mock.post("https://test.api/chat/completions").respond(
                json=_make_completion_response("Translated text")
            )
            await llm.get_reply_async("system", "user")
        payload = json.loads(route.calls.last.request.content)
        assert payload["reasoning_effort"] == "high"

    def test_get_http_client_forwards_proxy_settings(self, mocker):
        proxy = ProxySettings(enabled=False)
        mock_create = mocker.patch("src.llm.create_client")
        llm = LLM(
            base_url="https://test.api",
            api_key="test-key",
            model="test-model",
            temperature=0.3,
            proxy_settings=proxy,
        )
        llm._get_http_client()
        mock_create.assert_called_once_with(
            base_url="https://test.api",
            headers={"Authorization": "Bearer test-key"},
            timeout=LLM_TIMEOUT,
            proxy_settings=proxy,
        )

    def test_get_http_client_proxy_settings_none_by_default(self, mocker):
        mock_create = mocker.patch("src.llm.create_client")
        llm = LLM(
            base_url="https://test.api",
            api_key="test-key",
            model="test-model",
            temperature=0.3,
        )
        llm._get_http_client()
        assert mock_create.call_args.kwargs["proxy_settings"] is None

    @pytest.mark.asyncio
    async def test_get_reply_timeout_retry(self, llm: LLM, mocker):
        mock_sleep = mocker.patch("asyncio.sleep")
        with respx.mock() as mock:
            route = mock.post("https://test.api/chat/completions")
            route.side_effect = [
                httpx.TimeoutException("timeout"),
                httpx.Response(200, json=_make_completion_response("Recovered text")),
            ]

            text, _ = await llm.get_reply_async("system", "user")
        assert text == "Recovered text"
        assert len(route.calls) == 2
        mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_get_reply_rate_limit_retry(self, llm: LLM, mocker):
        mock_sleep = mocker.patch("asyncio.sleep")
        with respx.mock() as mock:
            route = mock.post("https://test.api/chat/completions")
            route.side_effect = [
                httpx.Response(429),
                httpx.Response(200, json=_make_completion_response("Recovered text")),
            ]

            text, _ = await llm.get_reply_async("system", "user")
        assert text == "Recovered text"
        assert len(route.calls) == 2
        mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_get_reply_connection_error_retry(self, llm: LLM, mocker):
        mock_sleep = mocker.patch("asyncio.sleep")
        with respx.mock() as mock:
            route = mock.post("https://test.api/chat/completions")
            route.side_effect = [
                httpx.ConnectError("conn refused"),
                httpx.Response(200, json=_make_completion_response("Recovered text")),
            ]

            text, _ = await llm.get_reply_async("system", "user")
        assert text == "Recovered text"
        assert len(route.calls) == 2
        mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_get_reply_transport_error_retry(self, llm: LLM, mocker):
        mock_sleep = mocker.patch("asyncio.sleep")
        with respx.mock() as mock:
            route = mock.post("https://test.api/chat/completions")
            route.side_effect = [
                httpx.ReadError("conn reset"),
                httpx.Response(200, json=_make_completion_response("Recovered text")),
            ]

            text, _ = await llm.get_reply_async("system", "user")
        assert text == "Recovered text"
        assert len(route.calls) == 2
        mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_get_reply_timeout_exhausted(self, llm: LLM, mocker):
        mock_sleep = mocker.patch("asyncio.sleep")
        with respx.mock() as mock:
            route = mock.post("https://test.api/chat/completions")
            route.side_effect = httpx.TimeoutException("timeout")

            with pytest.raises(RuntimeError, match="timed out after 5 retries"):
                await llm.get_reply_async("system", "user")
        assert len(route.calls) == 5
        assert mock_sleep.call_args_list == [
            mocker.call(1.0),
            mocker.call(2.0),
            mocker.call(4.0),
            mocker.call(8.0),
        ]

    @pytest.mark.asyncio
    async def test_get_reply_rate_limit_exhausted(self, llm: LLM, mocker):
        mock_sleep = mocker.patch("asyncio.sleep")
        with respx.mock() as mock:
            route = mock.post("https://test.api/chat/completions")
            route.side_effect = [httpx.Response(429)] * 5

            with pytest.raises(
                RuntimeError, match="Rate limit exceeded after 5 retries"
            ):
                await llm.get_reply_async("system", "user")
        assert len(route.calls) == 5
        assert mock_sleep.call_args_list == [
            mocker.call(1.0),
            mocker.call(2.0),
            mocker.call(4.0),
            mocker.call(8.0),
        ]

    @pytest.mark.asyncio
    async def test_get_reply_connection_error_exhausted(self, llm: LLM, mocker):
        mock_sleep = mocker.patch("asyncio.sleep")
        with respx.mock() as mock:
            route = mock.post("https://test.api/chat/completions")
            route.side_effect = httpx.ConnectError("conn refused")

            with pytest.raises(RuntimeError, match="Network error after 5 retries"):
                await llm.get_reply_async("system", "user")
        assert len(route.calls) == 5
        assert mock_sleep.call_args_list == [
            mocker.call(1.0),
            mocker.call(2.0),
            mocker.call(4.0),
            mocker.call(8.0),
        ]

    @pytest.mark.asyncio
    async def test_get_reply_empty_content(self, llm: LLM):
        with respx.mock() as mock:
            mock.post("https://test.api/chat/completions").respond(
                json=_make_completion_response(None)
            )

            with pytest.raises(
                RuntimeError,
                match="Translation failed: received empty response from API",
            ):
                await llm.get_reply_async("system", "user")

    @pytest.mark.asyncio
    async def test_get_reply_unexpected_exception(self, llm: LLM):
        with respx.mock() as mock:
            route = mock.post("https://test.api/chat/completions")
            route.side_effect = ValueError("unexpected")

            with pytest.raises(RuntimeError, match="Translation failed"):
                await llm.get_reply_async("system", "user")
        assert len(route.calls) == 1

    @pytest.mark.asyncio
    async def test_get_reply_http_500_no_retry(self, llm: LLM):
        with respx.mock() as mock:
            route = mock.post("https://test.api/chat/completions").respond(
                status_code=500
            )

            with pytest.raises(RuntimeError, match="Translation failed"):
                await llm.get_reply_async("system", "user")
        assert len(route.calls) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response_kwargs",
        [
            {"content": b"not json"},
            {"json": {}},
            {"json": {"choices": []}},
        ],
    )
    async def test_get_reply_malformed_response(self, llm: LLM, response_kwargs):
        with respx.mock() as mock:
            route = mock.post("https://test.api/chat/completions").respond(
                **response_kwargs
            )

            with pytest.raises(RuntimeError, match="Translation failed"):
                await llm.get_reply_async("system", "user")
        assert len(route.calls) == 1

    @pytest.mark.asyncio
    async def test_init_no_api_key_lazy_raise(self):
        llm = LLM(
            base_url="https://test.api",
            api_key="",
            model="test-model",
            temperature=0.3,
        )
        with pytest.raises(
            ValueError, match="Set the OPENROUTER_API_KEY environment variable"
        ):
            await llm.get_reply_async("system", "user")

    @pytest.mark.asyncio
    async def test_close_awaits_client_aclose_and_resets(self, llm: LLM):
        mock_client = AsyncMock()
        llm._client = mock_client

        await llm.close()

        mock_client.aclose.assert_awaited_once()
        assert llm._client is None

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, llm: LLM):
        mock_client = AsyncMock()
        llm._client = mock_client

        await llm.close()
        await llm.close()

        mock_client.aclose.assert_awaited_once()
        assert llm._client is None

    @pytest.mark.asyncio
    async def test_close_is_noop_without_client(self, llm: LLM):
        assert llm._client is None
        await llm.close()
        assert llm._client is None


class TestResolveAndLogCost:
    @pytest.fixture(autouse=True)
    def _reset_toml_path(self):
        Settings._toml_path = None
        yield
        Settings._toml_path = None

    @pytest.mark.asyncio
    async def test_known_costs(self, mocker):
        mock_fetch = mocker.patch("src.llm.fetch_cost", side_effect=[1.50, 2.50])

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url="https://api.example.com/cost"),
        )

        await resolve_and_log_cost(["id-1", "id-2"], "test-key", cfg)
        assert mock_fetch.call_count == 2
        mock_fetch.assert_any_call(
            "id-1", "test-key", cfg.cost, client=None, proxy_settings=cfg.proxy
        )
        mock_fetch.assert_any_call(
            "id-2", "test-key", cfg.cost, client=None, proxy_settings=cfg.proxy
        )

    @pytest.mark.asyncio
    async def test_forwards_injected_client_to_fetch_cost(self, mocker):
        mock_fetch = mocker.patch("src.llm.fetch_cost", side_effect=[1.50, 2.50])
        injected = AsyncMock()

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url="https://api.example.com/cost"),
        )

        await resolve_and_log_cost(["id-1", "id-2"], "test-key", cfg, client=injected)
        assert mock_fetch.call_count == 2
        mock_fetch.assert_any_call(
            "id-1", "test-key", cfg.cost, client=injected, proxy_settings=cfg.proxy
        )
        mock_fetch.assert_any_call(
            "id-2", "test-key", cfg.cost, client=injected, proxy_settings=cfg.proxy
        )

    @pytest.mark.asyncio
    async def test_passes_client_none_when_not_injected(self, mocker):
        mock_fetch = mocker.patch("src.llm.fetch_cost", side_effect=[1.50])

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url="https://api.example.com/cost"),
        )

        await resolve_and_log_cost(["id-1"], "test-key", cfg)
        mock_fetch.assert_called_once_with(
            "id-1", "test-key", cfg.cost, client=None, proxy_settings=cfg.proxy
        )

    @pytest.mark.asyncio
    async def test_unknown_costs(self, mocker):
        mock_fetch = mocker.patch("src.llm.fetch_cost", side_effect=[1.50, None])

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url="https://api.example.com/cost"),
        )

        await resolve_and_log_cost(["id-1", "id-2"], "test-key", cfg)
        assert mock_fetch.call_count == 2

    @pytest.mark.asyncio
    async def test_no_url_configured(self, mocker):
        mock_fetch = mocker.patch("src.llm.fetch_cost")

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url=None),
        )

        await resolve_and_log_cost(["id-1"], "test-key", cfg)
        mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_completion_ids(self, mocker):
        mock_fetch = mocker.patch("src.llm.fetch_cost")

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url="https://api.example.com/cost"),
        )

        await resolve_and_log_cost([], "test-key", cfg)
        mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_tuple_with_known_costs(self, mocker):
        mocker.patch("src.llm.fetch_cost", side_effect=[1.50, 2.50])

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url="https://api.example.com/cost"),
        )

        total, currency, unknowns = await resolve_and_log_cost(
            ["id-1", "id-2"], "test-key", cfg
        )
        assert total == 4.0
        assert currency == "USD"
        assert unknowns == 0

    @pytest.mark.asyncio
    async def test_returns_none_total_with_all_unknown(self, mocker):
        mocker.patch("src.llm.fetch_cost", side_effect=[None, None])

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url="https://api.example.com/cost"),
        )

        total, currency, unknowns = await resolve_and_log_cost(
            ["id-1", "id-2"], "test-key", cfg
        )
        assert total is None
        assert currency == "USD"
        assert unknowns == 2

    @pytest.mark.asyncio
    async def test_returns_total_only_for_known(self, mocker):
        mocker.patch("src.llm.fetch_cost", side_effect=[1.50, None])

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url="https://api.example.com/cost"),
        )

        total, _, unknowns = await resolve_and_log_cost(
            ["id-1", "id-2"], "test-key", cfg
        )
        assert total == 1.50
        assert unknowns == 1

    @pytest.mark.asyncio
    async def test_returns_none_when_no_url(self, mocker):
        mock_fetch = mocker.patch("src.llm.fetch_cost")

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url=None),
        )

        total, currency, unknowns = await resolve_and_log_cost(
            ["id-1"], "test-key", cfg
        )
        mock_fetch.assert_not_called()
        assert total is None
        assert currency == "USD"
        assert unknowns == 0

    @pytest.mark.asyncio
    async def test_returns_none_when_empty_ids(self, mocker):
        mock_fetch = mocker.patch("src.llm.fetch_cost")

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url="https://api.example.com/cost"),
        )

        total, _, _ = await resolve_and_log_cost([], "test-key", cfg)
        mock_fetch.assert_not_called()
        assert total is None

    @pytest.mark.asyncio
    async def test_fetch_cost_raises_logs_warning_and_counts_unknown(self, mocker):
        mocker.patch(
            "src.llm.fetch_cost",
            side_effect=[httpx.HTTPError("boom"), 1.25],
        )

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url="https://api.example.com/cost"),
        )

        total, _, unknowns = await resolve_and_log_cost(
            ["id-1", "id-2"], "test-key", cfg
        )
        assert total == 1.25
        assert unknowns == 1
