import httpx
import pytest
import respx
from openai import APIConnectionError, APITimeoutError, RateLimitError
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from src.config import CostSettings
from src.llm import LLM, _find_key, fetch_cost


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
    async def test_http_error(self, cost_settings: CostSettings):
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
    async def test_connection_error(self, cost_settings: CostSettings):
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


def _make_completion(content: str | None, completion_id: str = "test-id") -> ChatCompletion:
    return ChatCompletion(
        id=completion_id,
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=content,
                    role="assistant",
                ),
            )
        ],
        created=1,
        model="test-model",
        object="chat.completion",
    )


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
    async def test_get_reply_success(self, llm: LLM, mocker):
        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion(
            "Translated text"
        )
        llm._client = mock_client

        text, cid = await llm.get_reply_async("system", "user")
        assert text == "Translated text"
        assert cid == "test-id"
        mock_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            temperature=0.3,
        )

    @pytest.mark.asyncio
    async def test_get_reply_timeout_retry(self, llm: LLM, mocker):
        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.side_effect = [
            APITimeoutError(request=httpx.Request("POST", "https://example.com")),
            _make_completion("Recovered text"),
        ]
        llm._client = mock_client
        mock_sleep = mocker.patch("asyncio.sleep")
        mocker.patch(
            "src.llm.asyncio.to_thread",
            side_effect=lambda fn, **kw: fn(**kw),
        )

        text, _ = await llm.get_reply_async("system", "user")
        assert text == "Recovered text"
        assert mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_get_reply_rate_limit_retry(self, llm: LLM, mocker):
        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.side_effect = [
            RateLimitError(
                message="Rate limit",
                response=mocker.MagicMock(status_code=429),
                body=None,
            ),
            _make_completion("Recovered text"),
        ]
        llm._client = mock_client
        mock_sleep = mocker.patch("asyncio.sleep")
        mocker.patch(
            "src.llm.asyncio.to_thread",
            side_effect=lambda fn, **kw: fn(**kw),
        )

        text, _ = await llm.get_reply_async("system", "user")
        assert text == "Recovered text"
        assert mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_get_reply_connection_error_retry(self, llm: LLM, mocker):
        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.side_effect = [
            APIConnectionError(request=mocker.MagicMock()),
            _make_completion("Recovered text"),
        ]
        llm._client = mock_client
        mock_sleep = mocker.patch("asyncio.sleep")
        mocker.patch(
            "src.llm.asyncio.to_thread",
            side_effect=lambda fn, **kw: fn(**kw),
        )

        text, _ = await llm.get_reply_async("system", "user")
        assert text == "Recovered text"
        assert mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_get_reply_timeout_exhausted(self, llm: LLM, mocker):
        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.side_effect = APITimeoutError(
            request=httpx.Request("POST", "https://example.com")
        )
        llm._client = mock_client
        mock_sleep = mocker.patch("asyncio.sleep")
        mocker.patch(
            "src.llm.asyncio.to_thread",
            side_effect=lambda fn, **kw: fn(**kw),
        )

        with pytest.raises(RuntimeError, match="timed out after 5 retries"):
            await llm.get_reply_async("system", "user")
        assert mock_client.chat.completions.create.call_count == 5
        assert mock_sleep.call_args_list == [
            mocker.call(1.0),
            mocker.call(2.0),
            mocker.call(4.0),
            mocker.call(8.0),
        ]

    @pytest.mark.asyncio
    async def test_get_reply_rate_limit_exhausted(self, llm: LLM, mocker):
        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit",
            response=mocker.MagicMock(status_code=429),
            body=None,
        )
        llm._client = mock_client
        mock_sleep = mocker.patch("asyncio.sleep")
        mocker.patch(
            "src.llm.asyncio.to_thread",
            side_effect=lambda fn, **kw: fn(**kw),
        )

        with pytest.raises(RuntimeError, match="Rate limit exceeded after 5 retries"):
            await llm.get_reply_async("system", "user")
        assert mock_client.chat.completions.create.call_count == 5
        assert mock_sleep.call_args_list == [
            mocker.call(1.0),
            mocker.call(2.0),
            mocker.call(4.0),
            mocker.call(8.0),
        ]

    @pytest.mark.asyncio
    async def test_get_reply_connection_error_exhausted(self, llm: LLM, mocker):
        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=mocker.MagicMock()
        )
        llm._client = mock_client
        mock_sleep = mocker.patch("asyncio.sleep")
        mocker.patch(
            "src.llm.asyncio.to_thread",
            side_effect=lambda fn, **kw: fn(**kw),
        )

        with pytest.raises(RuntimeError, match="Network error after 5 retries"):
            await llm.get_reply_async("system", "user")
        assert mock_client.chat.completions.create.call_count == 5
        assert mock_sleep.call_args_list == [
            mocker.call(1.0),
            mocker.call(2.0),
            mocker.call(4.0),
            mocker.call(8.0),
        ]

    @pytest.mark.asyncio
    async def test_get_reply_empty_content(self, llm: LLM, mocker):
        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion(None)
        llm._client = mock_client

        with pytest.raises(
            RuntimeError,
            match="Translation failed: received empty response from API",
        ):
            await llm.get_reply_async("system", "user")

    @pytest.mark.asyncio
    async def test_get_reply_unexpected_exception(self, llm: LLM, mocker):
        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.side_effect = ValueError("unexpected")
        llm._client = mock_client

        with pytest.raises(RuntimeError, match="Translation failed"):
            await llm.get_reply_async("system", "user")

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
