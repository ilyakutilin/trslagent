"""LLM client for translation and review via OpenRouter API.

Provides an async OpenAI-compatible client with retry logic and a cost-fetching
utility for completion generation metadata.
"""

import asyncio
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from openai.types import ReasoningEffort
from openai.types.chat import ChatCompletionMessageParam

from src.config import CostSettings, logger


class LLM:
    """Async wrapper around an OpenAI-compatible chat completions API.

    Attributes:
        base_url: Base URL of the API endpoint.
        api_key: API key used for authentication.
        model: Model identifier string (e.g. "openai/gpt-4o").
        temperature: Sampling temperature (may be None for models that don't
            support it).
        reasoning_effort: Optional reasoning effort level for compatible models.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float | None,
        reasoning_effort: ReasoningEffort = None,
    ) -> None:
        """Initialize the LLM client.

        Args:
            base_url: Base URL of the API endpoint.
            api_key: API key for authentication.
            model: Model identifier string.
            temperature: Sampling temperature or None.
            reasoning_effort: Optional reasoning effort level.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self._client = None
        self._client_lock = asyncio.Lock()

    def _get_llm_client(self) -> OpenAI:
        """Create and return a synchronous OpenAI client.

        Returns:
            Configured OpenAI client instance.

        Raises:
            ValueError: If no API key is set.
        """
        if not self.api_key:
            raise ValueError("Set the OPENROUTER_API_KEY environment variable.")
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    async def get_reply_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, str]:
        """Send a chat completion request with retry logic.

        Lazily initializes the underlying OpenAI client on first call.
        Retries on timeout, rate-limit, and connection errors with exponential
        backoff (up to 5 attempts).

        Args:
            system_prompt: System-level instruction for the model.
            user_prompt: User-level content to process.

        Returns:
            A tuple of (response content, completion ID).

        Raises:
            RuntimeError: If all retries are exhausted for a transient error,
                or if a non-retryable exception occurs.
            ValueError: If the API returns an empty response content.
        """
        if not self._client:
            async with self._client_lock:
                if not self._client:
                    self._client = self._get_llm_client()

        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                messages: list[ChatCompletionMessageParam] = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ]
                kwargs: dict = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                }
                if self.reasoning_effort is not None:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                response = await asyncio.to_thread(
                    self._client.chat.completions.create,
                    **kwargs,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError(
                        "Translation failed: received empty response from API"
                    )
                return content.strip(), response.id

            except APITimeoutError as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Request timed out after {max_retries} retries. "
                        f"The API is taking too long to respond.\n"
                        f"Original error: {str(e)}"
                    )
                wait_time = base_delay * (2**attempt)
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                continue

            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Rate limit exceeded after {max_retries} retries. "
                        f"Please wait a moment and try again.\n"
                        f"Original error: {str(e)}"
                    )
                wait_time = base_delay * (2**attempt)
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                continue

            except APIConnectionError as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Network error after {max_retries} retries. "
                        f"Please check your internet connection and try again.\n"
                        f"Original error: {str(e)}"
                    )
                wait_time = base_delay * (2**attempt)
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                continue

            except Exception as e:
                raise RuntimeError(f"Translation failed: {str(e)}") from e

        return "", ""


def _find_key(obj: Any, key: str) -> Any:
    """Recursively search for a key in a nested dict/list structure.

    Args:
        obj: A dict, list, or other value to search.
        key: The dictionary key to look for.

    Returns:
        The value associated with *key* in the first matching dict, or None
        if not found.
    """
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            result = _find_key(v, key)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _find_key(item, key)
            if result is not None:
                return result
    return None


async def fetch_cost(
    completion_id: str,
    api_key: str,
    cost_settings: CostSettings,
) -> float | None:
    """Fetch the generation cost for a completion from the API.

    Queries the configured generation info URL, extracts the cost value using
    the key specified in *cost_settings*, and returns it as a float.

    Args:
        completion_id: The completion ID returned by the LLM.
        api_key: API key for authentication (passed as Bearer token).
        cost_settings: Configuration with the info URL and cost key.

    Returns:
        The cost as a float, or None if the fetch failed, the key was missing,
        or the value is not numeric.
    """
    if not cost_settings.generation_info_url:
        return None
    url = f"{cost_settings.generation_info_url}?id={completion_id}"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
    except Exception:
        logger.warning(
            f"Failed to fetch cost for completion {completion_id}: {url}",
            exc_info=True,
        )
        return None

    value = _find_key(data, cost_settings.cost_key)
    if value is None:
        logger.warning(
            f"Key '{cost_settings.cost_key}' not found in cost response "
            f"for completion {completion_id}"
        )
        return None
    if not isinstance(value, (int, float)):
        logger.warning(
            f"Cost value for key '{cost_settings.cost_key}' is not a number: "
            f"{type(value).__name__} for completion {completion_id}"
        )
        return None
    return float(value)
