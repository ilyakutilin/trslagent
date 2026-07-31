"""LLM client for translation and review via OpenRouter API.

Provides an async HTTP client for OpenAI-compatible chat completions APIs
with retry logic and a cost-fetching utility for completion generation
metadata.
"""

import asyncio
from typing import Any, Literal

import httpx

from src.config import CostSettings, Settings, logger


class _RateLimitError(Exception):
    """Raised when the chat completions API returns an HTTP 429 response."""


class LLM:
    """Async HTTP client for an OpenAI-compatible chat completions API.

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
        reasoning_effort: (
            Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None
        ) = None,
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
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    def _get_http_client(self) -> httpx.AsyncClient:
        """Create and return an async HTTP client.

        Returns:
            Configured httpx.AsyncClient instance.

        Raises:
            ValueError: If no API key is set.
        """
        if not self.api_key:
            raise ValueError("Set the OPENROUTER_API_KEY environment variable.")
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=600,
        )

    async def get_reply_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, str]:
        """Send a chat completion request with retry logic.

        Lazily initializes the underlying httpx AsyncClient on first call.
        Retries on timeout, rate-limit, and connection errors with exponential
        backoff (up to 5 attempts).

        Args:
            system_prompt: System-level instruction for the model.
            user_prompt: User-level content to process.

        Returns:
            A tuple of (response content, completion ID).

        Raises:
            RuntimeError: If all retries are exhausted for a transient error,
                if a non-retryable exception occurs, or if the API returns
                an empty response content.
        """
        if not self._client:
            async with self._client_lock:
                if not self._client:
                    # TODO: the AsyncClient is never closed — add an async
                    # close() hook and call it from main() / the email
                    # pipeline to release the connection pool.
                    self._client = self._get_http_client()

        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                payload: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    "temperature": self.temperature,
                }
                if self.reasoning_effort is not None:
                    payload["reasoning_effort"] = self.reasoning_effort
                response = await self._client.post("/chat/completions", json=payload)
                if response.status_code == 429:
                    raise _RateLimitError(
                        response.text or f"HTTP status {response.status_code}"
                    )
                # TODO: decide whether 502/503/504 should be retried like
                # 429 — currently any 5xx raises immediately (documented
                # "other errors raise immediately" contract).
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if content is None:
                    raise ValueError(
                        "Translation failed: received empty response from API"
                    )
                completion_id = data["id"]
                return content.strip(), completion_id

            except httpx.TimeoutException as e:
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

            except _RateLimitError as e:
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

            except httpx.TransportError as e:
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
    headers: dict[str, str] = {"User-Agent": "curl/8.0", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    max_retries = 5
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
            break
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Cost fetch attempt {attempt + 1}/{max_retries} failed for "
                    f"completion {completion_id}: {type(e).__name__}: {e}"
                )
                await asyncio.sleep(3)
    else:
        logger.warning(
            f"Failed to fetch cost for completion {completion_id} after "
            f"{max_retries} retries: {type(last_error).__name__}: {last_error}"
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


async def resolve_and_log_cost(
    completion_ids: list[str],
    api_key: str,
    cfg: Settings,
) -> tuple[float | None, str, int]:
    """Fetch and log the total cost of all LLM completions.

    Fetches costs concurrently for each completion ID via the generation info URL.
    Logs warnings for failed fetches and returns aggregate totals.

    Args:
        completion_ids: List of LLM completion IDs to query cost for.
        api_key: API key for authentication.
        cfg: Application settings containing cost configuration.

    Returns:
        A tuple of (total_cost_or_None, currency, unknown_count), where
        total_cost_or_None is the sum of known costs (None if all are unknown),
        currency is the cost currency code, and unknown_count is the number of
        completions whose cost could not be resolved.
    """
    if not cfg.cost.generation_info_url:
        return None, cfg.cost.cost_currency, 0
    if not completion_ids:
        return None, cfg.cost.cost_currency, 0

    cost_tasks = [fetch_cost(cid, api_key, cfg.cost) for cid in completion_ids]
    cost_results = await asyncio.gather(*cost_tasks, return_exceptions=True)

    known_costs: list[float] = []
    unknown_count = 0
    for i, c in enumerate(cost_results):
        if isinstance(c, BaseException):
            logger.warning(f"Cost fetch failed for completion {completion_ids[i]}: {c}")
            unknown_count += 1
        elif c is None:
            unknown_count += 1
        else:
            known_costs.append(c)

    known_total = sum(known_costs) if known_costs else 0.0
    if unknown_count > 0 and not known_costs:
        logger.info("Cost: UNKNOWN")
        return None, cfg.cost.cost_currency, unknown_count
    elif unknown_count > 0:
        logger.info(
            f"Cost: {known_total:.2f} {cfg.cost.cost_currency}"
            f" ({unknown_count}/{len(completion_ids)} unknown)"
        )
    else:
        logger.info(f"Cost: {known_total:.2f} {cfg.cost.cost_currency}")

    return known_total if known_costs else None, cfg.cost.cost_currency, unknown_count
