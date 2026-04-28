import time

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from openai.types import ReasoningEffort
from openai.types.chat import ChatCompletionMessageParam


class LLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        reasoning_effort: ReasoningEffort = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self._client = None

    def _get_llm_client(self) -> OpenAI:
        if not self.api_key:
            raise ValueError("Set the OPENROUTER_API_KEY environment variable.")
        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def get_reply(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if not self._client:
            self._client = self._get_llm_client()

        # Retry logic with exponential backoff for API errors
        max_retries = 5
        base_delay = 1.0  # seconds

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
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,  # type: ignore
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError(
                        "Translation failed: received empty response from API"
                    )
                return content.strip()

            # Handle timeouts
            # (must be caught before APIConnectionError since it's a subclass)
            except APITimeoutError as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Request timed out after {max_retries} retries. "
                        f"The API is taking too long to respond.\n"
                        f"Original error: {str(e)}"
                    )
                wait_time = base_delay * (2**attempt)  # exponential backoff
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                continue

            # Handle rate limiting (429)
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Rate limit exceeded after {max_retries} retries. "
                        f"Please wait a moment and try again.\n"
                        f"Original error: {str(e)}"
                    )
                wait_time = base_delay * (2**attempt)  # exponential backoff
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                continue

            # Handle network/connection errors
            except APIConnectionError as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Network error after {max_retries} retries. "
                        f"Please check your internet connection and try again.\n"
                        f"Original error: {str(e)}"
                    )
                wait_time = base_delay * (2**attempt)
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                continue

            # Handle other errors (not retried)
            except Exception as e:
                # Re-raise any other exceptions without retrying
                raise RuntimeError(f"Translation failed: {str(e)}") from e

        # Fallback: should never reach here due to exception handling,
        # but satisfies type checker
        return ""
