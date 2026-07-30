"""Concurrency helpers for chunk-based pipeline execution."""

import asyncio
from typing import Awaitable, Callable, TypeVar

from src.config import logger

T = TypeVar("T")


async def gather_chunk_results(
    items: list[T],
    chunk_fn: Callable[[T], Awaitable[tuple[str | None, str | None]]],
    *,
    max_concurrent: int,
    delay_seconds: float,
    log_prefix: str,
) -> tuple[list[str], list[str]]:
    """Run chunk_fn over all items concurrently with semaphore + stagger delay.

    One item failure does not abort the others (raises are returned via
    return_exceptions). Logs a warning for each failed item and returns the
    successfully produced text chunks and completion IDs.

    Args:
        items: List of inputs to feed into chunk_fn one by one.
        chunk_fn: Async function that processes a single item and returns a
            tuple of (text, completion_id). Either may be None.
        max_concurrent: Maximum number of concurrent chunk_fn invocations.
        delay_seconds: Seconds to sleep between launching consecutive items
            when max_concurrent > 1.
        log_prefix: Short noun used in logging (e.g. "chunk", "review chunk").

    Returns:
        A tuple of (completed_texts, completion_ids).
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run(i: int, item: T) -> tuple[str | None, str | None]:
        if i > 0 and max_concurrent > 1:
            await asyncio.sleep(delay_seconds)
        async with semaphore:
            logger.info(f"Processing {log_prefix} {i + 1}/{len(items)}")
            return await chunk_fn(item)

    tasks = [_run(i, item) for i, item in enumerate(items)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    completed_texts: list[str] = []
    completion_ids: list[str] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning(
                f"{log_prefix.capitalize()} {i + 1} failed: {result}, skipping"
            )
            continue
        if result is None:
            logger.warning(
                f"{log_prefix.capitalize()} {i + 1} returned None unexpectedly, skipping"
            )
            continue
        text, completion_id = result
        if text is not None:
            completed_texts.append(text)
        if completion_id:
            completion_ids.append(completion_id)

    return completed_texts, completion_ids


__all__ = ["gather_chunk_results"]
