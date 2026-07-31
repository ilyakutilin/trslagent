"""Translation pipeline orchestration."""

import httpx

from src.config import Settings, logger
from src.llm import LLM, resolve_and_log_cost
from src.pipeline.concurrency import gather_chunk_results
from src.pipeline.glossary_context import GlossaryContext
from src.pipeline.result import PipelineResult, build_pipeline_result
from src.splitter import split_by_divider, split_text, stitch_chunks
from src.translator import Translator


def _build_translator(cfg: Settings, llm: LLM | None) -> Translator:
    assert cfg.input_data.source_lang is not None
    assert cfg.input_data.target_lang is not None
    return Translator(
        source_lang=cfg.input_data.source_lang,
        target_lang=cfg.input_data.target_lang,
        specialized_in=cfg.input_data.specialized_in,
        doc_type=cfg.input_data.doc_type,
        doc_title=cfg.input_data.doc_title,
        additional_instructions=cfg.input_data.additional_instructions,
        ref_source_text=cfg.input_data.ref_source_text,
        ref_target_text=cfg.input_data.ref_target_text,
        llm=llm,
    )


def _split_source_text(cfg: Settings) -> list[str]:
    text = cfg.input_data.source_text or ""
    if cfg.chunk.divider:
        chunks = split_by_divider(text=text, divider=cfg.chunk.divider)
        logger.info(
            f"Split text into {len(chunks)} chunks using divider '{cfg.chunk.divider}'"
        )
    else:
        chunks = split_text(text=text, chunk_size=cfg.chunk.size)
        logger.info(f"Split text into {len(chunks)} chunks (size={cfg.chunk.size})")
    return chunks


async def run_translation_pipeline(
    ctx: GlossaryContext,
    cfg: Settings,
    llm: LLM | None,
    http_client: httpx.AsyncClient | None = None,
) -> PipelineResult | None:
    """Run the translation pipeline over a single source text.

    Splits the source into chunks, matches glossary terms per chunk, calls
    the Translator concurrently, stitches results, and resolves costs.

    Args:
        ctx: Prepared glossary context.
        cfg: Application settings.
        llm: LLM instance, or None for print-prompt-only mode.
        http_client: Optional injected HTTP client used for cost fetches;
            when None, resolve_and_log_cost falls back to creating its own.

    Returns:
        A PipelineResult with the stitched translation and metadata, or None
        if LLM is unavailable (prompt-printing mode).
    """
    assert cfg.input_data.source_lang is not None
    assert cfg.input_data.target_lang is not None

    text = cfg.input_data.source_text or ""
    chunks = _split_source_text(cfg)
    translator = _build_translator(cfg, llm)
    is_extract = len(chunks) > 1
    auto_matched_total = [0]

    if llm is None:
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)} (length={len(chunk)})")
            _, _, user_g_str, auto_g_str = ctx.select_for_chunk(
                chunk, auto_matched_total
            )
            await translator.translate_chunk_async(
                chunk=chunk,
                user_glossary_str=user_g_str,
                auto_glossary_str=auto_g_str,
                is_extract=is_extract,
            )
        return None

    async def _translate_one(chunk: str) -> tuple[str | None, str | None]:
        _, _, user_g_str, auto_g_str = ctx.select_for_chunk(chunk, auto_matched_total)
        return await translator.translate_chunk_async(
            chunk=chunk,
            user_glossary_str=user_g_str,
            auto_glossary_str=auto_g_str,
            is_extract=is_extract,
        )

    translated_chunks, completion_ids = await gather_chunk_results(
        chunks,
        _translate_one,
        max_concurrent=cfg.chunk.max_concurrent,
        delay_seconds=cfg.chunk.delay_seconds,
        log_prefix="chunk",
    )

    logger.info(f"Stitching {len(translated_chunks)} chunks together")
    result = stitch_chunks(translated_chunks)
    logger.info(f"Translation complete: {len(text)} -> {len(result)} characters")

    cost_total, _, cost_unknowns = await resolve_and_log_cost(
        completion_ids, llm.api_key, cfg, client=http_client
    )

    return build_pipeline_result(
        text=result or "",
        source_lang=cfg.input_data.source_lang,
        target_lang=cfg.input_data.target_lang,
        source_chars=len(text),
        target_chars=len(result),
        chunk_count=len(chunks),
        model=cfg.llm.model,
        cost_total=cost_total,
        cost_currency=cfg.cost.cost_currency,
        cost_unknowns=cost_unknowns,
        auto_glossary_entries_matched=auto_matched_total[0],
        user_glossary_entries=len(ctx.user_entries),
        auto_glossary_enabled=cfg.input_data.auto_glossary,
        user_glossary_enabled=bool(cfg.input_data.user_glossary_lines),
        mode="translation",
        input_data=cfg.input_data,
    )


__all__ = ["run_translation_pipeline"]
