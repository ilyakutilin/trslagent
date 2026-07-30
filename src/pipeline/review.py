"""Review pipeline orchestration."""

from src.config import Settings, logger
from src.llm import LLM, resolve_and_log_cost
from src.pipeline.concurrency import gather_chunk_results
from src.pipeline.glossary_context import GlossaryContext
from src.pipeline.result import PipelineResult, build_pipeline_result
from src.reviewer import Reviewer
from src.splitter import split_by_divider, stitch_chunks


def _build_reviewer(cfg: Settings, llm: LLM | None) -> Reviewer:
    assert cfg.input_data.source_lang is not None
    assert cfg.input_data.target_lang is not None
    return Reviewer(
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


async def _run_review_divider(
    ctx: GlossaryContext,
    cfg: Settings,
    reviewer: Reviewer,
    llm: LLM | None,
) -> PipelineResult | None:
    """Run a divider-based chunked review pipeline.

    Splits source and target texts on the divider, asserts the chunk counts
    match, and processes pairs concurrently (or serially in prompt-print mode).
    """
    assert cfg.input_data.source_lang is not None
    assert cfg.input_data.target_lang is not None
    assert cfg.chunk.divider is not None

    source_text = cfg.input_data.source_text or ""
    target_text = cfg.input_data.target_text or ""

    src_chunks = split_by_divider(text=source_text, divider=cfg.chunk.divider)
    tgt_chunks = split_by_divider(text=target_text, divider=cfg.chunk.divider)
    logger.info(
        f"Split source into {len(src_chunks)} chunks, "
        f"target into {len(tgt_chunks)} chunks "
        f"using divider '{cfg.chunk.divider}'"
    )

    if len(src_chunks) != len(tgt_chunks):
        raise ValueError(
            f"Manual chunk count mismatch in review mode: "
            f"source has {len(src_chunks)} chunks, "
            f"target has {len(tgt_chunks)} chunks. "
            f"Chunk counts must be equal."
        )

    auto_matched_total = [0]

    if llm is None:
        for i, (src, tgt) in enumerate(zip(src_chunks, tgt_chunks)):
            logger.info(
                f"Processing review chunk {i + 1}/{len(src_chunks)} "
                f"(src_length={len(src)}, tgt_length={len(tgt)})"
            )
            _, _, user_g_str, auto_g_str = ctx.select_for_chunk(src, auto_matched_total)
            await reviewer.review_text_async(src, tgt, user_g_str, auto_g_str)
        return None

    async def _review_one(
        pair: tuple[str, str],
    ) -> tuple[str | None, str | None]:
        src, tgt = pair
        _, _, user_g_str, auto_g_str = ctx.select_for_chunk(src, auto_matched_total)
        return await reviewer.review_text_async(src, tgt, user_g_str, auto_g_str)

    review_texts, completion_ids = await gather_chunk_results(
        list(zip(src_chunks, tgt_chunks)),
        _review_one,
        max_concurrent=cfg.chunk.max_concurrent,
        delay_seconds=cfg.chunk.delay_seconds,
        log_prefix="review chunk",
    )

    result = stitch_chunks(review_texts)
    logger.info(
        f"Review complete: "
        f"source={len(source_text)} chars, "
        f"target={len(target_text)} chars, "
        f"result={len(result) if result else 0} chars"
    )

    cost_total, _, cost_unknowns = await resolve_and_log_cost(
        completion_ids, llm.api_key, cfg
    )

    return build_pipeline_result(
        text=result or "",
        source_lang=cfg.input_data.source_lang,
        target_lang=cfg.input_data.target_lang,
        source_chars=len(source_text),
        target_chars=len(target_text),
        chunk_count=len(src_chunks),
        model=cfg.llm.model,
        cost_total=cost_total,
        cost_currency=cfg.cost.cost_currency,
        cost_unknowns=cost_unknowns,
        auto_glossary_entries_matched=auto_matched_total[0],
        user_glossary_entries=len(ctx.user_entries),
        auto_glossary_enabled=cfg.input_data.auto_glossary,
        user_glossary_enabled=bool(cfg.input_data.user_glossary_lines),
        mode="review",
        input_data=cfg.input_data,
    )


async def _run_review_full(
    ctx: GlossaryContext,
    cfg: Settings,
    reviewer: Reviewer,
    llm: LLM | None,
) -> PipelineResult | None:
    """Run a non-chunked review pipeline over the full source/target texts."""
    assert cfg.input_data.source_lang is not None
    assert cfg.input_data.target_lang is not None

    source_text = cfg.input_data.source_text or ""
    target_text = cfg.input_data.target_text or ""

    auto_matched_total = [0]
    _, _, user_glossary_str, auto_glossary_str = ctx.select_for_chunk(
        source_text, auto_matched_total
    )

    result, completion_id = await reviewer.review_text_async(
        source_text=source_text,
        target_text=target_text,
        user_glossary_str=user_glossary_str,
        auto_glossary_str=auto_glossary_str,
    )

    completion_ids: list[str] = []
    if completion_id:
        completion_ids.append(completion_id)

    logger.info(
        f"Review complete: "
        f"source={len(source_text)} chars, "
        f"target={len(target_text)} chars, "
        f"result={len(result) if result else 0} chars"
    )

    cost_total, _, cost_unknowns = await resolve_and_log_cost(
        completion_ids, cfg.llm.api_key.get_secret_value(), cfg
    )

    return build_pipeline_result(
        text=result or "",
        source_lang=cfg.input_data.source_lang,
        target_lang=cfg.input_data.target_lang,
        source_chars=len(source_text),
        target_chars=len(target_text),
        chunk_count=1,
        model=cfg.llm.model,
        cost_total=cost_total,
        cost_currency=cfg.cost.cost_currency,
        cost_unknowns=cost_unknowns,
        auto_glossary_entries_matched=auto_matched_total[0],
        user_glossary_entries=len(ctx.user_entries),
        auto_glossary_enabled=cfg.input_data.auto_glossary,
        user_glossary_enabled=bool(cfg.input_data.user_glossary_lines),
        mode="review",
        input_data=cfg.input_data,
    )


async def run_review_pipeline(
    ctx: GlossaryContext,
    cfg: Settings,
    llm: LLM | None,
) -> PipelineResult | None:
    """Run the review pipeline over a source/target pair.

    Dispatches to the divider-based chunked path or the full-text path based
    on ``cfg.chunk.divider``.

    Args:
        ctx: Prepared glossary context.
        cfg: Application settings.
        llm: LLM instance, or None for print-prompt-only mode.

    Returns:
        A PipelineResult with the stitched review and metadata, or None if
        LLM is unavailable (prompt-printing mode).
    """
    reviewer = _build_reviewer(cfg, llm)
    if cfg.chunk.divider:
        return await _run_review_divider(ctx, cfg, reviewer, llm)
    return await _run_review_full(ctx, cfg, reviewer, llm)


__all__ = ["run_review_pipeline"]
