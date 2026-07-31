"""Pipeline facade: dispatches to translation/review pipelines and exports glossary matches."""

from src.config import Settings, logger
from src.geoblock import verify_ip_not_geoblocked
from src.glossary.dedup import deduplicate_user_auto
from src.glossary.stringify import stringify_entries
from src.http_client import create_client
from src.language_detection import resolve_languages
from src.llm import LLM
from src.pipeline.glossary_context import prepare_glossary_context
from src.pipeline.result import PipelineResult
from src.pipeline.review import run_review_pipeline
from src.pipeline.translation import run_translation_pipeline


async def main(cfg: Settings) -> PipelineResult | None:
    """Run the full translation or review pipeline.

    Determines the mode (translation or review) based on whether target_text
    is set, prepares the glossary context, constructs the LLM (if not in
    print_prompt_only mode), and dispatches to the appropriate pipeline.
    A single short-lived HTTP client is created for the run's cost fetches
    and closed when the pipeline finishes. The LLM client is closed in a
    finally block, so resources are released even when the pipeline fails.
    Proxy settings from ``cfg.proxy`` are threaded to both the HTTP client
    and the LLM. When geoblocking is enabled (``cfg.geoblock.countries`` is
    non-empty) and the run is not in print_prompt_only mode, the outgoing
    IP (the proxied IP when a proxy is configured) is verified against the
    blocked countries before any LLM call.

    Args:
        cfg: Application settings controlling all pipeline behavior.

    Returns:
        A PipelineResult with the output text and metadata, or None if LLM
        is unavailable (print_prompt_only mode).

    Raises:
        ValueError: If source text is empty.
        GeoblockError: If the outgoing IP is in a blocked country or its
            geography cannot be verified (when geoblocking is enabled).
    """
    resolve_languages(cfg)
    assert cfg.input_data.source_lang is not None
    assert cfg.input_data.target_lang is not None

    if not cfg.input_data.source_text:
        raise ValueError(
            "Source text is empty. "
            "Provide it via source_file_path, source_text, or set it "
            "programmatically before calling main()."
        )

    if cfg.geoblock.countries and not cfg.output_data.print_prompt_only:
        await verify_ip_not_geoblocked(cfg.geoblock.countries, proxy_settings=cfg.proxy)

    ctx = prepare_glossary_context(cfg)

    llm: LLM | None = None
    if not cfg.output_data.print_prompt_only:
        llm = LLM(
            base_url=cfg.llm.base_url,
            api_key=cfg.llm.api_key.get_secret_value(),
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            reasoning_effort=cfg.llm.reasoning_effort,
            proxy_settings=cfg.proxy,
        )

    async with create_client(proxy_settings=cfg.proxy) as http_client:
        try:
            if cfg.input_data.target_text:
                return await run_review_pipeline(ctx, cfg, llm, http_client)
            return await run_translation_pipeline(ctx, cfg, llm, http_client)
        finally:
            if llm is not None:
                await llm.close()


def export_glossary_matches(cfg: Settings) -> str:
    """Match glossary entries against source text and return them as a formatted string.

    Used by the --match-glossary CLI subcommand.

    Args:
        cfg: Application settings with input data and glossary configuration.

    Returns:
        A newline-separated string of all matched and user glossary entries,
        or an empty string if no auto glossary is available.
    """
    resolve_languages(cfg)
    assert cfg.input_data.source_lang is not None
    assert cfg.input_data.target_lang is not None

    ctx = prepare_glossary_context(cfg)

    if ctx.term_matcher is None:
        logger.warning("No auto glossary entries available for matching")
        return ""

    text = cfg.input_data.source_text or ""
    matched = ctx.term_matcher.match(
        text=text,
        lang=ctx.source_lang,
        lemmatizer=ctx.lemmatizer,
    )

    user_entries, auto_entries = deduplicate_user_auto(
        matched, ctx.user_entries, ctx.source_lang
    )

    logger.info(
        f"Glossary match: {len(matched)} auto entries matched, "
        f"{len(ctx.user_entries)} user entries, "
        f"{len(user_entries) + len(auto_entries)} total after dedup"
    )

    return stringify_entries(
        user_entries + auto_entries, ctx.source_lang, ctx.target_lang
    )


__all__ = ["main", "export_glossary_matches", "PipelineResult"]
