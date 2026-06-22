"""Pipeline orchestration: translation, review, and glossary matching.

This module ties together chunking, glossary matching, LLM prompting,
and cost resolution into a unified async pipeline driven by TOML config.
"""

import asyncio
from dataclasses import dataclass

from src.config import Settings, logger
from src.glossary.matcher import TermMatcher
from src.glossary.models import GlossaryEntry
from src.glossary.parser import AutoGlossaryParser, UserGlossaryParser
from src.language_detection import resolve_languages
from src.lemmatizer import Lemmatizer
from src.llm import LLM, fetch_cost
from src.reviewer import Reviewer
from src.splitter import split_by_divider, split_text, stitch_chunks
from src.translator import Translator

from iso639 import Lang


@dataclass
class PipelineResult:
    """Holds the result of a translation or review pipeline run.

    Attributes:
        text: The final output text (translation or review feedback).
        source_lang: Source language of the original text.
        target_lang: Target language of the output.
        source_chars: Character count of source text.
        target_chars: Character count of output text.
        chunk_count: Number of chunks the text was split into.
        model: LLM model identifier used.
        cost_total: Total cost of the run, or None if unavailable.
        cost_currency: Currency code for the cost (e.g. "USD").
        cost_unknowns: Number of chunks whose cost could not be resolved.
        auto_glossary_entries_matched: Total auto-glossary entries matched across chunks.
        user_glossary_entries: Number of user-supplied glossary entries.
        specialized_in: Optional domain specialization.
        doc_type: Optional document type.
        doc_title: Optional document title.
        additional_instructions: Optional extra instructions for the LLM.
        auto_glossary_enabled: Whether auto glossary matching was active.
        user_glossary_enabled: Whether a user glossary was provided.
        mode: Pipeline mode — "translation" or "review".
    """

    text: str
    source_lang: Lang
    target_lang: Lang
    source_chars: int
    target_chars: int
    chunk_count: int
    model: str
    cost_total: float | None
    cost_currency: str
    cost_unknowns: int
    auto_glossary_entries_matched: int
    user_glossary_entries: int
    specialized_in: str | None
    doc_type: str | None
    doc_title: str | None
    additional_instructions: str | None
    auto_glossary_enabled: bool
    user_glossary_enabled: bool
    mode: str


def _stringify_glossary(
    entries: list[GlossaryEntry],
    source_lang: Lang,
    target_lang: Lang,
) -> str:
    """Converts glossary entries to a newline-delimited string for prompt inclusion.

    Args:
        entries: List of GlossaryEntry objects to stringify.
        source_lang: Source language for term extraction.
        target_lang: Target language for translation extraction.

    Returns:
        A newline-separated string of term-to-translation mappings.
    """
    str_entries: list[str] = []
    for entry in entries:
        str_entry = entry.stringify(source_lang, target_lang)
        if str_entry is not None:
            str_entries.append(str_entry)

    return "\n".join(str_entries)


def _parse_glossaries(
    cfg: Settings,
    lemmatizer: Lemmatizer,
) -> tuple[list[GlossaryEntry], list[GlossaryEntry]]:
    """Parses auto and user glossaries from the given settings.

    Args:
        cfg: Application settings with glossary configuration.
        lemmatizer: Lemmatizer instance for term normalization.

    Returns:
        A tuple of (auto_entries, user_entries).
    """
    auto_entries: list[GlossaryEntry] = []
    if cfg.input_data.auto_glossary:
        auto_entries = AutoGlossaryParser(
            dir_path=cfg.glossary.xml_dir_path,
            lemmatizer=lemmatizer,
        ).parse()

    user_entries: list[GlossaryEntry] = []
    if cfg.input_data.user_glossary_lines:
        assert cfg.input_data.source_lang is not None
        assert cfg.input_data.target_lang is not None
        user_entries = UserGlossaryParser(
            user_glossary_lines=cfg.input_data.user_glossary_lines,
            source_lang=cfg.input_data.source_lang,
            target_lang=cfg.input_data.target_lang,
            lemmatizer=lemmatizer,
        ).parse()

    return auto_entries, user_entries


def _deduplicate_entries(
    matched: list[GlossaryEntry],
    user_entries: list[GlossaryEntry],
    source_lang: Lang,
) -> tuple[list[GlossaryEntry], list[GlossaryEntry]]:
    """Removes auto-matched entries that overlap with user-supplied glossary entries.

    Args:
        matched: Auto-matched glossary entries from the source text.
        user_entries: User-supplied glossary entries (take precedence).
        source_lang: Source language for comparing lemmatized terms.

    Returns:
        A tuple of (deduped_user_entries, deduped_auto_entries).
    """
    lemmatized_user_terms: list[str] = []
    for ge in user_entries:
        for term in [t for t in ge.terms if t.language == source_lang]:
            if term.lemmatized:
                lemmatized_user_terms.append(term.lemmatized)

    auto_only: list[GlossaryEntry] = []
    for ge in matched:
        to_include = True
        for term in [t for t in ge.terms if t.language == source_lang]:
            if term.lemmatized in lemmatized_user_terms:
                to_include = False
        if to_include:
            auto_only.append(ge)

    return user_entries.copy(), auto_only


async def _resolve_and_log_cost(
    completion_ids: list[str],
    api_key: str,
    cfg: Settings,
) -> tuple[float | None, str, int]:
    """Fetches and logs the total cost of all LLM completions.

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


async def main(cfg: Settings) -> PipelineResult | None:
    """Runs the full translation or review pipeline.

    Determines the mode (translation or review) based on whether target_text is set,
    splits text into chunks, runs the LLM with glossary awareness, stitches results,
    and resolves costs.

    Args:
        cfg: Application settings controlling all pipeline behavior.

    Returns:
        A PipelineResult with the output text and metadata, or None if LLM is unavailable
        (print_prompt_only mode).

    Raises:
        ValueError: If source text is empty or chunk counts mismatch in review mode.
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

    lemmatizer = Lemmatizer()

    auto_glossary_entries, user_glossary_entries = _parse_glossaries(cfg, lemmatizer)

    auto_glossary_enabled = cfg.input_data.auto_glossary
    user_glossary_enabled = bool(cfg.input_data.user_glossary_lines)
    auto_matched_total = 0
    user_glossary_entry_count = len(user_glossary_entries)

    llm = None
    if not cfg.output_data.print_prompt_only:
        llm = LLM(
            base_url=cfg.llm.base_url,
            api_key=cfg.llm.api_key.get_secret_value(),
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            reasoning_effort=cfg.llm.reasoning_effort,
        )

    if cfg.input_data.target_text:
        reviewer = Reviewer(
            source_lang=cfg.input_data.source_lang,
            target_lang=cfg.input_data.target_lang,
            specialized_in=cfg.input_data.specialized_in,
            doc_type=cfg.input_data.doc_type,
            doc_title=cfg.input_data.doc_title,
            additional_instructions=cfg.input_data.additional_instructions,
            llm=llm,
        )

        source_lang = cfg.input_data.source_lang
        target_lang = cfg.input_data.target_lang

        source_text = cfg.input_data.source_text or ""
        target_text = cfg.input_data.target_text or ""

        if cfg.chunk.divider:
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

            term_matcher = None
            if auto_glossary_entries:
                term_matcher = TermMatcher(glossary_entries=auto_glossary_entries)

            def _glossary_for_review_chunk(
                chunk: str,
            ) -> tuple[list[GlossaryEntry], list[GlossaryEntry]]:
                """Matches glossary terms in a review-mode source chunk.

                Args:
                    chunk: Source text chunk to match terms against.

                Returns:
                    A tuple of (user_entries, auto_entries) after deduplication.
                """
                nonlocal auto_matched_total
                if term_matcher is not None:
                    matched = term_matcher.match(
                        text=chunk, lang=source_lang, lemmatizer=lemmatizer
                    )
                    auto_matched_total += len(matched)
                    return _deduplicate_entries(
                        matched, user_glossary_entries, source_lang
                    )
                return user_glossary_entries.copy(), []

            review_results: list[str] = []
            completion_ids: list[str] = []

            if llm is None:
                for i, (src, tgt) in enumerate(zip(src_chunks, tgt_chunks)):
                    logger.info(
                        f"Processing review chunk {i + 1}/{len(src_chunks)} "
                        f"(src_length={len(src)}, tgt_length={len(tgt)})"
                    )
                    chunk_user_entries, chunk_auto_entries = _glossary_for_review_chunk(
                        src
                    )
                    user_g_str = _stringify_glossary(
                        chunk_user_entries, source_lang, target_lang
                    )
                    auto_g_str = _stringify_glossary(
                        chunk_auto_entries, source_lang, target_lang
                    )
                    await reviewer.review_text_async(src, tgt, user_g_str, auto_g_str)
                return None

            semaphore = asyncio.Semaphore(cfg.chunk.max_concurrent)

            async def _process_single_review_chunk(i, src_chunk, tgt_chunk):
                """Processes a single review chunk pair with concurrency control.

                Args:
                    i: Chunk index for logging and delay decisions.
                    src_chunk: Source text chunk.
                    tgt_chunk: Target translation chunk.

                Returns:
                    The result of reviewer.review_text_async for this chunk pair.
                """
                if i > 0 and cfg.chunk.max_concurrent > 1:
                    await asyncio.sleep(cfg.chunk.delay_seconds)
                async with semaphore:
                    logger.info(
                        f"Reviewing chunk {i + 1}/{len(src_chunks)} "
                        f"(src_length={len(src_chunk)}, tgt_length={len(tgt_chunk)})"
                    )
                    chunk_user_entries, chunk_auto_entries = _glossary_for_review_chunk(
                        src_chunk
                    )
                    user_g_str = _stringify_glossary(
                        chunk_user_entries, source_lang, target_lang
                    )
                    auto_g_str = _stringify_glossary(
                        chunk_auto_entries, source_lang, target_lang
                    )
                    return await reviewer.review_text_async(
                        src_chunk, tgt_chunk, user_g_str, auto_g_str
                    )

            tasks = [
                _process_single_review_chunk(i, src, tgt)
                for i, (src, tgt) in enumerate(zip(src_chunks, tgt_chunks))
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    logger.warning(f"Review chunk {i + 1} failed: {result}, skipping")
                elif result is not None:
                    review_text, chunk_id = result
                    if review_text is not None:
                        review_results.append(review_text)
                    if chunk_id:
                        completion_ids.append(chunk_id)

            result = stitch_chunks(review_results)

            logger.info(
                f"Review complete: "
                f"source={len(source_text)} chars, "
                f"target={len(target_text)} chars, "
                f"result={len(result) if result else 0} chars"
            )

            api_key = cfg.llm.api_key.get_secret_value()
            cost_total, _, cost_unknowns = await _resolve_and_log_cost(
                completion_ids, api_key, cfg
            )

            return PipelineResult(
                text=result or "",
                source_lang=source_lang,
                target_lang=target_lang,
                source_chars=len(source_text),
                target_chars=len(target_text),
                chunk_count=len(src_chunks),
                model=cfg.llm.model,
                cost_total=cost_total,
                cost_currency=cfg.cost.cost_currency,
                cost_unknowns=cost_unknowns,
                auto_glossary_entries_matched=auto_matched_total,
                user_glossary_entries=user_glossary_entry_count,
                specialized_in=cfg.input_data.specialized_in,
                doc_type=cfg.input_data.doc_type,
                doc_title=cfg.input_data.doc_title,
                additional_instructions=cfg.input_data.additional_instructions,
                auto_glossary_enabled=auto_glossary_enabled,
                user_glossary_enabled=user_glossary_enabled,
                mode="review",
            )

        if auto_glossary_entries:
            term_matcher = TermMatcher(glossary_entries=auto_glossary_entries)
            matched = term_matcher.match(
                text=source_text,
                lang=source_lang,
                lemmatizer=lemmatizer,
            )
            auto_matched_total = len(matched)
            review_user_entries, review_auto_entries = _deduplicate_entries(
                matched, user_glossary_entries, source_lang
            )
        else:
            review_user_entries = user_glossary_entries.copy()
            review_auto_entries = []

        user_glossary_str = _stringify_glossary(
            review_user_entries, source_lang, target_lang
        )
        auto_glossary_str = _stringify_glossary(
            review_auto_entries, source_lang, target_lang
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

        api_key = cfg.llm.api_key.get_secret_value() if llm else ""

        logger.info(
            f"Review complete: "
            f"source={len(source_text)} chars, "
            f"target={len(target_text)} chars, "
            f"result={len(result) if result else 0} chars"
        )

        cost_total, _, cost_unknowns = await _resolve_and_log_cost(
            completion_ids, api_key, cfg
        )

        return PipelineResult(
            text=result or "",
            source_lang=source_lang,
            target_lang=target_lang,
            source_chars=len(source_text),
            target_chars=len(target_text),
            chunk_count=1,
            model=cfg.llm.model,
            cost_total=cost_total,
            cost_currency=cfg.cost.cost_currency,
            cost_unknowns=cost_unknowns,
            auto_glossary_entries_matched=auto_matched_total,
            user_glossary_entries=user_glossary_entry_count,
            specialized_in=cfg.input_data.specialized_in,
            doc_type=cfg.input_data.doc_type,
            doc_title=cfg.input_data.doc_title,
            additional_instructions=cfg.input_data.additional_instructions,
            auto_glossary_enabled=auto_glossary_enabled,
            user_glossary_enabled=user_glossary_enabled,
            mode="review",
        )

    translator = Translator(
        source_lang=cfg.input_data.source_lang,
        target_lang=cfg.input_data.target_lang,
        specialized_in=cfg.input_data.specialized_in,
        doc_type=cfg.input_data.doc_type,
        doc_title=cfg.input_data.doc_title,
        additional_instructions=cfg.input_data.additional_instructions,
        llm=llm,
    )

    text = cfg.input_data.source_text or ""
    if cfg.chunk.divider:
        chunks = split_by_divider(text=text, divider=cfg.chunk.divider)
        logger.info(
            f"Split text into {len(chunks)} chunks using divider '{cfg.chunk.divider}'"
        )
    else:
        chunks = split_text(text=text, chunk_size=cfg.chunk.size)
        logger.info(f"Split text into {len(chunks)} chunks (size={cfg.chunk.size})")

    is_extract = len(chunks) > 1

    term_matcher = None
    if auto_glossary_entries:
        term_matcher = TermMatcher(glossary_entries=auto_glossary_entries)

    source_lang = cfg.input_data.source_lang
    target_lang = cfg.input_data.target_lang

    translated_chunks: list[str] = []

    def _get_chunk_glossary(
        chunk: str,
        term_matcher: TermMatcher | None,
        user_entries: list[GlossaryEntry],
        source_lang: Lang,
        lemmatizer: Lemmatizer,
    ) -> tuple[list[GlossaryEntry], list[GlossaryEntry]]:
        """Matches glossary terms in a translation-mode text chunk.

        Args:
            chunk: Source text chunk to match terms against.
            term_matcher: TermMatcher instance, or None if auto glossary is disabled.
            user_entries: User-supplied glossary entries.
            source_lang: Source language for matching.
            lemmatizer: Lemmatizer for term normalization.

        Returns:
            A tuple of (user_entries, auto_entries) after deduplication.
        """
        nonlocal auto_matched_total
        if term_matcher is not None:
            matched = term_matcher.match(
                text=chunk,
                lang=source_lang,
                lemmatizer=lemmatizer,
            )
            auto_matched_total += len(matched)
            return _deduplicate_entries(matched, user_entries, source_lang)
        return user_entries.copy(), []

    if llm is None:
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)} (length={len(chunk)})")
            chunk_user_entries, chunk_auto_entries = _get_chunk_glossary(
                chunk, term_matcher, user_glossary_entries, source_lang, lemmatizer
            )
            user_glossary_str = _stringify_glossary(
                chunk_user_entries, source_lang, target_lang
            )
            auto_glossary_str = _stringify_glossary(
                chunk_auto_entries, source_lang, target_lang
            )
            await translator.translate_chunk_async(
                chunk=chunk,
                user_glossary_str=user_glossary_str,
                auto_glossary_str=auto_glossary_str,
                is_extract=is_extract,
            )
        return None

    semaphore = asyncio.Semaphore(cfg.chunk.max_concurrent)

    async def _process_single_chunk(
        i: int, chunk: str
    ) -> tuple[str | None, str | None]:
        """Processes a single translation chunk with concurrency control.

        Args:
            i: Chunk index for logging and delay decisions.
            chunk: Source text chunk to translate.

        Returns:
            The result of translator.translate_chunk_async for this chunk.
        """
        if i > 0 and cfg.chunk.max_concurrent > 1:
            await asyncio.sleep(cfg.chunk.delay_seconds)

        async with semaphore:
            logger.info(f"Processing chunk {i + 1}/{len(chunks)} (length={len(chunk)})")

            chunk_user_entries, chunk_auto_entries = _get_chunk_glossary(
                chunk, term_matcher, user_glossary_entries, source_lang, lemmatizer
            )
            user_glossary_str = _stringify_glossary(
                chunk_user_entries, source_lang, target_lang
            )
            auto_glossary_str = _stringify_glossary(
                chunk_auto_entries, source_lang, target_lang
            )

            return await translator.translate_chunk_async(
                chunk=chunk,
                user_glossary_str=user_glossary_str,
                auto_glossary_str=auto_glossary_str,
                is_extract=is_extract,
            )

    tasks = [_process_single_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    completion_ids: list[str] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning(f"Chunk {i + 1} failed with error: {result}, skipping")
        elif result is not None:
            chunk_text, chunk_id = result
            if chunk_text is not None:
                translated_chunks.append(chunk_text)
            if chunk_id:
                completion_ids.append(chunk_id)
        else:
            logger.warning(
                f"Chunk {i + 1} translation returned None unexpectedly, skipping"
            )

    logger.info(f"Stitching {len(translated_chunks)} chunks together")
    result = stitch_chunks(translated_chunks)
    logger.info(f"Translation complete: {len(text)} -> {len(result)} characters")

    api_key = cfg.llm.api_key.get_secret_value()
    cost_total, _, cost_unknowns = await _resolve_and_log_cost(
        completion_ids, api_key, cfg
    )

    return PipelineResult(
        text=result or "",
        source_lang=source_lang,
        target_lang=target_lang,
        source_chars=len(text),
        target_chars=len(result),
        chunk_count=len(chunks),
        model=cfg.llm.model,
        cost_total=cost_total,
        cost_currency=cfg.cost.cost_currency,
        cost_unknowns=cost_unknowns,
        auto_glossary_entries_matched=auto_matched_total,
        user_glossary_entries=user_glossary_entry_count,
        specialized_in=cfg.input_data.specialized_in,
        doc_type=cfg.input_data.doc_type,
        doc_title=cfg.input_data.doc_title,
        additional_instructions=cfg.input_data.additional_instructions,
        auto_glossary_enabled=auto_glossary_enabled,
        user_glossary_enabled=user_glossary_enabled,
        mode="translation",
    )


def export_glossary_matches(cfg: Settings) -> str:
    """Matches glossary entries against source text and returns them as a formatted string.

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

    lemmatizer = Lemmatizer()

    auto_glossary_entries, user_glossary_entries = _parse_glossaries(cfg, lemmatizer)

    source_lang = cfg.input_data.source_lang
    target_lang = cfg.input_data.target_lang
    text = cfg.input_data.source_text or ""

    if not auto_glossary_entries:
        logger.warning("No auto glossary entries available for matching")
        return ""

    term_matcher = TermMatcher(glossary_entries=auto_glossary_entries)

    matched = term_matcher.match(
        text=text,
        lang=source_lang,
        lemmatizer=lemmatizer,
    )

    user_entries, auto_entries = _deduplicate_entries(
        matched, user_glossary_entries, source_lang
    )

    logger.info(
        f"Glossary match: {len(matched)} auto entries matched, "
        f"{len(user_glossary_entries)} user entries, "
        f"{len(user_entries) + len(auto_entries)} total after dedup"
    )

    all_entries = user_entries + auto_entries
    return _stringify_glossary(all_entries, source_lang, target_lang)
