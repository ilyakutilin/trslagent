from src.config import Settings, logger
from src.glossary.matcher import TermMatcher
from src.glossary.models import GlossaryEntry
from src.glossary.parser import MainGlossaryParser, ProjectGlossaryParser
from src.lemmatizer import Lemmatizer
from src.llm import LLM
from src.splitter import split_text, stitch_chunks
from src.translator import Translator

from iso639 import Lang


def _stringify_glossary(
    entries: list[GlossaryEntry],
    source_lang: Lang,
    target_lang: Lang,
) -> str:
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
    main_entries: list[GlossaryEntry] = []
    if cfg.input_data.auto_glossary:
        main_entries = MainGlossaryParser(
            dir_path=cfg.glossary.xml_dir_path,
            lemmatizer=lemmatizer,
        ).parse()

    project_entries: list[GlossaryEntry] = []
    if cfg.input_data.glossary_lines:
        project_entries = ProjectGlossaryParser(
            project_glossary_lines=cfg.input_data.glossary_lines,
            source_lang=cfg.input_data.source_lang,
            target_lang=cfg.input_data.target_lang,
            lemmatizer=lemmatizer,
        ).parse()

    return main_entries, project_entries


def _deduplicate_entries(
    matched: list[GlossaryEntry],
    project_entries: list[GlossaryEntry],
    source_lang: Lang,
) -> list[GlossaryEntry]:
    lemmatized_project_terms: list[str] = []
    for ge in project_entries:
        for term in [t for t in ge.terms if t.language == source_lang]:
            if term.lemmatized:
                lemmatized_project_terms.append(term.lemmatized)

    all_entries = project_entries.copy()
    for ge in matched:
        to_include = True
        for term in [t for t in ge.terms if t.language == source_lang]:
            if term.lemmatized in lemmatized_project_terms:
                to_include = False
        if to_include:
            all_entries.append(ge)

    return all_entries


def main(cfg: Settings) -> str | None:
    lemmatizer = Lemmatizer()

    main_glossary_entries, project_glossary_entries = _parse_glossaries(cfg, lemmatizer)

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
        raise NotImplementedError("Review functionality is not yet implemented")

    translator = Translator(
        source_lang=cfg.input_data.source_lang,
        target_lang=cfg.input_data.target_lang,
        specialized_in=cfg.input_data.specialized_in,
        doc_type=cfg.input_data.doc_type,
        doc_title=cfg.input_data.doc_title,
        llm=llm,
    )

    text = cfg.input_data.source_text or ""
    chunks = split_text(text=text, chunk_size=cfg.chunk.size)
    logger.info(f"Split text into {len(chunks)} chunks (size={cfg.chunk.size})")

    is_extract = len(chunks) > 1

    term_matcher = None
    if main_glossary_entries:
        term_matcher = TermMatcher(glossary_entries=main_glossary_entries)

    source_lang = cfg.input_data.source_lang
    target_lang = cfg.input_data.target_lang

    translated_chunks: list[str] = []

    for i, chunk in enumerate(chunks):
        logger.info(
            f"Processing chunk {i + 1}/{len(chunks)} (length={len(chunk)})"
        )

        chunk_glossary_entries: list[GlossaryEntry]
        if term_matcher is not None:
            matched = term_matcher.match(
                text=chunk,
                lang=source_lang,
                lemmatizer=lemmatizer,
            )
            chunk_glossary_entries = _deduplicate_entries(
                matched, project_glossary_entries, source_lang
            )
        else:
            chunk_glossary_entries = project_glossary_entries.copy()

        glossary_str = _stringify_glossary(
            chunk_glossary_entries, source_lang, target_lang
        )

        translated = translator.translate_chunk(
            chunk=chunk,
            glossary_str=glossary_str,
            is_extract=is_extract,
        )

        if translated is not None:
            translated_chunks.append(translated)
        elif llm is None:
            if i == len(chunks) - 1:
                return None
        else:
            logger.warning(
                f"Chunk {i+1} translation returned None unexpectedly, skipping"
            )

    logger.info(f"Stitching {len(translated_chunks)} chunks together")
    result = stitch_chunks(translated_chunks)
    logger.info(
        f"Translation complete: {len(text)} -> {len(result)} characters"
    )

    return result


def export_glossary_matches(cfg: Settings) -> str:
    lemmatizer = Lemmatizer()

    main_glossary_entries, project_glossary_entries = _parse_glossaries(cfg, lemmatizer)

    source_lang = cfg.input_data.source_lang
    target_lang = cfg.input_data.target_lang
    text = cfg.input_data.source_text or ""

    if not main_glossary_entries:
        logger.warning("No main glossary entries available for matching")
        return ""

    term_matcher = TermMatcher(glossary_entries=main_glossary_entries)

    matched = term_matcher.match(
        text=text,
        lang=source_lang,
        lemmatizer=lemmatizer,
    )

    all_entries = _deduplicate_entries(
        matched, project_glossary_entries, source_lang
    )

    logger.info(
        f"Glossary match: {len(matched)} main entries matched, "
        f"{len(project_glossary_entries)} project entries, "
        f"{len(all_entries)} total after dedup"
    )

    return _stringify_glossary(all_entries, source_lang, target_lang)
