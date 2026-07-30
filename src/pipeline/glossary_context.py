"""Pipeline glossary preparation: lemmatizer, parsers, matcher, per-chunk selection."""

from dataclasses import dataclass

from iso639 import Lang

from src.config import Settings
from src.glossary.dedup import deduplicate_user_auto
from src.glossary.matcher import TermMatcher
from src.glossary.models import GlossaryEntry
from src.glossary.parser import AutoGlossaryParser, UserGlossaryParser
from src.glossary.stringify import stringify_entries
from src.lemmatizer import Lemmatizer, parse_known_abbrs


@dataclass
class GlossaryContext:
    """Holds the lemmatizer, parsed glossary entries, and term matcher.

    Attributes:
        lemmatizer: Lemmatizer used for term normalization.
        auto_entries: Entries parsed from the auto glossary (XML files).
        user_entries: Entries parsed from the user-supplied glossary.
        term_matcher: Aho-Corasick matcher built from auto_entries, or None
            when no auto glossary is available.
        source_lang: Source language for the pipeline run.
        target_lang: Target language for the pipeline run.
    """

    lemmatizer: Lemmatizer
    auto_entries: list[GlossaryEntry]
    user_entries: list[GlossaryEntry]
    term_matcher: TermMatcher | None
    source_lang: Lang
    target_lang: Lang

    def select_for_chunk(
        self,
        chunk: str,
        matched_total: list[int],
    ) -> tuple[list[GlossaryEntry], list[GlossaryEntry], str, str]:
        """Match + dedup + stringify glossary entries for a single chunk.

        Args:
            chunk: Source text chunk to match terms against.
            matched_total: Single-element list used as an accumulator for
                the cumulative auto-match count across chunks. The caller
                passes a fresh list once and reads ``matched_total[0]`` after
                the pipeline completes.

        Returns:
            A tuple of (user_entries, auto_entries, user_glossary_str,
            auto_glossary_str). When no auto glossary is configured, the
            user_entries are returned without matching and auto_entries is
            empty.
        """
        if self.term_matcher is not None:
            matched = self.term_matcher.match(
                text=chunk,
                lang=self.source_lang,
                lemmatizer=self.lemmatizer,
            )
            matched_total[0] += len(matched)
            user_entries, auto_entries = deduplicate_user_auto(
                matched, self.user_entries, self.source_lang
            )
        else:
            user_entries = self.user_entries.copy()
            auto_entries = []

        user_glossary_str = stringify_entries(
            user_entries, self.source_lang, self.target_lang
        )
        auto_glossary_str = stringify_entries(
            auto_entries, self.source_lang, self.target_lang
        )
        return user_entries, auto_entries, user_glossary_str, auto_glossary_str


def _resolve_known_abbrs(cfg: Settings) -> set[str]:
    if cfg.glossary.known_abbrs_file_path:
        return parse_known_abbrs(cfg.glossary.known_abbrs_file_path)
    return set()


def _parse_glossaries(
    cfg: Settings,
    lemmatizer: Lemmatizer,
) -> tuple[list[GlossaryEntry], list[GlossaryEntry]]:
    """Parse auto and user glossaries from the given settings.

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


def prepare_glossary_context(cfg: Settings) -> GlossaryContext:
    """Build a fully prepared glossary context for the pipeline.

    Resolves known abbreviations, instantiates the lemmatizer, parses both
    auto and user glossaries, and constructs the term matcher if auto entries
    are available.

    Args:
        cfg: Application settings with input data and glossary configuration.

    Returns:
        A populated GlossaryContext. The caller is responsible for using
        ``select_for_chunk`` to acquire per-chunk glossary strings.
    """
    assert cfg.input_data.source_lang is not None
    assert cfg.input_data.target_lang is not None

    lemmatizer = Lemmatizer(known_abbrs=_resolve_known_abbrs(cfg))
    auto_entries, user_entries = _parse_glossaries(cfg, lemmatizer)

    term_matcher: TermMatcher | None = None
    if auto_entries:
        term_matcher = TermMatcher(glossary_entries=auto_entries)

    return GlossaryContext(
        lemmatizer=lemmatizer,
        auto_entries=auto_entries,
        user_entries=user_entries,
        term_matcher=term_matcher,
        source_lang=cfg.input_data.source_lang,
        target_lang=cfg.input_data.target_lang,
    )


__all__ = ["GlossaryContext", "prepare_glossary_context"]
