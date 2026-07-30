"""Deduplicate auto-matched glossary entries against user-supplied entries."""

from iso639 import Lang

from src.glossary.models import GlossaryEntry


def deduplicate_user_auto(
    matched: list[GlossaryEntry],
    user_entries: list[GlossaryEntry],
    source_lang: Lang,
) -> tuple[list[GlossaryEntry], list[GlossaryEntry]]:
    """Drop auto-matched entries whose lemmatized source term overlaps user entries.

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


__all__ = ["deduplicate_user_auto"]
