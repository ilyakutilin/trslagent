"""Convert glossary entries to newline-delimited prompt strings."""

from iso639 import Lang

from src.glossary.models import GlossaryEntry


def stringify_entries(
    entries: list[GlossaryEntry],
    source_lang: Lang,
    target_lang: Lang,
) -> str:
    """Convert glossary entries to a newline-delimited string for prompt inclusion.

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
