"""
Text Splitting Module
=====================
Splits large documents into chunks that respect sentence and paragraph boundaries.
Uses RecursiveCharacterTextSplitter for sentence-safe chunking.
"""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings

# ── Configuration ────────────────────────────────────────────────────────────

settings = get_settings()

# Legacy constants for backward compatibility (deprecated, use settings object)
CHUNK_SIZE = settings.chunking.size
CHUNK_OVERLAP = settings.chunking.overlap


# ── Text splitting ────────────────────────────────────────────────────────────


def split_text(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Splits text into chunks that respect sentence and paragraph boundaries.

    Strategy:
      1. First tries to split on double newlines (paragraphs).
      2. Falls back to single newlines, then sentences (". "), then words.
    This means it will never cut in the middle of a sentence if avoidable.

    chunk_size is in characters (≈ chunk_size/4 tokens).
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        chunk_size=chunk_size * 4,  # convert approximate token count → chars
        chunk_overlap=chunk_overlap * 4,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_text(text)
    return chunks


def split_by_structure(text: str) -> list[str]:
    """
    Optional pre-split step: splits on headings or chapter markers FIRST,
    then each section is further split by split_text() if needed.
    Use this for documents with clear structural markers (chapters, sections).
    """
    # Matches common heading patterns: "Chapter 1", "## Section", "PART I", etc.
    heading_pattern = re.compile(
        r"(?m)^(?:(chapter\s+\w+|part\s+\w+|section\s+\w+|#{1,3}\s+.+))$",
        re.IGNORECASE,
    )
    sections = heading_pattern.split(text)
    all_chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        # If the section is small enough, keep as-is; otherwise chunk it
        if len(section) <= CHUNK_SIZE * 4:
            all_chunks.append(section)
        else:
            all_chunks.extend(split_text(section))
    return all_chunks
