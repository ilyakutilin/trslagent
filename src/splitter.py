"""
Text Splitting Module
=====================
Splits large documents into chunks that respect sentence and paragraph boundaries.
Uses RecursiveCharacterTextSplitter for sentence-safe chunking.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings

# ── Configuration ────────────────────────────────────────────────────────────

settings = get_settings()

# Legacy constants for backward compatibility (deprecated, use settings object)
CHUNK_SIZE = settings.chunk.size
CHUNK_OVERLAP = settings.chunk.overlap


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

    chunk_size is in characters.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_text(text)
    return chunks
