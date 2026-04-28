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


def _find_seam(tail: str, head: str) -> int:
    """
    Returns the character offset in `head` where unique content begins,
    i.e. skips any content that is duplicated from `tail`.
    Tries progressively smaller suffixes of `tail` against the prefix of `head`.
    """
    words = tail.split()
    # Try matching the last N words of tail against start of head
    for n in range(min(20, len(words)), 2, -1):
        snippet = " ".join(words[-n:])
        idx = head.find(snippet)
        if idx != -1:
            return idx + len(snippet)
    return 0  # no overlap found, just concatenate


def truncate_at_sentence_boundary(text: str, window: int = 400) -> str:
    """
    Truncates text at the last sentence or paragraph boundary within a character window.

    Args:
        text: The text to potentially truncate
        window: Maximum characters to consider from the end

    Returns:
        Text truncated at the last sentence/paragraph boundary, or up to `window`
        characters if no boundary is found.
    """
    # Take the last N characters to consider
    if len(text) <= window:
        return text.strip()

    tail = text[-window:]

    # Sentence boundaries: period, exclamation mark, question mark, or double newline
    boundary_markers = [(". ", "! ", "? "), ("\n\n", "\n\n"), "\n"]

    # Find the last occurrence of any boundary marker
    last_boundary_idx = -1
    for marker in boundary_markers:
        idx = tail.rfind(marker)
        if idx != -1:
            last_boundary_idx = max(last_boundary_idx, idx + len(marker))

    # If we found a boundary, truncate there; otherwise use full window
    if last_boundary_idx > 0:
        truncated = text[-window:-last_boundary_idx]
    else:
        truncated = tail.strip()

    return truncated


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
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


def stitch_chunks(chunks: list[str], chunk_overlap: int) -> str:
    """
    Joins chunks. Because of overlap, the tail of chunk N and the
    head of chunk N+1 may be near-duplicate sentences. This does a simple
    deduplication heuristic at the seams.
    """
    overlap_chars = chunk_overlap * 4
    if not chunks:
        return ""
    result = chunks[0]
    for chunk in chunks[1:]:
        # Find the longest suffix of `result` that appears at start of `chunk`
        overlap_window = result[-overlap_chars:]
        seam = _find_seam(overlap_window, chunk)
        result += "\n" + chunk[seam:]
    return result
