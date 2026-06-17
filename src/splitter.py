"""
Text Splitting Module
=====================
Splits large documents into chunks that respect sentence and paragraph boundaries.
Uses RecursiveCharacterTextSplitter for sentence-safe chunking.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str, chunk_size: int) -> list[str]:
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
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_text(text)
    return chunks


def stitch_chunks(chunks: list[str]) -> str:
    """
    Joins chunks. Since chunks are strictly non-overlapping, this simply
    joins them with a newline separator.
    """
    if not chunks:
        return ""
    return "\n".join(chunks)
