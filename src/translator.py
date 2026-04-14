"""
AI Translation Agent
====================
Translates large documents using:
- RecursiveCharacterTextSplitter for sentence-safe chunking
- ChromaDB + multilingual-e5-large for glossary RAG
- OpenRouter API for LLM translation
"""

# TODO: Implement logging instead of print()

# TODO: No parallel/batch translation
# Chunks are translated sequentially.
# For large documents, concurrent requests (e.g., concurrent.futures.ThreadPoolExecutor)
# could significantly speed things up, with configurable max workers.

# TODO: No progress resumption for long documents
# If the process crashes at chunk 50/100, there's no way to resume.
# Fix: save translated chunks incrementally to a temp file and support --resume.

# TODO: Better seam detection in stitch_chunks()
# The word-based overlap heuristic is fragile — common phrases can produce false matches
# Consider using difflib.SequenceMatcher for more robust overlap detection.

# TODO: .gitignore` doesn't include chroma_db/
# The ChromaDB persistence directory (./chroma_db/) should be in .gitignore.

# TODO: main.py is unused boilerplate
# The main.py at project root just prints "Hello from trslagent!" and doesn't reference
# the actual translation functionality. It should either be the CLI entry point
# or be removed.

# TODO: README CLI examples don't match src/ layout
# README shows python translator.py but the file is at src/translator.py.
# The Python examples use top-level imports that won't work.
# Fix: update README to match actual structure, or add a proper CLI entry point
# via pyproject.toml [project.scripts].

# TODO: Configurable e5 prefix for alternative embedding models
# The "query: " / "passage: " prefixes are specific to e5 models and hardcoded.
# If someone swaps the model, these prefixes would be wrong.
# Fix: make prefixes configurable or auto-detect based on model name.

import os
import re
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI  # OpenRouter is OpenAI-compatible
from openai.types.chat import ChatCompletionMessageParam

from src.glossary_manager import GlossaryManager

# ── Configuration ────────────────────────────────────────────────────────────

# TODO: Fix this: OPENROUTER_API_KEY captured at module import time
# OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "") is evaluated once
# at import. If the env var is set after import, it won't be picked up.
# Fix: read it lazily inside get_llm_client().

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"  # change to any OpenRouter model
# TODO: Fix this: CHUNK_SIZE naming is misleading
# CHUNK_SIZE = 1500 is described as "tokens" but split_text() multiplies by 4
# to convert to characters. This 4× ratio is wrong for CJK text (~1–2 chars/token)
# and many European languages with longer words.
# Fix: rename to something like CHUNK_SIZE_CHARS and set it directly in characters,
# or use a proper tokenizer (e.g., tiktoken) for accurate token counting.

CHUNK_SIZE = 1500  # tokens (approximate; splitter works in chars, ~4 chars/token)
CHUNK_OVERLAP = 150  # overlap between chunks to avoid losing context at seams
GLOSSARY_TOP_K = 20  # how many glossary terms to retrieve per chunk


# ── OpenRouter client (OpenAI-compatible) ────────────────────────────────────


def get_llm_client() -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError("Set the OPENROUTER_API_KEY environment variable.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


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


# ── Translation prompt ────────────────────────────────────────────────────────


def build_prompt(
    chunk: str,
    source_lang: str,
    target_lang: str,
    glossary_terms: list[dict],
    previous_translated: Optional[str] = None,
) -> str:
    """
    Builds the translation prompt with:
    - Mandatory glossary terms (retrieved via RAG)
    - Optional previous segment tail for context continuity
    """
    # TODO: Use system message for LLM prompt structure
    # Currently the entire prompt goes as a single user message.
    # Separating instructions (system) from content (user) is more idiomatic
    # for chat models and typically improves instruction-following.
    glossary_section = ""
    if glossary_terms:
        lines = [f"  {t['source']} → {t['target']}" for t in glossary_terms]
        # TODO: Fix this: All retrieved glossary terms are marked "MANDATORY" even when irrelevant
        # Retrieving top-20 terms by embedding similarity doesn't guarantee they appear
        # in the chunk. The prompt says "you MUST use these" for terms that aren't even
        # in the text — wasting tokens and potentially confusing the model.
        # Fix: filter terms to only include those that actually appear (even partially)
        # in the chunk text, or add a relevance threshold on the distance score.

        glossary_section = (
            "\n\nMANDATORY GLOSSARY (you MUST use these translations exactly as given, "
            "do not paraphrase or substitute them):\n" + "\n".join(lines)
        )

    context_section = ""
    if previous_translated:
        # Pass only the tail of the previous chunk to save tokens
        # TODO: Fix this: previous_translated truncation is crude
        # tail = previous_translated[-400:].strip() cuts at an arbitrary character
        # boundary — potentially mid-word or mid-sentence.
        # Fix: truncate at the last sentence boundary (e.g., last `.` or `\n`)
        # within the window.

        tail = previous_translated[-400:].strip()
        context_section = f"\n\nPREVIOUS SEGMENT (for context and style continuity — do NOT retranslate this):\n{tail}"

    prompt = (
        f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}.\n"
        f"Rules:\n"
        f"  1. Use the mandatory glossary terms exactly as specified.\n"
        f"  2. Preserve formatting, paragraph breaks, and punctuation.\n"
        f"  3. Do not add explanations or commentary — output ONLY the translated text.\n"
        f"  4. Maintain the tone and style of the original."
        f"{glossary_section}"
        f"{context_section}"
        f"\n\nTEXT TO TRANSLATE:\n{chunk}"
    )
    return prompt


# ── Core translation logic ────────────────────────────────────────────────────


def translate_chunk(
    client: OpenAI,
    chunk: str,
    source_lang: str,
    target_lang: str,
    glossary_terms: list[dict],
    previous_translated: Optional[str],
    model: str = DEFAULT_MODEL,
) -> str:
    # TODO: Fix this: No error handling / retry for OpenRouter API calls
    # translate_chunk() has no try/except. Network errors, rate limits (429),
    # or API timeouts will crash the entire multi-chunk process with no way to resume.
    # Fix: add retry with exponential backoff (e.g., tenacity or a simple loop)
    # and meaningful error messages.

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are a professional translator. Translate text from "
                       f"{source_lang} to {target_lang}.\n"
                       "Rules:\n"
                       "  1. Use the mandatory glossary terms exactly as specified.\n"
                       "  2. Preserve formatting, paragraph breaks, and punctuation.\n"
                       "  3. Do not add explanations or commentary — output ONLY "
                       "the translated text.\n"
                       "  4. Maintain the tone and style of the original.",
        },
        {
            "role": "user",
            "content": build_prompt(
                chunk, source_lang, target_lang, glossary_terms, previous_translated
            ),
        },
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,  # low temperature = more consistent/literal translation
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Translation failed: received empty response from API")
    return content.strip()


def stitch_chunks(
    translated_chunks: list[str], overlap_chars: int = CHUNK_OVERLAP * 4
) -> str:
    """
    Joins translated chunks. Because of overlap, the tail of chunk N and the
    head of chunk N+1 may be near-duplicate sentences. This does a simple
    deduplication heuristic at the seams.
    """
    if not translated_chunks:
        return ""
    result = translated_chunks[0]
    for chunk in translated_chunks[1:]:
        # Find the longest suffix of `result` that appears at start of `chunk`
        overlap_window = result[-overlap_chars:]
        seam = _find_seam(overlap_window, chunk)
        result += "\n" + chunk[seam:]
    return result


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


# ── Main entry point ──────────────────────────────────────────────────────────


def translate_document(
    text: str,
    source_lang: str,
    target_lang: str,
    glossary_manager: GlossaryManager,
    model: str = DEFAULT_MODEL,
    use_structural_split: bool = False,
    verbose: bool = True,
) -> str:
    """
    Full pipeline:
      1. Split text into safe chunks
      2. For each chunk: retrieve relevant glossary terms, translate
      3. Stitch chunks back together
    """
    client = get_llm_client()

    # Step 1: Split
    if use_structural_split:
        chunks = split_by_structure(text)
    else:
        chunks = split_text(text)

    if verbose:
        print(f"Split into {len(chunks)} chunks.")

    # Step 2: Translate chunk by chunk
    translated_chunks = []
    previous_translated = None

    for i, chunk in enumerate(chunks):
        if verbose:
            print(f"Translating chunk {i + 1}/{len(chunks)}...")

        # Retrieve relevant glossary terms for this chunk
        glossary_terms = glossary_manager.retrieve(
            query_text=chunk,
            source_lang=source_lang,
            target_lang=target_lang,
            top_k=GLOSSARY_TOP_K,
        )

        translated = translate_chunk(
            client=client,
            chunk=chunk,
            source_lang=source_lang,
            target_lang=target_lang,
            glossary_terms=glossary_terms,
            previous_translated=previous_translated,
            model=model,
        )
        translated_chunks.append(translated)
        previous_translated = translated

    # Step 3: Stitch
    result = stitch_chunks(translated_chunks)
    if verbose:
        print("Translation complete.")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from src.glossary_manager import GlossaryManager

    parser = argparse.ArgumentParser(description="AI Translation Agent")
    parser.add_argument("input_file", help="Path to the text file to translate")
    parser.add_argument("source_lang", help="Source language (e.g. 'English')")
    parser.add_argument("target_lang", help="Target language (e.g. 'German')")
    parser.add_argument(
        "--glossary", default="glossary.csv", help="Path to glossary CSV file"
    )
    parser.add_argument("--output", default="translated.txt", help="Output file path")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="OpenRouter model string"
    )
    parser.add_argument(
        "--structural-split", action="store_true", help="Pre-split by headings/chapters"
    )
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    gm = GlossaryManager()
    gm.load_glossary(args.glossary)

    result = translate_document(
        text=text,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        glossary_manager=gm,
        model=args.model,
        use_structural_split=args.structural_split,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"Saved to {args.output}")