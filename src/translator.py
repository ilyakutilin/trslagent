"""Document translation service with glossary-aware LLM-based translation.

This module provides the core translation functionality that handles:
- Chunking large documents for efficient LLM processing
- Glossary matching and term consistency enforcement
- Context-aware prompt construction
- Translation continuity between document segments
- Glossary priority handling (project glossary overrides main glossary)
"""

from iso639 import Lang

from src.config import get_settings, logger
from src.glossary.matcher import TermMatcher
from src.glossary.models import GlossaryEntry
from src.lemmatizer import Lemmatizer
from src.llm import LLM
from src.splitter import split_text, stitch_chunks, truncate_at_sentence_boundary

# ── Configuration ────────────────────────────────────────────────────────────

settings = get_settings()


class Translator:
    """Main translator orchestrator for document translation with glossary support.

    Handles the complete translation workflow including text chunking, glossary matching,
    prompt construction, LLM interaction, and result stitching. Maintains translation
    consistency across document segments and enforces glossary term usage.

    Attributes:
        source_lang: Source language for translation
        target_lang: Target language for translation
        specialized_in: Optional domain specialization for translator persona
        text: Full input text to be translated
        doc_type: Optional document type for context (e.g. "technical manual", "legal document")
        doc_title: Optional document title for context
        llm: LLM instance for generating translations. If None, will only output prompts
        lemmatizer: Lemmatizer instance for term normalization
        main_glossary_entries: Global glossary entries available for all documents
        project_glossary_entries: Project-specific glossary entries with higher priority
    """

    def __init__(
        self,
        source_lang: Lang,
        target_lang: Lang,
        specialized_in: str | None,
        text: str,
        doc_type: str | None,
        doc_title: str | None,
        llm: LLM | None,
        lemmatizer: Lemmatizer,
        main_glossary_entries: list[GlossaryEntry],
        project_glossary_entries: list[GlossaryEntry],
    ) -> None:
        """Initialize Translator instance with configuration and dependencies.

        Args:
            source_lang: Source language for translation
            target_lang: Target language for translation
            specialized_in: Optional domain specialization for translator persona
            text: Full input text to be translated
            doc_type: Optional document type for context
            doc_title: Optional document title for context
            llm: LLM instance for generating translations.
                If None, prompts will be printed only
            lemmatizer: Lemmatizer instance for term normalization
            main_glossary_entries: Global glossary entries available for all documents
            project_glossary_entries: Project-specific glossary entries
                (override main glossary)
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.specialized_in = specialized_in
        self.text = text
        self.doc_type = doc_type
        self.doc_title = doc_title
        self.llm = llm
        self.lemmatizer = lemmatizer
        self.main_glossary_entries = main_glossary_entries
        self.project_glossary_entries = project_glossary_entries

        logger.info(
            f"Translator initialized: {source_lang} -> {target_lang}, "
            f"specialized_in={specialized_in}, "
            f"text_length={len(text)}, "
            f"doc_type={doc_type}, doc_title={doc_title}, "
            f"llm_available={llm is not None}, "
            f"main_glossary_entries={len(main_glossary_entries)}, "
            f"project_glossary_entries={len(project_glossary_entries)}"
        )

    def _match_main_glossary_entries_for_chunk(
        self, chunk: str, matcher: TermMatcher
    ) -> list[GlossaryEntry]:
        """Find main glossary entries that appear in a specific text chunk.

        Uses the provided term matcher to identify relevant glossary terms
        present in the chunk text, with lemmatization for better matching.

        Args:
            chunk: Text chunk to search for glossary terms
            matcher: TermMatcher instance initialized with main glossary entries

        Returns:
            List of GlossaryEntry objects that match terms in the chunk
        """
        matched = matcher.match(
            text=chunk, lang=self.source_lang, lemmatizer=self.lemmatizer
        )
        logger.debug(
            f"Matched {len(matched)} main glossary entries "
            f"for chunk of length {len(chunk)}"
        )
        return matched

    def _combine_glossaries_for_chunk(
        self,
        chunk_glossary_entries: list[GlossaryEntry],
        project_glossary_entries: list[GlossaryEntry],
    ) -> list[GlossaryEntry]:
        """Combine main and project glossaries with priority resolution.

        Project glossary entries take precedence. Any term present in both
        glossaries will be taken only from the project glossary to avoid conflicts.

        Args:
            chunk_glossary_entries: Main glossary entries matched for current chunk
            project_glossary_entries: Project-specific glossary entries

        Returns:
            Combined list of glossary entries without conflicting terms
        """
        # TODO: Cache lemmatized_project_terms - no need to parse them for every chunk
        lemmatized_project_terms: list[str] = []
        for ge in project_glossary_entries:
            for term in [t for t in ge.terms if t.language == self.source_lang]:
                if term.lemmatized:
                    lemmatized_project_terms.append(term.lemmatized)

        final_chunk_entries = project_glossary_entries.copy()
        for ge in chunk_glossary_entries:
            to_include = True
            for term in [t for t in ge.terms if t.language == self.source_lang]:
                if term.lemmatized in lemmatized_project_terms:
                    to_include = False

            if to_include:
                final_chunk_entries.append(ge)

        logger.debug(
            f"Combined glossaries: {len(chunk_glossary_entries)} main + "
            f"{len(project_glossary_entries)} project = "
            f"{len(final_chunk_entries)} total entries"
        )
        return final_chunk_entries

    def _stringify_glossary(self, entries: list[GlossaryEntry]) -> str:
        """Convert glossary entries list to human-readable string format for prompts.

        Formats each glossary entry with source and target language terms,
        suitable for inclusion in LLM prompts.

        Args:
            entries: List of GlossaryEntry objects to stringify

        Returns:
            Newline-separated string of formatted glossary entries
        """
        str_entries: list[str] = []
        for entry in entries:
            str_entry = entry.stringify(self.source_lang, self.target_lang)
            if str_entry is not None:
                str_entries.append(str_entry)

        result = "\n".join(str_entries)
        logger.debug(
            f"Stringified {len(entries)} glossary entries into {len(result)} characters"
        )
        return result

    def _build_system_prompt(
        self,
        is_extract: bool,
        previous_translated: str | None,
        chunk_glossary: str | None,
    ) -> str:
        """Construct system prompt for LLM translation request.

        Builds a context-rich prompt including translator persona, document context,
        glossary instructions, and previous translation segment for continuity.

        Args:
            is_extract: True if chunk is part of a larger document,
                False for full document
            previous_translated: Translated text from previous chunk for continuity
                context
            chunk_glossary: Stringified glossary entries relevant for this chunk

        Returns:
            Complete system prompt string ready for LLM
        """
        specialization_section = (
            f" specialized in {self.specialized_in}" if self.specialized_in else ""
        )

        text_description_section = ""
        if any((self.doc_type, self.doc_title)):
            an_extract_from = " an extract from" if is_extract else ""
            doc_type = self.doc_type if self.doc_type else "document"
            titled = f" titled '{self.doc_title}'" if self.doc_title else ""
            text_description_section = (
                f"\nThe text for translation is {an_extract_from}a {doc_type}{titled}."
            )

        glossary_section = ""
        if chunk_glossary:
            glossary_section = (
                "\nUse the following dictionary when translating. If a term is in the "
                "dictionary, its translation shall be taken from the dictionary.\n"
                f"<dictionary start>\n{chunk_glossary}\n<dictionary end>"
            )

        context_section = ""
        if previous_translated:
            tail = truncate_at_sentence_boundary(previous_translated, window=400)
            context_section = (
                "\nTranslation of the PREVIOUS SEGMENT (for your reference and for "
                f"context and style continuity — do NOT retranslate this):\n{tail}"
            )

        # TODO: Implement Analyze New Terms
        # analyze_new_terms = (
        #     "After the translation analyze the extract for the terms that"
        #     f"{' are not yet in the dictionary but' if glossary else ''} "
        #     "you consider important or frequ ently repeated - provide them "
        #     f"in a separate block after the translation in the {self.source_lang} "
        #     f"term = {self.target_lang} term format."
        # )

        result = (
            f"You are a professional experienced translator{specialization_section}. "
            "Your task is to translate the text provided by the user "
            f"from {self.source_lang} into {self.target_lang}."
            f"{text_description_section}"
            f"{glossary_section}"
            f"{context_section}"
        )
        logger.debug(
            f"Built system prompt with specialization={bool(self.specialized_in)}, "
            f"doc_context={bool(text_description_section)}, "
            f"glossary={bool(glossary_section)}, "
            f"context={bool(context_section)}"
        )
        return result

    def _build_user_prompt(self, chunk: str) -> str:
        """Construct user prompt containing the text to be translated.

        Args:
            chunk: Text chunk to be translated

        Returns:
            Formatted user prompt string
        """
        result = f"Text for translation:\n{chunk}"
        logger.debug(f"Built user prompt with chunk length {len(chunk)}")
        return result

    def _print_prompts(self, system_prompt: str, user_prompt: str) -> None:
        """Print constructed prompts to console for debugging purposes.

        Formats and displays system and user prompts with clear separators.

        Args:
            system_prompt: System prompt string to print
            user_prompt: User prompt string to print
        """
        # TODO: Add chunk identification
        print(f"\n{'=' * 10} SYSTEM PROMPT {'=' * 10}")
        print(system_prompt)
        print("=" * 35 + "\n" * 2)
        print(f"{'=' * 11} USER PROMPT {'=' * 11}")
        print(user_prompt)
        print("=" * 35 + "\n")

    def translate_document(self) -> str | None:
        """Execute full document translation workflow.

        Splits document into chunks, processes each chunk with glossary matching and
        context-aware translation, then stitches translated chunks back together.
        Maintains translation continuity between adjacent chunks.

        Returns:
            Fully translated document string if LLM is available, None if only
            prompts were printed (when LLM instance is not provided)
        """
        logger.info(f"Starting document translation: text length {len(self.text)}")

        chunks: list[str] = split_text(
            text=self.text,
            chunk_size=settings.chunk.size,
            chunk_overlap=settings.chunk.overlap,
        )

        logger.info(
            f"Split text into {len(chunks)} chunks (size={settings.chunk.size}, "
            f"overlap={settings.chunk.overlap})"
        )

        translated_chunks: list[str] = []
        previous_translated = None

        len_chunks = len(chunks)
        chunk_is_extract = len_chunks > 1
        term_matcher = TermMatcher(glossary_entries=self.main_glossary_entries)
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)} (length={len(chunk)})")

            # TODO: Implement matching project glossary for current chunk
            matched_entries = self._match_main_glossary_entries_for_chunk(
                chunk=chunk, matcher=term_matcher
            )
            chunk_glossary_entries = self._combine_glossaries_for_chunk(
                chunk_glossary_entries=matched_entries,
                project_glossary_entries=self.project_glossary_entries,
            )
            chunk_glossary_str = self._stringify_glossary(chunk_glossary_entries)

            system_prompt = self._build_system_prompt(
                is_extract=chunk_is_extract,
                previous_translated=previous_translated,
                chunk_glossary=chunk_glossary_str,
            )
            user_prompt = self._build_user_prompt(chunk)

            if self.llm is None:
                logger.warning("LLM not available, printing prompts only")
                self._print_prompts(system_prompt, user_prompt)
                if i == len_chunks - 1:
                    return None
                continue

            # TODO: Verify that the overall prompt is within reasonable limits
            logger.debug(f"Sending chunk {i + 1} to LLM")
            translated_chunk = self.llm.get_reply(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            logger.debug(
                f"Received translation for chunk {i + 1}: {len(translated_chunk)} "
                "characters"
            )

            translated_chunks.append(translated_chunk)
            previous_translated = translated_chunk

        result = stitch_chunks(translated_chunks, chunk_overlap=settings.chunk.overlap)
        logger.info(
            f"Translation complete: {len(self.text)} -> {len(result)} characters"
        )

        return result
