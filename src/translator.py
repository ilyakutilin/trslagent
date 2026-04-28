"""
AI Translation Agent
====================
Translates large documents using:
- ChromaDB + multilingual-e5-large for glossary RAG
- OpenRouter API for LLM translation
"""

from iso639 import Lang

from src.config import get_settings
from src.glossary_matcher import TermMatcher
from src.lemmatizer import Lemmatizer
from src.llm import LLM
from src.models import GlossaryEntry
from src.splitter import split_text, stitch_chunks, truncate_at_sentence_boundary

# ── Configuration ────────────────────────────────────────────────────────────

settings = get_settings()


class Translator:
    def __init__(
        self,
        source_lang: Lang,
        target_lang: Lang,
        specialized_in: str | None,
        text: str,
        doc_type: str | None,
        doc_title: str | None,
        llm: LLM,
        lemmatizer: Lemmatizer,
        main_glossary_entries: list[GlossaryEntry],
        project_glossary_entries: list[GlossaryEntry],
    ) -> None:
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

    def _match_main_glossary_entries_for_chunk(self, chunk: str) -> list[GlossaryEntry]:
        term_matcher = TermMatcher(glossary_entries=self.main_glossary_entries)
        return term_matcher.match(
            text=chunk, lang=self.source_lang, lemmatizer=self.lemmatizer
        )

    def _combine_glossaries_for_chunk(
        self,
        chunk_glossary_entries: list[GlossaryEntry],
        project_glossary_entries: list[GlossaryEntry],
    ) -> list[GlossaryEntry]:
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

        return final_chunk_entries

    def _stringify_glossary(self, entries: list[GlossaryEntry]) -> str:
        str_entries: list[str] = []
        for entry in entries:
            str_entry = entry.stringify(self.source_lang, self.target_lang)
            if str_entry is not None:
                str_entries.append(str_entry)

        return "\n".join(str_entries)

    def _build_system_prompt(
        self,
        is_extract: bool,
        previous_translated: str | None,
        chunk_glossary: str | None,
    ) -> str:
        specialization_section = (
            f" specialized in {self.specialized_in}" if self.specialized_in else ""
        )

        text_descriprion_section = ""
        if any((self.doc_type, self.doc_title)):
            an_extract_from = " an extract from" if is_extract else ""
            doc_type = self.doc_type if self.doc_type else "document"
            titled = f" titled '{self.doc_title}'" if self.doc_title else ""
            text_descriprion_section = (
                f"\nThe text for translation is {an_extract_from}a {doc_type}{titled}."
            )

        glossary_section = ""
        if chunk_glossary:
            glossary_section = (
                "\nUse the following dictionary when translating. If a term is in the "
                "dictionary, its translation shall be taken from the dictionary.\n"
                "<dictionary start>\n{chunk_glossary}\n<dictionary end>"
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

        return (
            f"You are a professional experienced translator{specialization_section}. "
            "Your task is to translate the text provided by the user "
            f"from {self.source_lang} into {self.target_lang}."
            f"{text_descriprion_section}"
            f"{glossary_section}"
            f"{context_section}"
        )

    def _build_user_prompt(
        self,
    ) -> str:
        return ""

    def translate_document(self) -> str:
        chunks: list[str] = split_text(
            text=self.text,
            chunk_size=settings.chunk.size,
            chunk_overlap=settings.chunk.overlap,
        )

        translated_chunks: list[str] = []
        previous_translated = None

        chunk_is_extract = len(chunks) > 1
        for chunk in chunks:
            matched_entries = self._match_main_glossary_entries_for_chunk(chunk)
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
            user_prompt = self._build_user_prompt()
            translated_chunk = self.llm.get_reply(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            translated_chunks.append(translated_chunk)
            previous_translated = translated_chunk

        result = stitch_chunks(translated_chunks, chunk_overlap=settings.chunk.overlap)

        return result
