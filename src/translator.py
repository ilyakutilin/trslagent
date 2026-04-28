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
from src.models import GlossaryEntry, Term
from src.splitter import split_text, stitch_chunks

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
    ) -> dict[str, str]:
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

        res_dict: dict[str, str] = {}
        for ge in final_chunk_entries:
            source_part = ge.stringify_lang(self.source_lang)
            target_part = ge.stringify_lang(self.target_lang)
            res_dict[source_part] = target_part

        return res_dict

    def _build_system_prompt(
        self,
        is_extract: bool,
        previous_translated: str | None,
        chunk_glossary: dict[str, str] | None,
    ) -> str:
        return ""

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
            chunk_glossary = self._combine_glossaries_for_chunk(
                chunk_glossary_entries=matched_entries,
                project_glossary_entries=self.project_glossary_entries,
            )

            system_prompt = self._build_system_prompt(
                is_extract=chunk_is_extract,
                previous_translated=previous_translated,
                chunk_glossary=chunk_glossary,
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
