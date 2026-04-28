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
        return []

    def _combine_glossaries_for_chunk(
        self,
        chunk: str,
        chunk_glossary_entries: list[GlossaryEntry],
        project_glossary_entries: list[GlossaryEntry],
    ) -> dict[str, str]:
        return {"": ""}

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

        for chunk in chunks:
            matched_entries = self._match_main_glossary_entries_for_chunk(chunk)
            chunk_glossary = self._combine_glossaries_for_chunk(
                chunk=chunk,
                chunk_glossary_entries=matched_entries,
                project_glossary_entries=self.project_glossary_entries,
            )

            system_prompt = self._build_system_prompt(
                is_extract=len(chunks) > 1,
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
