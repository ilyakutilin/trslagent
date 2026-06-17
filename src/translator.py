"""Single-chunk translation with glossary-aware LLM prompting.

This module provides the core translation unit: building prompts with glossary
and document context, and calling the LLM for a single text chunk. Chunking,
glossary matching across the document, and result stitching are handled by the
caller.
"""

from iso639 import Lang

from src.config import logger
from src.llm import LLM


class Translator:
    """Translates a single text chunk with glossary support.

    Handles prompt construction and LLM interaction for one chunk at a time.

    Attributes:
        source_lang: Source language for translation
        target_lang: Target language for translation
        specialized_in: Optional domain specialization for translator persona
        doc_type: Optional document type for context
        doc_title: Optional document title for context
        llm: LLM instance for generating translations. If None, will only output prompts
    """

    def __init__(
        self,
        source_lang: Lang,
        target_lang: Lang,
        specialized_in: str | None,
        doc_type: str | None,
        doc_title: str | None,
        llm: LLM | None,
    ) -> None:
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.specialized_in = specialized_in
        self.doc_type = doc_type
        self.doc_title = doc_title
        self.llm = llm

        logger.info(
            f"Translator initialized: {source_lang.name} -> {target_lang.name}, "
            f"specialized_in={specialized_in}, "
            f"doc_type={doc_type}, doc_title={doc_title}, "
            f"llm_available={llm is not None}"
        )

    def _build_system_prompt(
        self,
        is_extract: bool,
        user_glossary_str: str,
        auto_glossary_str: str,
    ) -> str:
        specialization_section = (
            f" specialized in {self.specialized_in}" if self.specialized_in else ""
        )

        text_description_section = ""
        if any((self.doc_type, self.doc_title)):
            an_extract_from = "an extract from " if is_extract else ""
            doc_type = self.doc_type if self.doc_type else "document"
            titled = f" titled '{self.doc_title}'" if self.doc_title else ""
            text_description_section = (
                f"\nThe text for translation is {an_extract_from}a {doc_type}{titled}."
            )

        glossary_section = ""
        if user_glossary_str:
            glossary_section += (
                "\nThe following terms were provided by the user. If a term "
                "appears in the source text, its translation must be taken "
                "from this dictionary strictly.\n"
                f"<user dictionary start>\n{user_glossary_str}\n"
                "<user dictionary end>"
            )
        if auto_glossary_str:
            glossary_section += (
                "\nThe following terms were automatically matched from a "
                "reference glossary. Note that some terms may not be "
                "contextually relevant. Consider and use them only if "
                "applicable; otherwise ignore.\n"
                f"<auto dictionary start>\n{auto_glossary_str}\n"
                "<auto dictionary end>"
            )

        result = (
            f"You are a professional experienced translator{specialization_section}. "
            "Your task is to translate the text provided by the user "
            f"from {self.source_lang.name} into {self.target_lang.name}."
            f"{text_description_section}"
            f"{glossary_section}"
        )
        logger.debug(f"Built system prompt: {result}")
        return result

    def _build_user_prompt(self, chunk: str) -> str:
        result = f"Text for translation:\n{chunk}"
        logger.debug(f"Built user prompt: {result}")
        return result

    def _print_prompts(self, system_prompt: str, user_prompt: str) -> None:
        # TODO: Add chunk identification
        print(f"\n{'=' * 10} SYSTEM PROMPT {'=' * 10}")
        print(system_prompt)
        print("=" * 35 + "\n" * 2)
        print(f"{'=' * 11} USER PROMPT {'=' * 11}")
        print(user_prompt)
        print("=" * 35 + "\n")

    async def translate_chunk_async(
        self, chunk: str, user_glossary_str: str, auto_glossary_str: str,
        is_extract: bool
    ) -> tuple[str | None, str | None]:
        logger.info(f"Translating chunk (length={len(chunk)})")

        system_prompt = self._build_system_prompt(
            is_extract=is_extract,
            user_glossary_str=user_glossary_str,
            auto_glossary_str=auto_glossary_str,
        )
        user_prompt = self._build_user_prompt(chunk)

        if self.llm is None:
            logger.warning("LLM not available, printing prompts only")
            self._print_prompts(system_prompt, user_prompt)
            return None, None

        logger.debug("Sending chunk to LLM")
        translated_chunk, completion_id = await self.llm.get_reply_async(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        logger.debug(f"Received translation: {len(translated_chunk)} characters")

        return translated_chunk, completion_id
