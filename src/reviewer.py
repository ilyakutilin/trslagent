"""Full-text review with glossary-aware LLM prompting.

This module provides proofreading/review: building prompts with glossary
and document context, and calling the LLM to check existing translations
for critical mistakes. Review can be chunked when `chunk.divider` is set,
mirroring the translation path's concurrency pattern.
"""

from iso639 import Lang

from src.config import logger
from src.llm import LLM


class Reviewer:
    """Reviews an existing translation for mistakes and improvements.

    Handles prompt construction and LLM interaction for the full source
    and target texts at once (no chunking).

    Attributes:
        source_lang: Source language of the original text
        target_lang: Target language of the translation under review
        specialized_in: Optional domain specialization for reviewer persona
        doc_type: Optional document type for context
        doc_title: Optional document title for context
        llm: LLM instance for generating reviews. If None, will only output prompts
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
            f"Reviewer initialized: {source_lang.name} -> {target_lang.name}, "
            f"specialized_in={specialized_in}, "
            f"doc_type={doc_type}, doc_title={doc_title}, "
            f"llm_available={llm is not None}"
        )

    def _build_system_prompt(
        self, user_glossary_str: str, auto_glossary_str: str
    ) -> str:
        specialization_section = (
            f" specialized in {self.specialized_in}" if self.specialized_in else ""
        )

        text_description_section = ""
        if any((self.doc_type, self.doc_title)):
            doc_type = self.doc_type if self.doc_type else "document"
            titled = f" titled '{self.doc_title}'" if self.doc_title else ""
            text_description_section = (
                f"\nThe text is a {doc_type}{titled}."
            )

        has_user_glossary = bool(user_glossary_str)
        has_any_glossary = bool(user_glossary_str or auto_glossary_str)

        dictionary_part = ""
        if user_glossary_str:
            dictionary_part += (
                "The translator was instructed to strictly follow these "
                "user-supplied dictionary terms. Make sure that if a term "
                "is in this dictionary, its translation corresponds "
                "to the dictionary.\n"
                f"<user dictionary start>\n{user_glossary_str}\n"
                "<user dictionary end>\n"
            )
        if auto_glossary_str:
            dictionary_part += (
                "The translator was provided with the following "
                "automatically matched terms from a reference glossary. "
                "Note that some of these terms may not be contextually "
                "relevant and the translator was not required to use them. "
                "Consider them as supplementary reference only.\n"
                f"<auto dictionary start>\n{auto_glossary_str}\n"
                "<auto dictionary end>\n"
            )

        result = (
            f"You are a professional experienced translator and editor"
            f"{specialization_section}. "
            "Your task is to review / proofread the existing translation "
            f"from {self.source_lang.name} into {self.target_lang.name}."
            f"{text_description_section}\n"
            "User prompt structure: "
            f"'{self.source_lang.name}: <source text here> "
            f"|| {self.target_lang.name}: <translation to be reviewed here>'.\n"
            "Do not mind the style, focus on critical mistakes like "
            "missing / untranslated parts, wrong translation (distortion of "
            "sense), unnecessary additions, spelling mistakes, wrong numbers "
            "and codes"
            f"{', deviation from the set dictionary' if has_user_glossary else ''} etc.\n"
            f"{dictionary_part}"
            "As a result provide a list of mistakes and potential "
            "improvements to the translation."
        )
        logger.debug(f"Built system prompt: {result}")
        return result

    def _build_user_prompt(
        self, source_text: str, target_text: str
    ) -> str:
        result = (
            f"{self.source_lang.name}: {source_text} "
            f"|| {self.target_lang.name}: {target_text}"
        )
        logger.debug(f"Built user prompt: {result}")
        return result

    def _print_prompts(self, system_prompt: str, user_prompt: str) -> None:
        print(f"\n{'=' * 10} SYSTEM PROMPT {'=' * 10}")
        print(system_prompt)
        print("=" * 35 + "\n" * 2)
        print(f"{'=' * 11} USER PROMPT {'=' * 11}")
        print(user_prompt)
        print("=" * 35 + "\n")

    async def review_text_async(
        self, source_text: str, target_text: str,
        user_glossary_str: str, auto_glossary_str: str,
    ) -> tuple[str | None, str | None]:
        logger.info(
            f"Reviewing text "
            f"(source_length={len(source_text)}, "
            f"target_length={len(target_text)})"
        )

        system_prompt = self._build_system_prompt(
            user_glossary_str=user_glossary_str,
            auto_glossary_str=auto_glossary_str,
        )
        user_prompt = self._build_user_prompt(source_text, target_text)

        if self.llm is None:
            logger.warning("LLM not available, printing prompts only")
            self._print_prompts(system_prompt, user_prompt)
            return None, None

        logger.debug("Sending to LLM for review")
        result, completion_id = await self.llm.get_reply_async(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        logger.debug(
            f"Received review: {len(result)} characters"
        )

        return result, completion_id
