from unittest.mock import AsyncMock

import pytest
from iso639 import Lang

from src.translator import Translator


@pytest.fixture
def translator(en_lang: Lang, ru_lang: Lang) -> Translator:
    return Translator(
        source_lang=en_lang,
        target_lang=ru_lang,
        specialized_in=None,
        doc_type=None,
        doc_title=None,
        llm=None,
    )


class TestBuildSystemPrompt:
    def test_no_specialization_no_glossary(self, translator: Translator):
        result = translator._build_system_prompt(
            is_extract=False,
            user_glossary_str="",
            auto_glossary_str="",
        )
        assert "You are a professional experienced translator" in result
        assert "from English into Russian" in result
        assert "specialized in" not in result
        assert "extract from" not in result
        assert "user dictionary" not in result
        assert "auto dictionary" not in result

    def test_with_specialized_in(self, translator: Translator):
        translator.specialized_in = "medicine"
        result = translator._build_system_prompt(
            is_extract=False,
            user_glossary_str="",
            auto_glossary_str="",
        )
        assert "specialized in medicine" in result

    def test_with_doc_type_and_title(self, translator: Translator):
        translator.doc_type = "report"
        translator.doc_title = "Annual Review"
        result = translator._build_system_prompt(
            is_extract=False,
            user_glossary_str="",
            auto_glossary_str="",
        )
        assert "The text for translation is " in result
        assert "report" in result
        assert "Annual Review" in result

    def test_is_extract_true(self, translator: Translator):
        translator.doc_type = "report"
        translator.doc_title = "Test"
        result = translator._build_system_prompt(
            is_extract=True,
            user_glossary_str="",
            auto_glossary_str="",
        )
        assert "an extract from" in result

    def test_is_extract_false(self, translator: Translator):
        translator.doc_type = "report"
        translator.doc_title = "Test"
        result = translator._build_system_prompt(
            is_extract=False,
            user_glossary_str="",
            auto_glossary_str="",
        )
        assert "an extract from" not in result
        assert "The text for translation is " in result

    def test_is_extract_false_no_doc_context(self, translator: Translator):
        result = translator._build_system_prompt(
            is_extract=False,
            user_glossary_str="",
            auto_glossary_str="",
        )
        assert "an extract from" not in result

    def test_is_extract_true_no_doc_context(self, translator: Translator):
        result = translator._build_system_prompt(
            is_extract=True,
            user_glossary_str="",
            auto_glossary_str="",
        )
        assert "an extract from" not in result
        assert "The text for translation is" not in result

    def test_with_user_glossary(self, translator: Translator):
        glossary = "flow meter = расходомер\npressure valve = клапан давления"
        result = translator._build_system_prompt(
            is_extract=False,
            user_glossary_str=glossary,
            auto_glossary_str="",
        )
        assert "user dictionary" in result
        assert "<user dictionary start>" in result
        assert "<user dictionary end>" in result
        assert "flow meter" in result
        assert "auto dictionary" not in result

    def test_with_auto_glossary(self, translator: Translator):
        glossary = "flow meter = расходомер\npressure valve = клапан давления"
        result = translator._build_system_prompt(
            is_extract=False,
            user_glossary_str="",
            auto_glossary_str=glossary,
        )
        assert "auto dictionary" in result
        assert "<auto dictionary start>" in result
        assert "<auto dictionary end>" in result
        assert "flow meter" in result
        assert "user dictionary" not in result

    def test_with_both_glossaries(self, translator: Translator):
        user_glossary = "flow meter = расходомер"
        auto_glossary = "pressure valve = клапан давления"
        result = translator._build_system_prompt(
            is_extract=False,
            user_glossary_str=user_glossary,
            auto_glossary_str=auto_glossary,
        )
        assert "user dictionary" in result
        assert "auto dictionary" in result
        assert result.index("user dictionary") < result.index("auto dictionary")
        assert "flow meter" in result
        assert "pressure valve" in result


class TestBuildUserPrompt:
    def test_wraps_text_correctly(self, translator: Translator, sample_text: str):
        result = translator._build_user_prompt(sample_text)
        assert result == f"Text for translation:\n{sample_text}"

    def test_empty_chunk(self, translator: Translator):
        result = translator._build_user_prompt("")
        assert result == "Text for translation:\n"


class TestTranslateChunkAsync:
    @pytest.fixture
    def translator_with_llm(
        self, en_lang: Lang, ru_lang: Lang, mock_llm: AsyncMock
    ) -> Translator:
        return Translator(
            source_lang=en_lang,
            target_lang=ru_lang,
            specialized_in=None,
            doc_type=None,
            doc_title=None,
            llm=mock_llm,
        )

    @pytest.mark.asyncio
    async def test_llm_available(
        self, translator_with_llm: Translator, sample_text: str, mock_llm: AsyncMock
    ):
        translated, cid = await translator_with_llm.translate_chunk_async(
            chunk=sample_text,
            user_glossary_str="",
            auto_glossary_str="",
            is_extract=False,
        )
        assert translated == "Mocked translation text"
        assert cid == "mock-completion-id"
        mock_llm.get_reply_async.assert_called_once()
        call_kwargs = mock_llm.get_reply_async.call_args.kwargs
        assert "professional experienced translator" in call_kwargs["system_prompt"]
        assert "English" in call_kwargs["system_prompt"]
        assert "Russian" in call_kwargs["system_prompt"]
        assert call_kwargs["user_prompt"].startswith("Text for translation:\n")
        assert sample_text in call_kwargs["user_prompt"]

    @pytest.mark.asyncio
    async def test_llm_none(self, translator: Translator, sample_text: str, capsys):
        translated, cid = await translator.translate_chunk_async(
            chunk=sample_text,
            user_glossary_str="",
            auto_glossary_str="",
            is_extract=False,
        )
        assert translated is None
        assert cid is None

        captured = capsys.readouterr()
        assert "SYSTEM PROMPT" in captured.out
        assert "You are a professional experienced translator" in captured.out
        assert "from English into Russian" in captured.out
        assert "USER PROMPT" in captured.out
        assert captured.out.index("SYSTEM PROMPT") < captured.out.index("USER PROMPT")
        assert f"Text for translation:\n{sample_text}" in captured.out

    @pytest.mark.asyncio
    async def test_with_glossary_passed_to_llm(
        self, translator_with_llm: Translator, sample_text: str, mock_llm: AsyncMock
    ):
        user_glossary = "flow meter = расходомер"
        auto_glossary = "pressure valve = клапан давления"
        await translator_with_llm.translate_chunk_async(
            chunk=sample_text,
            user_glossary_str=user_glossary,
            auto_glossary_str=auto_glossary,
            is_extract=False,
        )
        call_kwargs = mock_llm.get_reply_async.call_args.kwargs
        assert "user dictionary" in call_kwargs["system_prompt"]
        assert "auto dictionary" in call_kwargs["system_prompt"]
        assert "flow meter" in call_kwargs["system_prompt"]
        assert "pressure valve" in call_kwargs["system_prompt"]

    @pytest.mark.asyncio
    async def test_is_extract_passed_to_llm(
        self, translator_with_llm: Translator, sample_text: str, mock_llm: AsyncMock
    ):
        translator_with_llm.doc_type = "report"
        translator_with_llm.doc_title = "Test Doc"
        await translator_with_llm.translate_chunk_async(
            chunk=sample_text,
            user_glossary_str="",
            auto_glossary_str="",
            is_extract=True,
        )
        call_kwargs = mock_llm.get_reply_async.call_args.kwargs
        assert "an extract from" in call_kwargs["system_prompt"]
