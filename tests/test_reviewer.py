from unittest.mock import AsyncMock

import pytest
from iso639 import Lang

from src.reviewer import Reviewer


@pytest.fixture
def reviewer(en_lang: Lang, ru_lang: Lang) -> Reviewer:
    return Reviewer(
        source_lang=en_lang,
        target_lang=ru_lang,
        specialized_in=None,
        doc_type=None,
        doc_title=None,
        llm=None,
    )


class TestBuildSystemPrompt:
    def test_no_specialization_no_glossary_no_doc_context(
        self, reviewer: Reviewer
    ):
        result = reviewer._build_system_prompt(
            user_glossary_str="",
            auto_glossary_str="",
        )
        assert "professional experienced translator" in result
        assert "from English into Russian" in result
        assert "specialized in" not in result
        assert "user dictionary" not in result
        assert "auto dictionary" not in result
        assert "deviation from the set dictionary" not in result

    def test_with_specialized_in(self, reviewer: Reviewer):
        reviewer.specialized_in = "medicine"
        result = reviewer._build_system_prompt(
            user_glossary_str="",
            auto_glossary_str="",
        )
        assert "specialized in medicine" in result

    def test_with_doc_type_and_title(self, reviewer: Reviewer):
        reviewer.doc_type = "report"
        reviewer.doc_title = "Annual Review"
        result = reviewer._build_system_prompt(
            user_glossary_str="",
            auto_glossary_str="",
        )
        assert "report" in result
        assert "Annual Review" in result

    def test_with_user_glossary(self, reviewer: Reviewer):
        glossary = "flow meter = расходомер\npressure valve = клапан давления"
        result = reviewer._build_system_prompt(
            user_glossary_str=glossary,
            auto_glossary_str="",
        )
        assert "user dictionary" in result
        assert "<user dictionary start>" in result
        assert "<user dictionary end>" in result
        assert "flow meter" in result
        assert "deviation from the set dictionary" in result
        assert "auto dictionary" not in result

    def test_with_auto_glossary(self, reviewer: Reviewer):
        glossary = "flow meter = расходомер\npressure valve = клапан давления"
        result = reviewer._build_system_prompt(
            user_glossary_str="",
            auto_glossary_str=glossary,
        )
        assert "auto dictionary" in result
        assert "<auto dictionary start>" in result
        assert "<auto dictionary end>" in result
        assert "flow meter" in result
        assert "deviation from the set dictionary" not in result
        assert "user dictionary" not in result

    def test_with_both_glossaries(self, reviewer: Reviewer):
        user_glossary = "flow meter = расходомер"
        auto_glossary = "pressure valve = клапан давления"
        result = reviewer._build_system_prompt(
            user_glossary_str=user_glossary,
            auto_glossary_str=auto_glossary,
        )
        assert "user dictionary" in result
        assert "auto dictionary" in result
        assert "flow meter" in result
        assert "pressure valve" in result
        assert "deviation from the set dictionary" in result


class TestBuildUserPrompt:
    def test_formats_source_and_target(self, reviewer: Reviewer):
        source = "The quick brown fox"
        target = "Быстрая коричневая лиса"
        result = reviewer._build_user_prompt(source, target)
        assert result == "English: The quick brown fox || Russian: Быстрая коричневая лиса"


class TestReviewTextAsync:
    @pytest.fixture
    def reviewer_with_llm(
        self, en_lang: Lang, ru_lang: Lang, mock_llm: AsyncMock
    ) -> Reviewer:
        return Reviewer(
            source_lang=en_lang,
            target_lang=ru_lang,
            specialized_in=None,
            doc_type=None,
            doc_title=None,
            llm=mock_llm,
        )

    @pytest.mark.asyncio
    async def test_llm_available(
        self, reviewer_with_llm: Reviewer, mock_llm: AsyncMock
    ):
        source = "The quick brown fox."
        target = "Быстрая коричневая лиса."
        review, cid = await reviewer_with_llm.review_text_async(
            source_text=source,
            target_text=target,
            user_glossary_str="",
            auto_glossary_str="",
        )
        assert review == "Mocked translation text"
        assert cid == "mock-completion-id"
        mock_llm.get_reply_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_none(self, reviewer: Reviewer, capsys):
        source = "The quick brown fox."
        target = "Быстрая коричневая лиса."
        review, cid = await reviewer.review_text_async(
            source_text=source,
            target_text=target,
            user_glossary_str="",
            auto_glossary_str="",
        )
        assert review is None
        assert cid is None

        captured = capsys.readouterr()
        assert "SYSTEM PROMPT" in captured.out
        assert "USER PROMPT" in captured.out
        assert source in captured.out
        assert target in captured.out
