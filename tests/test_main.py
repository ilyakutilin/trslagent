from unittest.mock import AsyncMock

import pytest
from iso639 import Lang
from pydantic import SecretStr

from src.config import ChunkSettings, CostSettings, InputData, LLMSettings, OutputData, Settings
from src.glossary.models import GlossaryEntry, Term
from src.main import (
    _deduplicate_entries,
    _resolve_and_log_cost,
    _stringify_glossary,
    export_glossary_matches,
    main,
)


@pytest.fixture(autouse=True)
def _reset_toml_path():
    Settings._toml_path = None
    yield
    Settings._toml_path = None


def _make_term(lang_str: str, value: str, lemmatized: str | None = None) -> Term:
    return Term(language=Lang(lang_str), value=value, lemmatized=lemmatized)


def _make_entry(
    entry_id: int,
    en_value: str,
    ru_value: str,
    en_lemma: str | None = None,
    ru_lemma: str | None = None,
) -> GlossaryEntry:
    return GlossaryEntry(
        id=entry_id,
        terms=frozenset([
            _make_term("en", en_value, en_lemma),
            _make_term("ru", ru_value, ru_lemma),
        ]),
    )


class TestStringifyGlossary:
    def test_matching_langs(self, en_lang: Lang, ru_lang: Lang):
        entries = [
            _make_entry(1, "flow meter", "расходомер", "flow meter", "расходомер"),
            _make_entry(2, "pressure valve", "клапан давления", "pressure valve", "клапан давления"),
        ]
        result = _stringify_glossary(entries, en_lang, ru_lang)
        assert "flow meter = расходомер" in result
        assert "pressure valve = клапан давления" in result
        assert "\n" in result

    def test_mismatched_langs_skipped(self, en_lang: Lang, ru_lang: Lang):
        entry = GlossaryEntry(
            id=1,
            terms=frozenset([
                _make_term("en", "hello"),
                _make_term("fr", "bonjour"),
            ]),
        )
        result = _stringify_glossary([entry], en_lang, ru_lang)
        assert result == ""

    def test_empty_list(self, en_lang: Lang, ru_lang: Lang):
        assert _stringify_glossary([], en_lang, ru_lang) == ""


class TestDeduplicateEntries:
    def test_user_overrides_matched_auto(self, en_lang: Lang):
        user_entry = _make_entry(10, "flow meter", "расходомер", "flow meter", "расходомер")
        auto_entry = _make_entry(1, "flow meter", "расходомер", "flow meter", "расходомер")

        user_entries, auto_entries = _deduplicate_entries([auto_entry], [user_entry], en_lang)
        assert len(user_entries) == 1
        assert len(auto_entries) == 0

    def test_no_overlap(self, en_lang: Lang):
        user_entry = _make_entry(10, "flow meter", "расходомер", "flow meter", "расходомер")
        auto_entry = _make_entry(1, "pressure valve", "клапан давления", "pressure valve", "клапан давления")

        user_entries, auto_entries = _deduplicate_entries([auto_entry], [user_entry], en_lang)
        assert len(user_entries) == 1
        assert len(auto_entries) == 1

    def test_empty_inputs(self, en_lang: Lang):
        user_entries, auto_entries = _deduplicate_entries([], [], en_lang)
        assert user_entries == []
        assert auto_entries == []


class TestTranslationPipeline:
    @pytest.mark.asyncio
    async def test_source_text_chunks_translate_stitch(self, mocker):
        mocker.patch("src.main._parse_glossaries", return_value=([], []))
        mocker.patch("src.main.fetch_cost")

        mock_llm = AsyncMock()
        mock_llm.get_reply_async.return_value = ("Переведённый текст", "completion-1")
        mocker.patch("src.main.LLM", return_value=mock_llm)

        cfg = Settings(
            llm=LLMSettings(api_key=SecretStr("test-key")),
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="Hello world. This is a test.",
            ),
            chunk=ChunkSettings(size=1000, max_concurrent=1, delay_seconds=0),
        )

        result = await main(cfg)
        assert result == "Переведённый текст"
        mock_llm.get_reply_async.assert_called()

    @pytest.mark.asyncio
    async def test_with_user_glossary_no_auto(self, mocker):
        user_entries = [_make_entry(10, "flow meter", "расходомер", "flow meter", "расходомер")]
        mocker.patch("src.main._parse_glossaries", return_value=([], user_entries))
        mocker.patch("src.main.fetch_cost")

        mock_llm = AsyncMock()
        mock_llm.get_reply_async.return_value = ("Перевод", "completion-1")
        mocker.patch("src.main.LLM", return_value=mock_llm)

        cfg = Settings(
            llm=LLMSettings(api_key=SecretStr("test-key")),
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="The flow meter is broken.",
            ),
            chunk=ChunkSettings(size=1000, max_concurrent=1, delay_seconds=0),
        )

        result = await main(cfg)
        assert result == "Перевод"
        mock_llm.get_reply_async.assert_called()

    @pytest.mark.asyncio
    async def test_divider_based_chunking(self, mocker):
        mocker.patch("src.main._parse_glossaries", return_value=([], []))
        mocker.patch("src.main.fetch_cost")

        mock_llm = AsyncMock()
        mock_llm.get_reply_async.side_effect = [
            ("Chunk 1", "id-1"),
            ("Chunk 2", "id-2"),
        ]
        mocker.patch("src.main.LLM", return_value=mock_llm)

        text = "Section A\n----------\nSection B"
        cfg = Settings(
            llm=LLMSettings(api_key=SecretStr("test-key")),
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text=text,
            ),
            chunk=ChunkSettings(divider="-", max_concurrent=1, delay_seconds=0),
        )

        result = await main(cfg)
        assert result is not None
        assert "Chunk 1" in result
        assert "Chunk 2" in result


class TestReviewMode:
    @pytest.mark.asyncio
    async def test_non_chunked_review(self, mocker):
        mocker.patch("src.main._parse_glossaries", return_value=([], []))
        mocker.patch("src.main.fetch_cost")

        mock_llm = AsyncMock()
        mock_llm.get_reply_async.return_value = ("Review result", "id-1")
        mocker.patch("src.main.LLM", return_value=mock_llm)

        cfg = Settings(
            llm=LLMSettings(api_key=SecretStr("test-key")),
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="Hello world.",
                target_text="Привет мир.",
            ),
        )

        result = await main(cfg)
        assert result == "Review result"

    @pytest.mark.asyncio
    async def test_divider_review_equal_chunks(self, mocker):
        mocker.patch("src.main._parse_glossaries", return_value=([], []))
        mocker.patch("src.main.fetch_cost")

        mock_llm = AsyncMock()
        mock_llm.get_reply_async.side_effect = [
            ("Review 1", "id-1"),
            ("Review 2", "id-2"),
        ]
        mocker.patch("src.main.LLM", return_value=mock_llm)

        src = "Section A\n----------\nSection B"
        tgt = "Раздел А\n----------\nРаздел Б"

        cfg = Settings(
            llm=LLMSettings(api_key=SecretStr("test-key")),
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text=src,
                target_text=tgt,
            ),
            chunk=ChunkSettings(divider="-", max_concurrent=1, delay_seconds=0),
        )

        result = await main(cfg)
        assert result is not None
        assert "Review 1" in result
        assert "Review 2" in result

    @pytest.mark.asyncio
    async def test_divider_review_mismatch_raises(self, mocker):
        mocker.patch("src.main._parse_glossaries", return_value=([], []))

        src = "Section A\n----------\nSection B"
        tgt = "Single chunk"

        cfg = Settings(
            llm=LLMSettings(api_key=SecretStr("test-key")),
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text=src,
                target_text=tgt,
            ),
            chunk=ChunkSettings(divider="-"),
        )

        with pytest.raises(ValueError, match="Manual chunk count mismatch"):
            await main(cfg)


class TestPrintPromptOnly:
    @pytest.mark.asyncio
    async def test_print_prompt_only(self, mocker):
        mocker.patch("src.main._parse_glossaries", return_value=([], []))
        mock_llm_class = mocker.patch("src.main.LLM")

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="Hello world.",
            ),
            output_data=OutputData(print_prompt_only=True),
        )

        result = await main(cfg)
        assert result is None
        mock_llm_class.assert_not_called()


class TestChunkFailure:
    @pytest.mark.asyncio
    async def test_one_chunk_fails_others_succeed(self, mocker):
        mocker.patch("src.main._parse_glossaries", return_value=([], []))
        mocker.patch("src.main.fetch_cost")

        mock_llm = AsyncMock()
        mock_llm.get_reply_async.side_effect = [
            ("Chunk 1 OK", "id-1"),
            RuntimeError("Simulated failure"),
            ("Chunk 3 OK", "id-3"),
        ]
        mocker.patch("src.main.LLM", return_value=mock_llm)

        text = "Chunk A\n----------\nChunk B\n----------\nChunk C"
        cfg = Settings(
            llm=LLMSettings(api_key=SecretStr("test-key")),
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text=text,
            ),
            chunk=ChunkSettings(divider="-", max_concurrent=1, delay_seconds=0),
        )

        result = await main(cfg)
        assert result is not None
        assert "Chunk 1 OK" in result
        assert "Chunk 3 OK" in result
        assert "Chunk 2" not in result


class TestExportGlossaryMatches:
    def test_text_matched_against_auto_glossary(self, mocker, en_lang: Lang, ru_lang: Lang):
        user_entries = [_make_entry(10, "flow meter", "расходомер", "flow meter", "расходомер")]
        auto_entries = [
            _make_entry(1, "pressure valve", "клапан давления", "pressure valve", "клапан давления"),
        ]
        matched_entries = [
            _make_entry(1, "pressure valve", "клапан давления", "pressure valve", "клапан давления"),
        ]

        mock_lemmatizer = mocker.MagicMock()
        mocker.patch("src.main.Lemmatizer", return_value=mock_lemmatizer)
        mocker.patch("src.main._parse_glossaries", return_value=(auto_entries, user_entries))

        mock_matcher = mocker.patch("src.main.TermMatcher")
        mock_matcher_instance = mock_matcher.return_value
        mock_matcher_instance.match.return_value = matched_entries

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="The pressure valve is broken.",
            ),
        )

        result = export_glossary_matches(cfg)
        assert "pressure valve = клапан давления" in result
        assert "flow meter = расходомер" in result

    def test_no_auto_glossary(self, mocker):
        mocker.patch("src.main.Lemmatizer")
        mocker.patch("src.main._parse_glossaries", return_value=([], []))

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="Some text.",
            ),
        )

        result = export_glossary_matches(cfg)
        assert result == ""


class TestResolveAndLogCost:
    @pytest.mark.asyncio
    async def test_known_costs(self, mocker):
        mocker.patch("src.main.fetch_cost", side_effect=[1.50, 2.50])

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url="https://api.example.com/cost"),
        )

        await _resolve_and_log_cost(["id-1", "id-2"], "test-key", cfg)

    @pytest.mark.asyncio
    async def test_unknown_costs(self, mocker):
        mocker.patch("src.main.fetch_cost", side_effect=[1.50, None])

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url="https://api.example.com/cost"),
        )

        await _resolve_and_log_cost(["id-1", "id-2"], "test-key", cfg)

    @pytest.mark.asyncio
    async def test_no_url_configured(self, mocker):
        mock_fetch = mocker.patch("src.main.fetch_cost")

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url=None),
        )

        await _resolve_and_log_cost(["id-1"], "test-key", cfg)
        mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_completion_ids(self, mocker):
        mock_fetch = mocker.patch("src.main.fetch_cost")

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="x",
            ),
            cost=CostSettings(generation_info_url="https://api.example.com/cost"),
        )

        await _resolve_and_log_cost([], "test-key", cfg)
        mock_fetch.assert_not_called()
