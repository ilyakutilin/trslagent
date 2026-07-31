from unittest.mock import AsyncMock

import pytest
from iso639 import Lang
from pydantic import SecretStr

from src.config import (
    ChunkSettings,
    InputData,
    LLMSettings,
    OutputData,
    Settings,
)
from src.glossary.models import GlossaryEntry, Term
from src.main import export_glossary_matches, main


@pytest.fixture(autouse=True)
def _reset_toml_path():
    Settings._toml_path = None
    yield
    Settings._toml_path = None


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    from src.config import ProxySettings
    from src.http_client import create_client as _real_create_client

    def _wrapped(*args, **kwargs):
        kwargs["proxy_settings"] = ProxySettings(enabled=False)
        return _real_create_client(*args, **kwargs)

    monkeypatch.setattr("src.main.create_client", _wrapped)


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
        terms=frozenset(
            [
                _make_term("en", en_value, en_lemma),
                _make_term("ru", ru_value, ru_lemma),
            ]
        ),
    )


class TestTranslationPipeline:
    @pytest.mark.asyncio
    async def test_source_text_chunks_translate_stitch(self, mocker):
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=[],
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )
        mocker.patch("src.llm.resolve_and_log_cost")

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
        assert result is not None
        assert result.text == "Переведённый текст"
        mock_llm.get_reply_async.assert_called()

    @pytest.mark.asyncio
    async def test_with_user_glossary_no_auto(self, mocker):
        user_entries = [
            _make_entry(10, "flow meter", "расходомер", "flow meter", "расходомер")
        ]
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=user_entries,
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )
        mocker.patch("src.llm.resolve_and_log_cost")

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
        assert result is not None
        assert result.text == "Перевод"
        mock_llm.get_reply_async.assert_called()

    @pytest.mark.asyncio
    async def test_divider_based_chunking(self, mocker):
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=[],
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )
        mocker.patch("src.llm.resolve_and_log_cost")

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
        assert "Chunk 1" in result.text
        assert "Chunk 2" in result.text


class TestReviewMode:
    @pytest.mark.asyncio
    async def test_non_chunked_review(self, mocker):
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=[],
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )
        mocker.patch("src.llm.resolve_and_log_cost")

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
        assert result is not None
        assert result.text == "Review result"

    @pytest.mark.asyncio
    async def test_divider_review_equal_chunks(self, mocker):
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=[],
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )
        mocker.patch("src.llm.resolve_and_log_cost")

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
        assert "Review 1" in result.text
        assert "Review 2" in result.text

    @pytest.mark.asyncio
    async def test_divider_review_mismatch_raises(self, mocker):
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=[],
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )

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
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=[],
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )
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
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=[],
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )
        mocker.patch("src.llm.resolve_and_log_cost")

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
        text = result.text
        success_count = text.count("Chunk")
        assert "Chunk 1 OK" in text
        assert "Chunk 3 OK" in text
        assert "Chunk 2" not in text
        assert success_count == 2


class TestPipelineLifecycle:
    @pytest.mark.asyncio
    async def test_llm_closed_and_http_client_threaded_to_pipeline(self, mocker):
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(),
        )

        mock_llm = AsyncMock()
        mock_llm.api_key = "test-key"
        mock_llm.get_reply_async.return_value = ("Перевод", "completion-1")
        mocker.patch("src.main.LLM", return_value=mock_llm)

        mock_pipeline = mocker.patch(
            "src.main.run_translation_pipeline", new_callable=AsyncMock
        )
        mock_pipeline.return_value = mocker.MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_create_client = mocker.patch(
            "src.main.create_client", return_value=mock_client
        )

        cfg = Settings(
            llm=LLMSettings(api_key=SecretStr("test-key")),
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="Hello world.",
            ),
            chunk=ChunkSettings(size=1000, max_concurrent=1, delay_seconds=0),
        )

        result = await main(cfg)

        assert result is not None
        mock_create_client.assert_called_once_with(proxy_settings=cfg.proxy)
        mock_client.__aenter__.assert_awaited_once()
        mock_client.__aexit__.assert_awaited_once()
        mock_pipeline.assert_awaited_once()
        assert mock_pipeline.await_args.args[3] is mock_client
        mock_llm.close.assert_awaited_once()


class TestExportGlossaryMatches:
    def test_text_matched_against_auto_glossary(
        self, mocker, en_lang: Lang, ru_lang: Lang
    ):
        user_entries = [
            _make_entry(10, "flow meter", "расходомер", "flow meter", "расходомер")
        ]
        matched_entries = [
            _make_entry(
                1,
                "pressure valve",
                "клапан давления",
                "pressure valve",
                "клапан давления",
            ),
        ]

        mock_matcher = mocker.MagicMock()
        mock_matcher.match.return_value = matched_entries

        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=user_entries,
                term_matcher=mock_matcher,
                source_lang=en_lang,
                target_lang=ru_lang,
            ),
        )

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
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=[],
                term_matcher=None,
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
            ),
        )

        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="Some text.",
            ),
        )

        result = export_glossary_matches(cfg)
        assert result == ""


class TestPipelineResult:
    @pytest.mark.asyncio
    async def test_auto_glossary_matched_count(self, mocker):
        en_lang = Lang("en")
        ru_lang = Lang("ru")
        auto_entries = [
            _make_entry(
                1,
                "pressure valve",
                "клапан давления",
                "pressure valve",
                "клапан давления",
            ),
        ]
        user_entries = [
            _make_entry(10, "flow meter", "расходомер", "flow meter", "расходомер")
        ]

        mock_matcher = mocker.MagicMock()
        mock_matcher.match.return_value = [auto_entries[0]]

        def _select_for_chunk(chunk, mt):
            matched = mock_matcher.match.return_value
            mt[0] += len(matched)
            return ([], [], "", "")

        ctx_mock = mocker.MagicMock(
            user_entries=user_entries,
            term_matcher=mock_matcher,
            source_lang=en_lang,
            target_lang=ru_lang,
            select_for_chunk=_select_for_chunk,
        )
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=ctx_mock,
        )
        mocker.patch("src.llm.resolve_and_log_cost")

        mock_llm = AsyncMock()
        mock_llm.get_reply_async.return_value = ("Перевод", "completion-1")
        mocker.patch("src.main.LLM", return_value=mock_llm)

        cfg = Settings(
            llm=LLMSettings(api_key=SecretStr("test-key")),
            input_data=InputData(
                source_lang=en_lang,
                target_lang=ru_lang,
                source_text="The pressure valve is broken.",
            ),
            chunk=ChunkSettings(size=1000, max_concurrent=1, delay_seconds=0),
        )

        result = await main(cfg)
        assert result is not None
        assert result.auto_glossary_entries_matched == 1
        assert result.user_glossary_entries == 1
        assert result.mode == "translation"

    @pytest.mark.asyncio
    async def test_mode_is_review_when_target_provided(self, mocker):
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=[],
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )
        mocker.patch("src.llm.resolve_and_log_cost")

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
        assert result is not None
        assert result.mode == "review"

    @pytest.mark.asyncio
    async def test_user_glossary_enabled_flag(self, mocker):
        user_entries = [
            _make_entry(10, "flow meter", "расходомер", "flow meter", "расходомер")
        ]
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=user_entries,
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )
        mocker.patch("src.llm.resolve_and_log_cost")

        mock_llm = AsyncMock()
        mock_llm.get_reply_async.return_value = ("Перевод", "completion-1")
        mocker.patch("src.main.LLM", return_value=mock_llm)

        cfg = Settings(
            llm=LLMSettings(api_key=SecretStr("test-key")),
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="Hello.",
                user_glossary_lines=["term = translation"],
            ),
            chunk=ChunkSettings(size=1000, max_concurrent=1, delay_seconds=0),
        )

        result = await main(cfg)
        assert result is not None
        assert result.user_glossary_enabled is True

    @pytest.mark.asyncio
    async def test_auto_glossary_disabled_flag(self, mocker):
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=[],
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )
        mocker.patch("src.llm.resolve_and_log_cost")

        mock_llm = AsyncMock()
        mock_llm.get_reply_async.return_value = ("Перевод", "completion-1")
        mocker.patch("src.main.LLM", return_value=mock_llm)

        cfg = Settings(
            llm=LLMSettings(api_key=SecretStr("test-key")),
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_text="Hello.",
                auto_glossary=False,
            ),
            chunk=ChunkSettings(size=1000, max_concurrent=1, delay_seconds=0),
        )

        result = await main(cfg)
        assert result is not None
        assert result.auto_glossary_enabled is False


class TestMainRaisesOnEmptySource:
    @pytest.mark.asyncio
    async def test_raises_when_source_file_missing(self, mocker, tmp_path):
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=[],
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )

        missing = tmp_path / "nonexistent.txt"
        cfg = Settings(
            input_data=InputData(
                source_lang=Lang("en"),
                target_lang=Lang("ru"),
                source_file_path=missing,
            ),
        )
        with pytest.raises(ValueError, match="Source text is empty"):
            await main(cfg)


class TestAutoDetectTranslation:
    """Translation pipeline with auto-detected languages."""

    @pytest.mark.asyncio
    async def test_both_none_english_source_detected(self, mocker):
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=[],
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )
        mocker.patch("src.llm.resolve_and_log_cost")

        mock_llm = AsyncMock()
        mock_llm.get_reply_async.return_value = ("Перевод", "completion-1")
        mocker.patch("src.main.LLM", return_value=mock_llm)

        cfg = Settings(
            llm=LLMSettings(api_key=SecretStr("test-key")),
            input_data=InputData(
                source_lang=None,
                target_lang=None,
                source_text=(
                    "This is an English document that needs to be translated "
                    "into another language. The architecture is described here."
                ),
            ),
            chunk=ChunkSettings(size=1000, max_concurrent=1, delay_seconds=0),
        )

        result = await main(cfg)
        assert result is not None
        assert result.text == "Перевод"
        assert cfg.input_data.source_lang == Lang("en")
        assert cfg.input_data.target_lang == Lang("ru")

    @pytest.mark.asyncio
    async def test_source_set_target_none_defaults_ru(self, mocker):
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=[],
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )
        mocker.patch("src.llm.resolve_and_log_cost")

        mock_llm = AsyncMock()
        mock_llm.get_reply_async.return_value = ("Traducción", "completion-1")
        mocker.patch("src.main.LLM", return_value=mock_llm)

        cfg = Settings(
            llm=LLMSettings(api_key=SecretStr("test-key")),
            input_data=InputData(
                source_lang=Lang("fr"),
                target_lang=None,
                source_text="Texte français à traduire.",
            ),
            chunk=ChunkSettings(size=1000, max_concurrent=1, delay_seconds=0),
        )

        result = await main(cfg)
        assert result is not None
        assert result.text == "Traducción"
        assert cfg.input_data.target_lang == Lang("ru")


class TestAutoDetectReview:
    """Review pipeline with auto-detected languages."""

    @pytest.mark.asyncio
    async def test_both_none_review_detected(self, mocker):
        mocker.patch(
            "src.main.prepare_glossary_context",
            return_value=mocker.MagicMock(
                user_entries=[],
                select_for_chunk=lambda chunk, mt: ([], [], "", ""),
            ),
        )
        mocker.patch("src.llm.resolve_and_log_cost")

        mock_llm = AsyncMock()
        mock_llm.get_reply_async.return_value = ("Review OK", "id-1")
        mocker.patch("src.main.LLM", return_value=mock_llm)

        cfg = Settings(
            llm=LLMSettings(api_key=SecretStr("test-key")),
            input_data=InputData(
                source_lang=None,
                target_lang=None,
                source_text=(
                    "This is an English source document for review. "
                    "It contains enough text for reliable detection."
                ),
                target_text=(
                    "Это перевод на русский язык для проверки. "
                    "Текст содержит достаточно информации."
                ),
            ),
        )

        result = await main(cfg)
        assert result is not None
        assert result.text == "Review OK"
        assert cfg.input_data.source_lang == Lang("en")
        assert cfg.input_data.target_lang == Lang("ru")
