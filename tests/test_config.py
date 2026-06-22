import datetime
from pathlib import Path

import pytest
from iso639 import Lang
from src.config import (
    ChunkSettings,
    InputData,
    LLMSettings,
    LogSettings,
    OutputData,
    Settings,
    TomlConfigSource,
    get_settings,
    parse_lang,
)


@pytest.fixture
def reset_toml_path():
    original = Settings._toml_path
    yield
    Settings._toml_path = original


class TestParseLang:
    def test_iso639_1_code_en(self):
        result = parse_lang("en")
        assert isinstance(result, Lang)
        assert result.pt1 == "en"

    def test_iso639_1_code_ru(self):
        result = parse_lang("ru")
        assert isinstance(result, Lang)
        assert result.pt1 == "ru"

    def test_full_name_english(self):
        result = parse_lang("English")
        assert isinstance(result, Lang)
        assert result.pt1 == "en"

    def test_passthrough_lang_object(self):
        lang = Lang("fr")
        result = parse_lang(lang)
        assert result is lang

    def test_wrong_type_raises(self):
        with pytest.raises(ValueError, match="wrong class"):
            parse_lang(123)

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            parse_lang("invalid")


class TestLLMSettingsDefaults:
    def test_defaults(self):
        s = LLMSettings()
        assert s.base_url == "https://openrouter.ai/api/v1"
        assert s.model == "anthropic/claude-3.5-sonnet"
        assert s.temperature is None
        assert s.reasoning_effort is None


class TestChunkSettingsDefaults:
    def test_defaults(self):
        s = ChunkSettings()
        assert s.size == 6000
        assert s.divider is None
        assert s.max_concurrent == 3
        assert s.delay_seconds == 1.5


class TestLogSettingsDefaults:
    def test_defaults(self):
        s = LogSettings()
        assert s.level == "INFO"


class TestInputData:
    def test_raises_when_source_lang_equals_target_lang(self):
        with pytest.raises(ValueError, match="source and target langs"):
            InputData(
                source_lang=Lang("en"), target_lang=Lang("en"), source_text="test"
            )

    def test_reads_source_text_from_file(self, tmp_path):
        fp = tmp_path / "source.txt"
        fp.write_text("File content here.")
        data = InputData(
            source_lang=Lang("en"),
            target_lang=Lang("ru"),
            source_file_path=fp,
        )
        assert data.source_text == "File content here."

    def test_reads_target_text_from_file(self, tmp_path):
        source_fp = tmp_path / "source.txt"
        source_fp.write_text("source")
        target_fp = tmp_path / "target.txt"
        target_fp.write_text("target content")
        data = InputData(
            source_lang=Lang("en"),
            target_lang=Lang("ru"),
            source_file_path=source_fp,
            target_file_path=target_fp,
        )
        assert data.target_text == "target content"

    def test_reads_user_glossary_lines_from_file(self, tmp_path):
        source_fp = tmp_path / "source.txt"
        source_fp.write_text("source")
        gl_fp = tmp_path / "glossary.txt"
        gl_fp.write_text("term = translation\nanother = другой\n")
        data = InputData(
            source_lang=Lang("en"),
            target_lang=Lang("ru"),
            source_file_path=source_fp,
            user_glossary_file_path=str(gl_fp),
        )
        assert data.user_glossary_lines == [
            "term = translation\n",
            "another = другой\n",
        ]

    def test_user_glossary_lines_none_by_default(self, tmp_path):
        fp = tmp_path / "source.txt"
        fp.write_text("source")
        data = InputData(
            source_lang=Lang("en"),
            target_lang=Lang("ru"),
            source_file_path=fp,
        )
        assert data.user_glossary_lines is None

    def test_raises_on_missing_source_file_path(self):
        bad_path = Path("/nonexistent/path.txt")
        data = InputData(
            source_lang=Lang("en"),
            target_lang=Lang("ru"),
            source_file_path=bad_path,
        )
        assert data.source_text is None


class TestOutputData:
    FROZEN_NOW = datetime.datetime(2025, 1, 15, 12, 30, 45)

    @pytest.fixture(autouse=True)
    def freeze_time(self, monkeypatch):
        class frozen_datetime(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                return TestOutputData.FROZEN_NOW

        monkeypatch.setattr("src.config.datetime", frozen_datetime)

    def test_get_result_file_path_with_timestamps(self):
        od = OutputData(
            result_file_path=Path("files/result.md"),
            timestamped_result_filenames=True,
        )
        result = od.get_result_file_path()
        assert result == Path("files/result_2025-01-15_12-30-45.md")

    def test_get_result_file_path_without_timestamps(self):
        od = OutputData(
            result_file_path=Path("files/result.md"),
            timestamped_result_filenames=False,
        )
        result = od.get_result_file_path()
        assert result == Path("files/result.md")


class TestTomlConfigSource:
    def test_reads_values_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('[llm]\nmodel = "test-model"\n[chunk]\nsize = 8000\n')
        source = TomlConfigSource(Settings, toml_path)
        result = source()
        assert result["llm"]["model"] == "test-model"
        assert result["chunk"]["size"] == 8000

    def test_get_field_value_missing_key_returns_none(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('[llm]\nmodel = "test-model"\n')
        source = TomlConfigSource(Settings, toml_path)
        field_info = Settings.model_fields["chunk"]
        val, name, is_complex = source.get_field_value(field_info, "chunk")
        assert val is None
        assert name == "chunk"
        assert is_complex is True


class TestGetSettings:
    def test_integration_with_temp_toml(self, tmp_path, reset_toml_path):
        source_fp = tmp_path / "source.txt"
        source_fp.write_text("dummy")
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            "[chunk]\nsize = 9999\n"
            '[llm]\nmodel = "my-model"\n'
            "[input_data]\n"
            'source_lang = "en"\n'
            'target_lang = "ru"\n'
            f'source_file_path = "{source_fp}"\n'
        )
        settings = get_settings(toml_path)
        assert settings.chunk.size == 9999
        assert settings.llm.model == "my-model"
        assert settings.log.level == "INFO"
