import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from iso639 import Lang

from src.cli import cli
from src.config import Settings


@pytest.fixture(autouse=True)
def _reset_toml_path():
    Settings._toml_path = None
    yield
    Settings._toml_path = None


def _write_toml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


BASIC_TOML = """\
[llm]
api_key = "test-key"

[input_data]
source_lang = "en"
target_lang = "ru"
source_text = "Hello world."
"""


class TestValidTomlPath:
    def test_main_called_with_settings(self, tmp_path, mocker):
        mock_main = mocker.patch("src.cli.main", new_callable=AsyncMock)
        mock_main.return_value = "Translated text"

        toml_path = tmp_path / "config.toml"
        _write_toml(toml_path, BASIC_TOML)

        mocker.patch.object(sys, "argv", ["cli.py", str(toml_path)])
        cli()

        mock_main.assert_called_once()
        cfg = mock_main.call_args.kwargs["cfg"]
        assert cfg.input_data.source_lang == Lang("en")
        assert cfg.input_data.target_lang == Lang("ru")
        assert cfg.input_data.source_text == "Hello world."


class TestMissingTomlArgument:
    def test_prints_usage_and_exits(self, mocker):
        mock_main = mocker.patch("src.cli.main")
        mocker.patch.object(sys, "argv", ["cli.py"])

        with pytest.raises(SystemExit) as exc_info:
            cli()

        assert "Usage" in str(exc_info.value)
        mock_main.assert_not_called()

    def test_extra_arguments(self, mocker):
        mock_main = mocker.patch("src.cli.main")
        mocker.patch.object(sys, "argv", ["cli.py", "a.toml", "b.toml"])

        with pytest.raises(SystemExit) as exc_info:
            cli()

        assert "Usage" in str(exc_info.value)
        mock_main.assert_not_called()


class TestNonexistentTomlFile:
    def test_reports_file_not_found(self, tmp_path, mocker):
        mock_main = mocker.patch("src.cli.main")

        toml_path = tmp_path / "nonexistent.toml"
        mocker.patch.object(sys, "argv", ["cli.py", str(toml_path)])

        with pytest.raises(SystemExit) as exc_info:
            cli()

        assert "not found" in str(exc_info.value)
        mock_main.assert_not_called()


class TestMatchGlossary:
    def test_calls_export_glossary_matches(self, tmp_path, mocker):
        mock_export = mocker.patch(
            "src.cli.export_glossary_matches", return_value="matched entries"
        )
        mock_main = mocker.patch("src.cli.main")

        toml_path = tmp_path / "config.toml"
        _write_toml(toml_path, BASIC_TOML)

        output_path = tmp_path / "matches.txt"
        mocker.patch.object(sys, "argv", [
            "cli.py", "--match-glossary", "--match-output",
            str(output_path), str(toml_path),
        ])
        cli()

        mock_export.assert_called_once()
        mock_main.assert_not_called()
        assert output_path.read_text() == "matched entries"

    def test_missing_match_output_flag(self, tmp_path, mocker):
        mock_export = mocker.patch("src.cli.export_glossary_matches")

        toml_path = tmp_path / "config.toml"
        _write_toml(toml_path, BASIC_TOML)
        mocker.patch.object(sys, "argv", [
            "cli.py", "--match-glossary", str(toml_path),
        ])

        with pytest.raises(SystemExit) as exc_info:
            cli()

        assert "--match-output" in str(exc_info.value)
        mock_export.assert_not_called()

    def test_match_output_without_path_value(self, mocker):
        mock_export = mocker.patch("src.cli.export_glossary_matches")
        mocker.patch.object(sys, "argv", [
            "cli.py", "--match-glossary", "--match-output",
        ])

        with pytest.raises(SystemExit) as exc_info:
            cli()

        assert "path argument" in str(exc_info.value)
        mock_export.assert_not_called()


class TestPrintPromptOnly:
    def test_passed_through_to_config(self, tmp_path, mocker):
        mock_main = mocker.patch("src.cli.main", new_callable=AsyncMock)
        mock_main.return_value = None

        toml_path = tmp_path / "config.toml"
        _write_toml(toml_path, """\
[llm]
api_key = "test-key"

[input_data]
source_lang = "en"
target_lang = "ru"
source_text = "Hello world."

[output_data]
print_prompt_only = true
""")
        mocker.patch.object(sys, "argv", ["cli.py", str(toml_path)])
        cli()

        mock_main.assert_called_once()
        cfg = mock_main.call_args.kwargs["cfg"]
        assert cfg.output_data.print_prompt_only is True
