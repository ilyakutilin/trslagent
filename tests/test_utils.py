import pytest
from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue

from src.utils import (
    read_lines_from_file,
    read_str_from_file,
    validate_file,
    validate_lang,
)


class TestValidateLang:
    def test_iso639_1_code(self):
        result = validate_lang("en")
        assert isinstance(result, Lang)
        assert result.pt1 == "en"

    def test_full_english_name(self):
        result = validate_lang("English")
        assert isinstance(result, Lang)
        assert result.pt1 == "en"

    def test_russian(self):
        result = validate_lang("ru")
        assert isinstance(result, Lang)
        assert result.pt1 == "ru"

    def test_invalid_string(self):
        with pytest.raises(InvalidLanguageValue):
            validate_lang("invalid")

    def test_wrong_type(self):
        with pytest.raises(InvalidLanguageValue):
            validate_lang(42)  # type: ignore


class TestReadStrFromFile:
    def test_reads_content(self, tmp_path):
        fp = tmp_path / "test.txt"
        content = "Hello, world!\nLine two."
        fp.write_text(content)
        result = read_str_from_file(fp)
        assert result == content

    def test_nonexistent_raises(self, tmp_path):
        fp = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError):
            read_str_from_file(fp)


class TestReadLinesFromFile:
    def test_reads_lines(self, tmp_path):
        fp = tmp_path / "lines.txt"
        lines = ["line one\n", "line two\n", "line three\n"]
        fp.write_text("".join(lines))
        result = read_lines_from_file(fp)
        assert result == lines


class TestValidateFile:
    def test_readable_file_passes(self, tmp_path):
        fp = tmp_path / "good.txt"
        fp.write_text("some content")
        assert validate_file(fp) is None

    def test_nonexistent_raises(self, tmp_path):
        fp = tmp_path / "missing.txt"
        with pytest.raises(FileNotFoundError):
            validate_file(fp)
