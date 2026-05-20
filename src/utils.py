from pathlib import Path

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue


def validate_lang(raw_lang: str) -> Lang:
    try:
        return Lang(raw_lang)
    except InvalidLanguageValue as e:
        raise e


def _read_file(
    fp: str | Path, *, read_lines: bool, test_readability: bool = False
) -> str | list[str]:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            if read_lines:
                return f.readlines()

            if test_readability:
                return f.read(1024)

            return f.read()

    except (IOError, OSError, UnicodeDecodeError) as e:
        # IOError/OSError: File not found or no read permission
        # UnicodeDecodeError: File is binary or incompatible encoding
        raise e


def validate_file(fp: str | Path) -> None:
    _read_file(fp, read_lines=False, test_readability=True)


def read_str_from_file(fp: str | Path) -> str:
    text = _read_file(fp, read_lines=False)
    assert isinstance(text, str)
    return text


def read_lines_from_file(fp: str | Path) -> list[str]:
    lines = _read_file(fp, read_lines=True)
    assert isinstance(lines, list)
    return lines
