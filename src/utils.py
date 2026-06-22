"""Utility functions for file reading and language validation.

Provides helpers for reading text files, validating language codes, and
wrapping low-level file I/O with consistent error handling.
"""

from pathlib import Path

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue


def validate_lang(raw_lang: str) -> Lang:
    """Validate and return a Lang object from an ISO 639-1 code or full name.

    Args:
        raw_lang: A language string (e.g. ``"en"``, ``"English"``).

    Returns:
        The corresponding ``Lang`` instance.

    Raises:
        InvalidLanguageValue: If the string cannot be parsed as a valid language.
    """
    try:
        return Lang(raw_lang)
    except InvalidLanguageValue as e:
        raise e


def _read_file(
    fp: str | Path, *, read_lines: bool, test_readability: bool = False
) -> str | list[str]:
    """Read a file, optionally as lines or a readability check.

    Args:
        fp: Path to the file.
        read_lines: If True, returns a list of lines.
        test_readability: If True, reads only the first 1024 bytes.

    Returns:
        File contents as a string or a list of lines.

    Raises:
        IOError: If the file cannot be found or read.
        OSError: On OS-level read error.
        UnicodeDecodeError: If the file is binary or has an incompatible encoding.
    """
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
    """Check that a file is readable by attempting to read its first 1024 bytes.

    Args:
        fp: Path to the file.

    Raises:
        IOError: If the file cannot be found or read.
        OSError: On OS-level read error.
        UnicodeDecodeError: If the file is binary or has an incompatible encoding.
    """
    _read_file(fp, read_lines=False, test_readability=True)


def read_str_from_file(fp: str | Path) -> str:
    """Read an entire file and return its contents as a string.

    Args:
        fp: Path to the file.

    Returns:
        The file contents as a single string.

    Raises:
        IOError: If the file cannot be found or read.
        OSError: On OS-level read error.
        UnicodeDecodeError: If the file is binary or has an incompatible encoding.
    """
    text = _read_file(fp, read_lines=False)
    assert isinstance(text, str)
    return text


def read_lines_from_file(fp: str | Path) -> list[str]:
    """Read a file and return its contents as a list of lines.

    Args:
        fp: Path to the file.

    Returns:
        The file contents as a list of strings, one per line.

    Raises:
        IOError: If the file cannot be found or read.
        OSError: On OS-level read error.
        UnicodeDecodeError: If the file is binary or has an incompatible encoding.
    """
    lines = _read_file(fp, read_lines=True)
    assert isinstance(lines, list)
    return lines
