"""Data models for glossary entries, terms, cache state, and diffs."""

import hashlib
from pathlib import Path
from typing import NamedTuple

from iso639 import Lang


class Term:
    """A single glossary term in a specific language, optionally lemmatized.

    Attributes:
        language: The ISO 639 language of this term.
        value: The original (raw) text of the term.
        lemmatized: The lemmatized form of the term, or None if not lemmatized.
    """

    def __init__(
        self, language: Lang, value: str, lemmatized: str | None = None
    ) -> None:
        """Initialize a Term.

        Args:
            language: The ISO 639 language of this term.
            value: The original (raw) text of the term.
            lemmatized: The lemmatized form of the term, or None.
        """
        self.language = language
        self.value = value
        self.lemmatized = lemmatized

    def __eq__(self, other) -> bool:
        """Compare two Terms by language and value (ignoring lemmatized)."""
        if not isinstance(other, Term):
            return NotImplemented
        # ignore lemmatized intentionally
        return self.language.pt1 == other.language.pt1 and self.value == other.value

    def __hash__(self):
        """Hash based on language and value."""
        return hash((self.language.pt1, self.value))


class GlossaryEntry:
    """A glossary entry consisting of a unique ID and a set of terms.

    Attributes:
        id: Unique numeric identifier for this entry.
        terms: Frozen set of Term objects belonging to this entry.
    """

    def __init__(self, id: int, terms: frozenset[Term]) -> None:
        """Initialize a GlossaryEntry.

        Args:
            id: Unique numeric identifier for this entry.
            terms: Frozen set of Term objects belonging to this entry.
        """
        self.id = id
        self.terms = terms

    def stringify(self, source_lang: Lang, target_lang: Lang) -> str | None:
        """Format the entry as a 'source = target' string for display.

        Args:
            source_lang: The source language to select terms for.
            target_lang: The target language to select terms for.

        Returns:
            A string like "term1 | term2 = trans1 | trans2", or None if the
            entry's languages don't match the requested pair.
        """
        entry_langs = {t.language.pt1 for t in self.terms}
        requested_langs = {source_lang.pt1, target_lang.pt1}
        if entry_langs != requested_langs:
            return None

        source = " | ".join([t.value for t in self.terms if t.language == source_lang])
        target = " | ".join([t.value for t in self.terms if t.language == target_lang])
        return " = ".join((source, target))

    def __eq__(self, other):
        """Compare two GlossaryEntry objects by ID and terms."""
        if not isinstance(other, GlossaryEntry):
            return NotImplemented

        if self.id != other.id:
            return False

        return self.terms == other.terms


class GlossaryFile:
    """Represents a glossary XML file with its name, path, and content hash.

    Attributes:
        file_path: Path to the XML file.
        glossary_name: Stem (filename without extension) of the glossary file.
        hash: SHA-256 hex digest of the file contents.
    """

    def __init__(self, xml_file_path: str | Path) -> None:
        """Initialize a GlossaryFile.

        Args:
            xml_file_path: Path to the glossary XML file.
        """
        self.file_path = xml_file_path
        self.glossary_name = self._get_glossary_name()
        self.hash = self._get_file_hash()

    def _get_glossary_name(self) -> str:
        """Extract the glossary name from the file stem."""
        return Path(self.file_path).stem

    def _get_file_hash(self) -> str:
        """Compute the SHA-256 hex digest of the file contents."""
        with open(self.file_path, "rb") as f:
            digest = hashlib.file_digest(f, "sha256")

        return digest.hexdigest()


class GlossaryDiff(NamedTuple):
    """Named tuple representing the difference between old and new glossary states.

    Attributes:
        to_add: Brand new entries that need full lemmatization.
        to_update: Changed entries that need re-lemmatization.
        to_delete: IDs of entries that were removed.
    """

    to_add: list[GlossaryEntry]
    to_update: list[GlossaryEntry]
    to_delete: list[int]


class CachedEntries:
    """Container for cached glossary entries with a freshness flag.

    Attributes:
        entries: List of cached GlossaryEntry objects.
        are_up_to_date: True if the cache matches the current XML file hash.
    """

    def __init__(self, entries: list[GlossaryEntry], are_up_to_date: bool) -> None:
        """Initialize CachedEntries.

        Args:
            entries: List of cached GlossaryEntry objects.
            are_up_to_date: Whether the cache is up to date with the source XML.
        """
        self.entries = entries
        self.are_up_to_date = are_up_to_date
