import hashlib
from pathlib import Path
from typing import NamedTuple

from iso639 import Lang


class Term:
    def __init__(
        self, language: Lang, value: str, lemmatized: str | None = None
    ) -> None:
        self.language = language
        self.value = value
        self.lemmatized = lemmatized

    def __eq__(self, other) -> bool:
        if not isinstance(other, Term):
            return NotImplemented
        # ignore lemmatized intentionally
        return self.language.pt1 == other.language.pt1 and self.value == other.value

    def __hash__(self):
        return hash((self.language.pt1, self.value))


class GlossaryEntry:
    def __init__(self, id: int, terms: frozenset[Term]) -> None:
        self.id = id
        self.terms = terms

    def stringify(self, source_lang: Lang, target_lang: Lang) -> str | None:
        entry_langs = {t.language.pt1 for t in self.terms}
        requested_langs = {source_lang.pt1, target_lang.pt1}
        if entry_langs != requested_langs:
            return None

        source = " | ".join([t.value for t in self.terms if t.language == source_lang])
        target = " | ".join([t.value for t in self.terms if t.language == target_lang])
        return " = ".join((source, target))

    def __eq__(self, other):
        if not isinstance(other, GlossaryEntry):
            return NotImplemented

        if self.id != other.id:
            return False

        return self.terms == other.terms


class GlossaryFile:
    def __init__(self, xml_file_path: str | Path) -> None:
        self.file_path = xml_file_path
        self.glossary_name = self._get_glossary_name()
        self.hash = self._get_file_hash()

    def _get_glossary_name(self) -> str:
        return Path(self.file_path).stem

    def _get_file_hash(self) -> str:
        with open(self.file_path, "rb") as f:
            digest = hashlib.file_digest(f, "sha256")

        return digest.hexdigest()


class GlossaryDiff(NamedTuple):
    to_add: list[GlossaryEntry]  # brand new entries (need full lemmatization)
    to_update: list[GlossaryEntry]  # changed entries (need re-lemmatization)
    to_delete: list[int]  # IDs of removed entries


class CachedEntries:
    def __init__(self, entries: list[GlossaryEntry], are_up_to_date: bool) -> None:
        self.entries = entries
        self.are_up_to_date = are_up_to_date
