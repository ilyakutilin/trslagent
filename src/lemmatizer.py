"""Lemmatization utilities for glossary term matching.

Provides per-language lemmatizers (spaCy for English, pymorphy3 for Russian)
and a higher-level lemmatizer that processes glossary entries.
"""

import re
from pathlib import Path

import pymorphy3
import spacy
from iso639 import Lang
from tqdm import tqdm

from src.config import logger
from src.glossary.models import GlossaryEntry, Term


def parse_known_abbrs(abbr_file_path: str | Path) -> set[str]:
    """Load known abbreviations from a plain-text file (one per line).

    Args:
        abbr_file_path: Path to the abbreviations file.

    Returns:
        A set of non-empty abbreviation strings.
    """
    fp = Path(abbr_file_path)
    with open(fp, "r", encoding="utf-8") as f:
        abbrs = {line.strip() for line in f if line.strip()}
    return abbrs


class Lemmatizer:
    """Language-aware lemmatizer with known-abbreviation preservation.

    Supports English (spaCy) and Russian (pymorphy3). Words present in
    *known_abbrs* are returned as-is instead of being lemmatized.

    Attributes:
        known_abbrs: Set of abbreviations to pass through unchanged.
    """

    def __init__(self, known_abbrs: set[str] = set()) -> None:
        """Initialize the lemmatizer.

        Args:
            known_abbrs: Optional set of abbreviation strings to preserve.
        """
        self.nlp: spacy.language.Language | None = None
        self.morph: pymorphy3.MorphAnalyzer | None = None
        self.known_abbrs = known_abbrs
        known_abbrs_block = (
            f"{len(self.known_abbrs)} known abbrs loaded"
            if self.known_abbrs
            else "known abbrs NOT PROVIDED"
        )
        logger.warning(f"Lemmatizer initialized: ID {id(self)} | {known_abbrs_block}")

    def _lemmatize_english(self, text: str) -> list[str]:
        """Lemmatize an English text using spaCy.

        Args:
            text: The English text to lemmatize.

        Returns:
            A list of lemmatized tokens (lowercased), with known abbreviations
            kept as-is.
        """
        if self.nlp is None:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            logger.warning(f"SpaCy NLP created for EN lemmatization: ID {id(self.nlp)}")

        normalized = text.replace("-", " ").replace("–", " ")

        doc = self.nlp(normalized)
        lemmatized: list[str] = []
        for token in doc:
            if token.text in self.known_abbrs:
                lemmatized.append(token.text)
            else:
                lemmatized.append(token.lemma_.lower())
        return lemmatized

    def _lemmatize_russian(self, text: str) -> list[str]:
        """Lemmatize a Russian text using pymorphy3.

        Args:
            text: The Russian text to lemmatize.

        Returns:
            A list of lemmatized tokens, with known abbreviations kept as-is.
        """
        if self.morph is None:
            self.morph = pymorphy3.MorphAnalyzer()
            logger.warning(
                f"Pymorphy3 MorphAnalyzer created for RU lemmatization: ID {self.morph}"
            )
        words = re.findall(r"[а-яёА-ЯЁa-zA-Z0-9]+", text)

        lemmatized: list[str] = []
        lemma_cache: dict[str, str] = dict()
        for word in words:
            if word in self.known_abbrs:
                lemmatized.append(word)
                continue

            if word in lemma_cache:
                lemmatized.append(lemma_cache[word])
                continue

            parses = self.morph.parse(word)
            lemmatized_word = parses[0].normal_form if parses else word
            lemmatized.append(lemmatized_word)
            lemma_cache[word] = lemmatized_word
        return lemmatized

    def lemmatize(self, text: str, lang: Lang) -> list[str] | None:
        """Lemmatize text in the given language.

        Args:
            text: The text to lemmatize.
            lang: Language of the text (only "en" and "ru" are supported).

        Returns:
            A list of lemmatized tokens, or None if the language is not
            supported.
        """

        def unsupported(*args, **kwargs):
            return None

        lemmatizers = {
            Lang("en"): self._lemmatize_english,
            Lang("ru"): self._lemmatize_russian,
        }

        return lemmatizers.get(lang, unsupported)(text)


class GlossaryLemmatizer:
    """Processes glossary entries, adding lemmatized forms to each term.

    Attributes:
        lemmatizer: The underlying :class:`Lemmatizer` instance used for
            per-term lemmatization.
    """

    def __init__(
        self,
        lemmatizer: Lemmatizer | None = None,
    ) -> None:
        """Initialize the glossary lemmatizer.

        Args:
            lemmatizer: Optional :class:`Lemmatizer` instance. If omitted, a
                default one is created.
        """
        self.lemmatizer = lemmatizer if lemmatizer else Lemmatizer()

    def lemmatize_entries(self, entries: list[GlossaryEntry]) -> list[GlossaryEntry]:
        """Add lemmatized forms to a list of glossary entries.

        Each :class:`Term` within an entry is lemmatized via the underlying
        :class:`Lemmatizer`. If lemmatization fails for a term (unsupported
        language), the term is kept unchanged.

        Args:
            entries: Glossary entries to process.

        Returns:
            New list of glossary entries with lemmatized terms populated.
        """
        updated_entries: list[GlossaryEntry] = []
        for entry in tqdm(entries):
            updated_terms: list[Term] = []
            for term in entry.terms:
                term_lemmas = self.lemmatizer.lemmatize(term.value, term.language)
                if not term_lemmas:
                    logger.warning(f"Term '{term.value}' not lemmatized")
                    updated_terms.append(term)
                    continue
                lemmatized_term = " ".join(term_lemmas)
                updated_term = Term(
                    language=term.language,
                    value=term.value,
                    lemmatized=lemmatized_term,
                )
                updated_terms.append(updated_term)

            updated_entry = GlossaryEntry(id=entry.id, terms=frozenset(updated_terms))
            updated_entries.append(updated_entry)

        return updated_entries
