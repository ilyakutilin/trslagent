import re

import pymorphy3
import spacy
from iso639 import Lang
from tqdm import tqdm

from src.config import logger
from src.glossary.models import GlossaryEntry, Term


class Lemmatizer:
    def __init__(self) -> None:
        logger.warning(f"Lemmatizer initialized: ID {id(self)}")
        self.nlp: spacy.language.Language | None = None
        self.morph: pymorphy3.MorphAnalyzer | None = None

    def _lemmatize_english(self, text: str) -> list[str]:
        if self.nlp is None:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            logger.warning(f"SpaCy NLP created for EN lemmatization: ID {id(self.nlp)}")

        normalized = text.replace("-", " ").replace("–", " ")

        doc = self.nlp(normalized)
        # TODO: 'RFGI(C)' is lemmatized as ['rfgi(c', ')']
        return [token.lemma_.lower() for token in doc]

    def _lemmatize_russian(self, text: str) -> list[str]:
        if self.morph is None:
            self.morph = pymorphy3.MorphAnalyzer()
            logger.warning(
                f"Pymorphy3 MorphAnalyzer created for RU lemmatization: ID {self.morph}"
            )
        words = re.findall(r"[а-яёА-ЯЁa-zA-Z0-9]+", text)

        lemmatized: list[str] = []
        lemma_cache: dict[str, str] = dict()
        for word in words:
            if word in lemma_cache:
                lemmatized.append(lemma_cache[word])
                continue
            parses = self.morph.parse(word)
            lemmatized_word = parses[0].normal_form if parses else word
            lemmatized.append(lemmatized_word)
            lemma_cache[word] = lemmatized_word
        return lemmatized

    def lemmatize(self, text: str, lang: Lang) -> list[str] | None:
        def unsupported(*args, **kwargs):
            return None

        lemmatizers = {
            Lang("en"): self._lemmatize_english,
            Lang("ru"): self._lemmatize_russian,
        }

        return lemmatizers.get(lang, unsupported)(text)


class GlossaryLemmatizer:
    def __init__(
        self,
        lemmatizer: Lemmatizer | None = None,
    ) -> None:
        self.lemmatizer = lemmatizer if lemmatizer else Lemmatizer()

    def lemmatize_entries(self, entries: list[GlossaryEntry]) -> list[GlossaryEntry]:
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
