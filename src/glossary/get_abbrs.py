from tqdm import tqdm

from src.config import get_settings
from src.glossary.models import GlossaryEntry, Term
from src.glossary.parser import MainGlossaryParser
from src.lemmatizer import Lemmatizer


def is_abbreviation(word: str) -> bool:
    return (
        word.isupper()
        or (
            not word.istitle()
            and not word.islower()
            and any(c.isupper() for c in word[1:])
        )
    ) and len(word) <= 8


def _flatten_terms(entries: list[GlossaryEntry]) -> list[Term]:
    terms: list[Term] = []
    for entry in entries:
        for term in entry.terms:
            if term.lemmatized:
                terms.append(term)

    return terms


settings = get_settings()
lemmatizer = Lemmatizer()
main_glossary_entries = MainGlossaryParser(
    dir_path=settings.glossary.xml_dir_path,
    lemmatizer=lemmatizer,
).parse()
terms = _flatten_terms(main_glossary_entries)
capitalized_words: set[str] = set()
for term in tqdm(terms):
    if is_abbreviation(term.value):
        capitalized_words.add(f"{term.value}\n")
sorted_words = sorted(list(capitalized_words))

with open("files/abbrs", mode="w", encoding="utf-8") as f:
    f.writelines(sorted_words)
