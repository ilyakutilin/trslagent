import sys
from pathlib import Path

from tqdm import tqdm

from src.config import get_settings
from src.glossary.models import GlossaryEntry, Term
from src.glossary.parser import AutoGlossaryParser
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


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {sys.argv[0]} <path/to/config.toml>")

    toml_path = Path(sys.argv[1]).resolve()
    if not toml_path.is_file():
        sys.exit(f"Config file not found: {toml_path}")

    settings = get_settings(toml_path=toml_path)
    lemmatizer = Lemmatizer()
    auto_glossary_entries = AutoGlossaryParser(
        dir_path=settings.glossary.xml_dir_path,
        lemmatizer=lemmatizer,
    ).parse()
    terms = _flatten_terms(auto_glossary_entries)
    capitalized_words: set[str] = set()
    for term in tqdm(terms):
        if is_abbreviation(term.value):
            capitalized_words.add(f"{term.value}\n")
    sorted_words = sorted(list(capitalized_words))

    with open("files/abbrs", mode="w", encoding="utf-8") as f:
        f.writelines(sorted_words)


if __name__ == "__main__":
    main()
