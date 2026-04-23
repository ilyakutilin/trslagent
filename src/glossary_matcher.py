from typing import List, Tuple

import ahocorasick
from iso639 import Lang

from src.lemmatizer import Lemmatizer
from src.parser import Term


def build_automaton(lemmatized_terms: list[str]) -> ahocorasick.Automaton:
    """
    Build an Aho-Corasick automaton over lemmatized terms. Returns the automaton.
    """
    automaton = ahocorasick.Automaton()

    for term in lemmatized_terms:
        automaton.add_word(term, term)

    automaton.make_automaton()

    return automaton


def find_terms_in_text(
    text: str, lang: Lang, automaton: ahocorasick.Automaton, terms: list[Term]
) -> List[Tuple[int, int, str]]:
    """
    Find all terms in text, resolving overlaps by keeping longest match.
    Returns list of (start_word_idx, end_word_idx, original_term).
    """
    # Tokenize and lemmatize the text
    lemmatizer = Lemmatizer(text, lang)
    tokens = lemmatizer.lemmatize()
    if not tokens:
        return []

    # Build lemma string and track token start positions within it
    token_start_positions = []
    pos = 0
    for lemma in tokens:
        token_start_positions.append(pos)
        pos += len(lemma) + 1  # +1 for space separator

    lemma_string = " ".join(tokens)

    def char_pos_to_token_idx(char_pos: int) -> int:
        """Binary search: which token does this char position belong to?"""
        lo, hi = 0, len(token_start_positions) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if token_start_positions[mid] <= char_pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # Search with Aho-Corasick
    raw_matches = []
    for end_char_idx, matched_lemma in automaton.iter(lemma_string):
        start_char_idx = end_char_idx - len(matched_lemma) + 1

        start_tok = char_pos_to_token_idx(start_char_idx)
        end_tok = char_pos_to_token_idx(end_char_idx)

        # Enforce token-boundary alignment
        if token_start_positions[start_tok] != start_char_idx:
            continue
        expected_end = token_start_positions[end_tok] + len(tokens[end_tok]) - 1
        if expected_end != end_char_idx:
            continue

        raw_matches.append((start_tok, end_tok, matched_lemma))

    # Resolve overlaps: longest match wins
    raw_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    resolved = []
    occupied = set()

    for start_tok, end_tok, matched_lemma in raw_matches:
        span = set(range(start_tok, end_tok + 1))
        if span & occupied:
            continue
        occupied |= span
        for orig in [t.value for t in terms if t.lemmatized == matched_lemma]:
            resolved.append((start_tok, end_tok, orig))

    resolved.sort(key=lambda x: x[0])
    return resolved


# ---- Example usage ----
if __name__ == "__main__":
    LANG = "ru"

    en_raw_terms = [
        "COMPANY",
        "COMPANY MANUFACTURER",
        "CONTRACTOR",
        "RETURN CERTIFICATE",
        "geodetic survey project",
        "self-propelled trolley",
        "variable frequency drive",
        "drives",  # single word — will match "drive" and "drives"
        "running",  # lemmatizes to "run"
    ]

    ru_raw_terms = [
        "КОМПАНИЯ",
        "КОМПАНИЯ ПРОИЗВОДИТЕЛЬ",
        "ПОДРЯДЧИК",
        "СЕРТИФИКАТ ВОЗВРАТА СООРУЖЕНИЙ",
        "Проект производства геодезических работ",
        "Самоходная тележка",
        "Частотно-регулируемый привод",
    ]

    en_text = """
    The company manufacturer issued a return certificate to the contractor.
    A geodetic survey project was also approved.
    The self-propelled trolley is equipped with a variable frequency drive.
    The system is still running smoothly across all drives.
    """

    ru_text = """
    Компания производитель выдала сертификат возврата сооружений подрядчику.
    Также был утверждён проект производства геодезических работ.
    Самоходная тележка оснащена частотно-регулируемым приводом.
    """

    lemmatized_terms = []
    terms = []
    raw_terms = ru_raw_terms if LANG == "ru" else en_raw_terms
    for rt in raw_terms:
        lang = Lang(LANG)
        lemmas = Lemmatizer(rt, lang).lemmatize()
        if lemmas is None:
            continue
        lemmatized_term = " ".join(lemmas)
        lemmatized_terms.append(lemmatized_term)
        terms.append(Term(language=lang, value=rt, lemmatized=lemmatized_term))

    print("Building term index...")
    automaton = build_automaton(lemmatized_terms)

    print("Searching text...\n")
    text = ru_text if LANG == "ru" else en_text
    matches = find_terms_in_text(text, lang, automaton, terms)

    for start, end, term in matches:
        print(f"  Token [{start}:{end}]  →  {term}")
