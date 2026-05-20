import ahocorasick
from iso639 import Lang

from src.glossary.models import GlossaryEntry, Term
from src.lemmatizer import GlossaryLemmatizer, Lemmatizer


class TermMatcher:
    def __init__(
        self,
        glossary_entries: list[GlossaryEntry],
    ) -> None:
        self.entries = glossary_entries

    def _flatten_terms(self) -> list[Term]:
        terms: list[Term] = []
        for entry in self.entries:
            for term in entry.terms:
                if term.lemmatized:
                    terms.append(term)

        return terms

    def build_automaton(self, lemmatized_terms: list[str]) -> ahocorasick.Automaton:
        """
        Build an Aho-Corasick automaton over lemmatized terms. Returns the automaton.
        """
        automaton = ahocorasick.Automaton()

        for term in lemmatized_terms:
            automaton.add_word(term, term)

        automaton.make_automaton()

        return automaton

    def _build_term_entry_mapping(self) -> None:
        pass

    def find_glossary_entries_in_text(
        self,
        text: str,
        automaton: ahocorasick.Automaton,
        lang: Lang,
        lemmatizer: Lemmatizer,
    ) -> list[tuple[int, int, str]]:
        """
        Find all terms in text, resolving overlaps by keeping longest match.
        """
        # TODO: Fix: Finds more than needed.
        # NO (Normally Open), IS (Intrinsically Safe), CAN (Cancelled), can (обечайка),
        # RoW (Right of Way), Release, forced, contracted in almost every text.

        # Tokenize and lemmatize the text
        tokens = lemmatizer.lemmatize(text, lang)
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

        resolved: list[tuple[int, int, str]] = []
        occupied = set()

        for start_tok, end_tok, matched_lemma in raw_matches:
            span = set(range(start_tok, end_tok + 1))
            if span & occupied:
                continue
            occupied |= span
            resolved.append((start_tok, end_tok, matched_lemma))

        resolved.sort(key=lambda x: x[0])
        return resolved

    def _build_term_entry_map(self) -> dict[str, list[GlossaryEntry]]:
        term_entry_map: dict[str, list[GlossaryEntry]] = dict()
        for entry in self.entries:
            for term in entry.terms:
                if not term.lemmatized:
                    continue
                if term.lemmatized in term_entry_map:
                    term_entry_map[term.lemmatized].append(entry)
                else:
                    term_entry_map[term.lemmatized] = [entry]

        return term_entry_map

    def match(
        self, text: str, lang: Lang, lemmatizer: Lemmatizer | None = None
    ) -> list[GlossaryEntry]:
        if lemmatizer is None:
            lemmatizer = Lemmatizer()

        flat_terms = self._flatten_terms()
        automaton = self.build_automaton(
            [ft.lemmatized for ft in flat_terms if ft.lemmatized is not None]
        )

        term_entry_map = self._build_term_entry_map()
        found = self.find_glossary_entries_in_text(
            text=text, automaton=automaton, lang=lang, lemmatizer=lemmatizer
        )

        matched_entries: list[GlossaryEntry] = []
        for _, _, matched_lemma in found:
            entries = term_entry_map.get(matched_lemma)
            if entries:
                matched_entries.extend(
                    [e for e in entries if e.id not in [e.id for e in matched_entries]]
                )

        return matched_entries


def example(lang: str) -> None:
    raw_terms: dict[tuple[str, ...], tuple[str, ...]] = {
        ("COMPANY", "CPY"): ("КОМПАНИЯ", "КОМП"),
        ("COMPANY MANUFACTURER",): ("КОМПАНИЯ ПРОИЗВОДИТЕЛЬ",),
        ("CONTRACTOR", "CTR"): ("ПОДРЯДЧИК", "ПОДР"),
        ("RETURN CERTIFICATE",): ("СЕРТИФИКАТ ВОЗВРАТА СООРУЖЕНИЙ",),
        ("geodetic survey project",): ("Проект производства геодезических работ",),
        ("self-propelled trolley",): ("Самоходная тележка",),
        ("variable frequency drive", "drives"): ("Частотно-регулируемый привод",),
        ("running",): ("работает",),
    }

    entries: list[GlossaryEntry] = []
    for idx, (en, ru) in enumerate(raw_terms.items()):
        en_terms = [Term(language=Lang("en"), value=value) for value in en]
        ru_terms = [Term(language=Lang("ru"), value=value) for value in ru]
        terms = frozenset(en_terms + ru_terms)
        entry = GlossaryEntry(id=idx, terms=terms)
        entries.append(entry)

    lemmatizer = Lemmatizer()
    glossary_lemmatizer = GlossaryLemmatizer(lemmatizer)
    lemmatized_entries = glossary_lemmatizer.lemmatize_entries(entries)

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
    Система по-прежнему работает без нареканий
    """

    text: str | None = None
    if lang == "en":
        text = en_text
    elif lang == "ru":
        text = ru_text
    else:
        raise ValueError(f"Wrong language: {lang}")

    term_matcher = TermMatcher(glossary_entries=lemmatized_entries)
    matched_entries = term_matcher.match(
        text=text, lang=Lang(lang), lemmatizer=lemmatizer
    )
    for me in matched_entries:
        print(
            # TODO: Fix glossary entry stringifying in matcher example
            me.to_string(
                source_lang=Lang(lang), target_lang=Lang("ru" if lang == "en" else "en")
            )
        )


# ---- Example usage ----
if __name__ == "__main__":
    lang = input("Language ('en' or 'ru'): ")
    example(lang)
