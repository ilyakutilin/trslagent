import pytest
from iso639 import Lang

from src.glossary.matcher import TermMatcher
from src.glossary.models import GlossaryEntry, Term


class FakeLemmatizer:
    def __init__(self, supported: frozenset[Lang] | None = None):
        if supported is None:
            self._supported = {Lang("en"), Lang("ru")}
        else:
            self._supported = set(supported)

    def lemmatize(self, text: str, lang: Lang) -> list[str] | None:
        if lang not in self._supported:
            return None
        return text.lower().split()


def _make_entry(entry_id: int, *en_values: str) -> GlossaryEntry:
    en_terms = [Term(Lang("en"), v, lemmatized=v.lower()) for v in en_values]
    ru_terms = [
        Term(Lang("ru"), f"ru_{v}", lemmatized=f"ru_{v.lower()}")
        for v in en_values
    ]
    return GlossaryEntry(id=entry_id, terms=frozenset(en_terms + ru_terms))


class TestBuildAutomaton:
    def test_terms_inserted_can_iterate_matches(self):
        matcher = TermMatcher([])
        automaton = matcher.build_automaton(["flow meter", "pressure valve"])

        matches = list(automaton.iter("the flow meter is broken"))
        found = [m[1] for m in matches]
        assert "flow meter" in found

    def test_no_matches_in_text(self):
        matcher = TermMatcher([])
        automaton = matcher.build_automaton(["flow meter"])

        matches = list(automaton.iter("completely unrelated text"))
        assert len(matches) == 0


class TestFindGlossaryEntriesInText:
    @pytest.fixture
    def lemmatizer(self):
        return FakeLemmatizer()

    def test_no_matches(self, lemmatizer):
        matcher = TermMatcher([])
        automaton = matcher.build_automaton(["flow meter"])

        result = matcher.find_glossary_entries_in_text(
            "completely unrelated text here",
            automaton=automaton,
            lang=Lang("en"),
            lemmatizer=lemmatizer,
        )
        assert result == []

    def test_single_exact_match(self, lemmatizer):
        matcher = TermMatcher([])
        automaton = matcher.build_automaton(["flow meter"])

        result = matcher.find_glossary_entries_in_text(
            "install the flow meter today",
            automaton=automaton,
            lang=Lang("en"),
            lemmatizer=lemmatizer,
        )
        assert len(result) == 1
        assert result[0][2] == "flow meter"

    def test_overlapping_matches_longest_wins(self, lemmatizer):
        matcher = TermMatcher([])
        automaton = matcher.build_automaton(["flow", "flow meter"])

        result = matcher.find_glossary_entries_in_text(
            "the flow meter is calibrated",
            automaton=automaton,
            lang=Lang("en"),
            lemmatizer=lemmatizer,
        )
        assert len(result) == 1
        assert result[0][2] == "flow meter"

    def test_multi_word_term(self, lemmatizer):
        matcher = TermMatcher([])
        automaton = matcher.build_automaton(["pressure relief valve"])

        result = matcher.find_glossary_entries_in_text(
            "inspect the pressure relief valve carefully",
            automaton=automaton,
            lang=Lang("en"),
            lemmatizer=lemmatizer,
        )
        assert len(result) == 1
        assert result[0][2] == "pressure relief valve"

    def test_unsupported_language_returns_empty(self, lemmatizer):
        lemmatizer._supported = set()
        matcher = TermMatcher([])
        automaton = matcher.build_automaton(["flow meter"])

        result = matcher.find_glossary_entries_in_text(
            "the flow meter",
            automaton=automaton,
            lang=Lang("fr"),
            lemmatizer=lemmatizer,
        )
        assert result == []

    def test_token_boundary_subword_not_matched(self, lemmatizer):
        matcher = TermMatcher([])
        automaton = matcher.build_automaton(["flow"])

        result = matcher.find_glossary_entries_in_text(
            "the flowmeter is broken",
            automaton=automaton,
            lang=Lang("en"),
            lemmatizer=lemmatizer,
        )
        assert result == []

    def test_token_boundary_exact_token_matched(self, lemmatizer):
        matcher = TermMatcher([])
        automaton = matcher.build_automaton(["flow"])

        result = matcher.find_glossary_entries_in_text(
            "the flow is steady",
            automaton=automaton,
            lang=Lang("en"),
            lemmatizer=lemmatizer,
        )
        assert len(result) == 1
        assert result[0][2] == "flow"


class TestBuildTermEntryMap:
    def test_correct_mapping(self):
        e1 = _make_entry(1, "flow meter")
        e2 = _make_entry(2, "pressure valve", "relief valve")
        matcher = TermMatcher([e1, e2])

        mapping = matcher._build_term_entry_map()
        assert "flow meter" in mapping
        assert mapping["flow meter"] == [e1]
        assert "pressure valve" in mapping
        assert mapping["pressure valve"] == [e2]
        assert "relief valve" in mapping
        assert mapping["relief valve"] == [e2]

    def test_skips_unlemmatized_terms(self):
        raw_term = Term(Lang("en"), "unprocessed", lemmatized=None)
        entry = GlossaryEntry(id=1, terms=frozenset([raw_term]))
        matcher = TermMatcher([entry])

        mapping = matcher._build_term_entry_map()
        assert mapping == {}


class TestMatch:
    @pytest.fixture
    def lemmatizer(self):
        return FakeLemmatizer()

    def test_integration_lemmatized_entries_matched(self, lemmatizer):
        e1 = _make_entry(1, "flow meter")
        e2 = _make_entry(2, "pressure valve")
        matcher = TermMatcher([e1, e2])

        result = matcher.match(
            "install the flow meter and check the pressure valve",
            Lang("en"),
            lemmatizer=lemmatizer,
        )
        assert len(result) == 2
        assert e1 in result
        assert e2 in result

    def test_no_match_for_missing_term(self, lemmatizer):
        e1 = _make_entry(1, "flow meter")
        matcher = TermMatcher([e1])

        result = matcher.match(
            "the rocket launched successfully",
            Lang("en"),
            lemmatizer=lemmatizer,
        )
        assert result == []

    def test_no_lemmatized_terms_returns_empty(self, lemmatizer):
        raw_term = Term(Lang("en"), "unprocessed", lemmatized=None)
        entry = GlossaryEntry(id=1, terms=frozenset([raw_term]))
        matcher = TermMatcher([entry])

        result = matcher.match(
            "some text with unprocessed",
            Lang("en"),
            lemmatizer=lemmatizer,
        )
        assert result == []

    def test_duplicate_entry_ids_deduplicated(self, lemmatizer):
        e1 = _make_entry(1, "flow", "meter")
        matcher = TermMatcher([e1])

        result = matcher.match(
            "the flow is measured with a meter",
            Lang("en"),
            lemmatizer=lemmatizer,
        )
        assert result == [e1]

    def test_with_custom_lemmatizer(self, lemmatizer):
        e1 = _make_entry(1, "flow meter")
        matcher = TermMatcher([e1])

        result = matcher.match(
            "the flow meter works", Lang("en"), lemmatizer=lemmatizer
        )
        assert result == [e1]

    def test_with_default_lemmatizer(self, mocker):
        fake = FakeLemmatizer()
        mocker.patch("src.glossary.matcher.Lemmatizer", return_value=fake)
        e1 = _make_entry(1, "flow meter")
        matcher = TermMatcher([e1])

        result = matcher.match("the flow meter works", Lang("en"))
        assert result == [e1]
