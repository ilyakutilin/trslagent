import pytest
from iso639 import Lang

from src.glossary.models import GlossaryEntry, Term
from src.lemmatizer import GlossaryLemmatizer, Lemmatizer, parse_known_abbrs


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


@pytest.fixture(scope="session")
def lemmatizer():
    return Lemmatizer()


@pytest.fixture(scope="session")
def lemmatizer_with_abbrs():
    return Lemmatizer(known_abbrs={"NASA", "ГОСТ"})


@pytest.fixture
def fake_lemmatizer():
    return FakeLemmatizer()


class TestParseKnownAbbrs:
    def test_reads_from_temp_file_strips_whitespace(self, tmp_path):
        fp = tmp_path / "abbrs.txt"
        fp.write_text("  ABC  \n  DEF\n\n  GHI  \n")
        result = parse_known_abbrs(fp)
        assert result == {"ABC", "DEF", "GHI"}

    def test_empty_file_returns_empty_set(self, tmp_path):
        fp = tmp_path / "empty.txt"
        fp.write_text("")
        result = parse_known_abbrs(fp)
        assert result == set()

    def test_lines_with_only_whitespace_are_ignored(self, tmp_path):
        fp = tmp_path / "spaces.txt"
        fp.write_text("   \n ABC \n   \t  \n")
        result = parse_known_abbrs(fp)
        assert result == {"ABC"}


class TestLemmatizeEnglish:
    @pytest.mark.nlp
    def test_lemmatizes_english_text(self, lemmatizer):
        result = lemmatizer.lemmatize(
            "The foxes were jumping over the lazy dogs", Lang("en")
        )
        assert result == ["the", "fox", "be", "jump", "over", "the", "lazy", "dog"]

    @pytest.mark.nlp
    def test_preserves_known_abbreviation(self, lemmatizer_with_abbrs):
        result = lemmatizer_with_abbrs.lemmatize(
            "NASA launched a rocket yesterday", Lang("en")
        )
        assert result is not None
        assert result == ["NASA", "launch", "a", "rocket", "yesterday"]

    @pytest.mark.nlp
    def test_replaces_hyphens_with_spaces(self, lemmatizer):
        result = lemmatizer.lemmatize("the flow-meter is calibrated", Lang("en"))
        assert result == ["the", "flow", "meter", "be", "calibrate"]

    @pytest.mark.nlp
    def test_replaces_en_dash_with_spaces(self, lemmatizer):
        result = lemmatizer.lemmatize("the flow\u2013meter is calibrated", Lang("en"))
        assert result == ["the", "flow", "meter", "be", "calibrate"]

    @pytest.mark.nlp
    def test_empty_string_returns_empty_list(self, lemmatizer):
        result = lemmatizer.lemmatize("", Lang("en"))
        assert result == []

    @pytest.mark.nlp
    def test_abbreviations_only_input(self, lemmatizer_with_abbrs):
        result = lemmatizer_with_abbrs.lemmatize("NASA", Lang("en"))
        assert result == ["NASA"]

    @pytest.mark.nlp
    def test_case_sensitive_abbreviation_matching(self, lemmatizer_with_abbrs):
        result = lemmatizer_with_abbrs.lemmatize("Nasa launched a rocket", Lang("en"))
        assert result[0] != "Nasa"


class TestLemmatizeRussian:
    @pytest.mark.nlp
    def test_lemmatizes_russian_text(self, lemmatizer):
        result = lemmatizer.lemmatize("бегущие лисы прыгали", Lang("ru"))
        assert result == ["бежать", "лис", "прыгать"]

    @pytest.mark.nlp
    def test_preserves_known_abbreviation(self, lemmatizer_with_abbrs):
        result = lemmatizer_with_abbrs.lemmatize(
            "ГОСТ требует проверки", Lang("ru")
        )
        assert result is not None
        assert result == ["ГОСТ", "требовать", "проверка"]

    @pytest.mark.nlp
    def test_empty_string_returns_empty_list(self, lemmatizer):
        result = lemmatizer.lemmatize("", Lang("ru"))
        assert result == []

    @pytest.mark.nlp
    def test_abbreviations_only_input(self, lemmatizer_with_abbrs):
        result = lemmatizer_with_abbrs.lemmatize("ГОСТ", Lang("ru"))
        assert result == ["ГОСТ"]

    @pytest.mark.nlp
    def test_mixed_script_russian(self, lemmatizer):
        result = lemmatizer.lemmatize("проверка HTTP запроса", Lang("ru"))
        assert len(result) == 3


class TestLemmatizeUnsupported:
    def test_unsupported_language_returns_none(self, lemmatizer):
        result = lemmatizer.lemmatize("some text", Lang("fr"))
        assert result is None


class TestGlossaryLemmatizer:
    def test_entries_updated_with_lemmatized_forms(self, fake_lemmatizer):
        term_en = Term(Lang("en"), "Flow Meter")
        term_ru = Term(Lang("ru"), "Расходомер")
        entry = GlossaryEntry(id=1, terms=frozenset([term_en, term_ru]))

        gl = GlossaryLemmatizer(lemmatizer=fake_lemmatizer)  # pyright: ignore[reportArgumentType]
        result = gl.lemmatize_entries([entry])

        assert len(result) == 1
        updated_entry = result[0]
        assert updated_entry.id == 1
        for term in updated_entry.terms:
            assert term.lemmatized == term.value.lower()

    def test_unsupported_language_terms_kept_with_none_lemmatized(
        self, fake_lemmatizer
    ):
        unsupported_only = FakeLemmatizer(supported=frozenset({Lang("ru")}))
        term_en = Term(Lang("en"), "Flow Meter")
        term_ru = Term(Lang("ru"), "Расходомер")
        entry = GlossaryEntry(id=2, terms=frozenset([term_en, term_ru]))

        gl = GlossaryLemmatizer(lemmatizer=unsupported_only)  # pyright: ignore[reportArgumentType]
        result = gl.lemmatize_entries([entry])

        assert len(result) == 1
        updated_entry = result[0]
        terms_list = list(updated_entry.terms)
        ru_term = next(t for t in terms_list if t.language == Lang("ru"))
        en_term = next(t for t in terms_list if t.language == Lang("en"))
        assert ru_term.lemmatized is not None
        assert en_term.lemmatized is None

    def test_empty_entries_list_returns_empty(self, fake_lemmatizer):
        gl = GlossaryLemmatizer(lemmatizer=fake_lemmatizer)  # pyright: ignore[reportArgumentType]
        result = gl.lemmatize_entries([])
        assert result == []

    def test_monolingual_entry_lemmatizes_single_term(self, fake_lemmatizer):
        term_en = Term(Lang("en"), "Flow Meter")
        entry = GlossaryEntry(id=3, terms=frozenset([term_en]))

        gl = GlossaryLemmatizer(lemmatizer=fake_lemmatizer)
        result = gl.lemmatize_entries([entry])

        assert len(result) == 1
        updated_terms = list(result[0].terms)
        assert len(updated_terms) == 1
        assert updated_terms[0].lemmatized == "flow meter"

    def test_whitespace_only_term_kept_with_none_lemmatized(self, fake_lemmatizer):
        term_en = Term(Lang("en"), "   ")
        entry = GlossaryEntry(id=4, terms=frozenset([term_en]))

        gl = GlossaryLemmatizer(lemmatizer=fake_lemmatizer)  # pyright: ignore[reportArgumentType]
        result = gl.lemmatize_entries([entry])

        assert len(result) == 1
        updated_terms = list(result[0].terms)
        assert updated_terms[0].lemmatized is None
