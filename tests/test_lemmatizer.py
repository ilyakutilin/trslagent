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


class TestLemmatizeEnglish:
    @pytest.mark.spacy
    def test_lemmatizes_english_text(self):
        lemmatizer = Lemmatizer()
        result = lemmatizer.lemmatize("The foxes were jumping over the lazy dogs", Lang("en"))
        assert result == ["the", "fox", "be", "jump", "over", "the", "lazy", "dog"]

    @pytest.mark.spacy
    def test_preserves_known_abbreviation(self):
        lemmatizer = Lemmatizer(known_abbrs={"NASA"})
        result = lemmatizer.lemmatize("NASA launched a rocket yesterday", Lang("en"))
        assert result is not None
        assert result[0] == "NASA"

    @pytest.mark.spacy
    def test_replaces_hyphens_with_spaces(self):
        lemmatizer = Lemmatizer()
        result = lemmatizer.lemmatize("the flow-meter is calibrated", Lang("en"))
        assert result == ["the", "flow", "meter", "be", "calibrate"]

    @pytest.mark.spacy
    def test_replaces_en_dash_with_spaces(self):
        lemmatizer = Lemmatizer()
        result = lemmatizer.lemmatize("the flow\u2013meter is calibrated", Lang("en"))
        assert result == ["the", "flow", "meter", "be", "calibrate"]


class TestLemmatizeRussian:
    def test_lemmatizes_russian_text(self):
        lemmatizer = Lemmatizer()
        result = lemmatizer.lemmatize("бегущие лисы прыгали", Lang("ru"))
        assert result == ["бежать", "лис", "прыгать"]

    def test_preserves_known_abbreviation(self):
        lemmatizer = Lemmatizer(known_abbrs={"ГОСТ"})
        result = lemmatizer.lemmatize("ГОСТ требует проверки", Lang("ru"))
        assert result is not None
        assert result[0] == "ГОСТ"


class TestLemmatizeUnsupported:
    def test_unsupported_language_returns_none(self):
        lemmatizer = Lemmatizer()
        result = lemmatizer.lemmatize("some text", Lang("fr"))
        assert result is None


class TestGlossaryLemmatizer:
    def test_entries_updated_with_lemmatized_forms(self):
        fake = FakeLemmatizer()
        term_en = Term(Lang("en"), "Flow Meter")
        term_ru = Term(Lang("ru"), "Расходомер")
        entry = GlossaryEntry(id=1, terms=frozenset([term_en, term_ru]))

        gl = GlossaryLemmatizer(lemmatizer=fake)  # pyright: ignore[reportArgumentType]
        result = gl.lemmatize_entries([entry])

        assert len(result) == 1
        updated_entry = result[0]
        assert updated_entry.id == 1
        for term in updated_entry.terms:
            assert term.lemmatized == term.value.lower()

    def test_unsupported_language_terms_kept_with_none_lemmatized(self):
        fake = FakeLemmatizer(supported=frozenset({Lang("ru")}))
        term_en = Term(Lang("en"), "Flow Meter")
        term_ru = Term(Lang("ru"), "Расходомер")
        entry = GlossaryEntry(id=2, terms=frozenset([term_en, term_ru]))

        gl = GlossaryLemmatizer(lemmatizer=fake)  # pyright: ignore[reportArgumentType]
        result = gl.lemmatize_entries([entry])

        assert len(result) == 1
        updated_entry = result[0]
        terms_list = list(updated_entry.terms)
        ru_term = next(t for t in terms_list if t.language == Lang("ru"))
        en_term = next(t for t in terms_list if t.language == Lang("en"))
        assert ru_term.lemmatized is not None
        assert en_term.lemmatized is None
