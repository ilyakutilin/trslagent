from iso639 import Lang

from src.glossary.models import (
    CachedEntries,
    GlossaryDiff,
    GlossaryEntry,
    GlossaryFile,
    Term,
)


class TestTerm:
    def test_eq_same_language_same_value(self, en_lang):
        a = Term(en_lang, "flow meter", lemmatized="flow meter")
        b = Term(en_lang, "flow meter", lemmatized="flow m")
        assert a == b

    def test_eq_different_language(self, en_lang, ru_lang):
        a = Term(en_lang, "test")
        b = Term(ru_lang, "test")
        assert a != b

    def test_eq_only_lemmatized_differs(self, en_lang):
        a = Term(en_lang, "hello", lemmatized="hello")
        b = Term(en_lang, "hello", lemmatized="h")
        assert a == b

    def test_eq_not_a_term(self, en_lang):
        a = Term(en_lang, "test")
        assert a.__eq__("not a term") is NotImplemented

    def test_hash_consistent_with_eq(self, en_lang):
        a = Term(en_lang, "alpha", lemmatized="a")
        b = Term(en_lang, "alpha", lemmatized="b")
        s = {a, b}
        assert len(s) == 1


class TestGlossaryEntry:
    def test_eq_same_id_same_terms(self, en_lang, ru_lang):
        t1 = frozenset({Term(en_lang, "flow meter"), Term(ru_lang, "расходомер")})
        t2 = frozenset({Term(en_lang, "flow meter"), Term(ru_lang, "расходомер")})
        a = GlossaryEntry(1, t1)
        b = GlossaryEntry(1, t2)
        assert a == b

    def test_eq_different_id(self, en_lang, ru_lang):
        terms = frozenset({Term(en_lang, "flow meter"), Term(ru_lang, "расходомер")})
        a = GlossaryEntry(1, terms)
        b = GlossaryEntry(2, terms)
        assert a != b

    def test_stringify_matching_langs(self, en_lang, ru_lang):
        terms = frozenset({Term(en_lang, "flow meter"), Term(ru_lang, "расходомер")})
        entry = GlossaryEntry(1, terms)
        assert entry.stringify(en_lang, ru_lang) == "flow meter = расходомер"

    def test_stringify_synonyms(self, en_lang, ru_lang):
        terms = frozenset(
            {
                Term(en_lang, "a"),
                Term(en_lang, "b"),
                Term(ru_lang, "x"),
                Term(ru_lang, "y"),
            }
        )
        entry = GlossaryEntry(1, terms)
        result = entry.stringify(en_lang, ru_lang)
        assert result is not None
        assert "a" in result and "b" in result
        assert "x" in result and "y" in result
        assert " | " in result
        assert " = " in result

    def test_stringify_langs_dont_match(self, en_lang, ru_lang):
        terms = frozenset({Term(en_lang, "flow meter"), Term(ru_lang, "расходомер")})
        entry = GlossaryEntry(1, terms)
        fr = Lang("fr")
        assert entry.stringify(en_lang, fr) is None


class TestGlossaryFile:
    def test_hash_stability(self, tmp_path):
        f = tmp_path / "gloss.xml"
        f.write_text("<mtf></mtf>")
        a = GlossaryFile(str(f))
        b = GlossaryFile(str(f))
        assert a.hash == b.hash

    def test_glossary_name_returns_stem(self, tmp_path):
        f = tmp_path / "myglossary.xml"
        f.write_text("<mtf></mtf>")
        gf = GlossaryFile(str(f))
        assert gf.glossary_name == "myglossary"


class TestGlossaryDiff:
    def test_construction(self, en_lang, ru_lang):
        entry = GlossaryEntry(
            1,
            frozenset({Term(en_lang, "flow meter"), Term(ru_lang, "расходомер")}),
        )
        diff = GlossaryDiff(to_add=[entry], to_update=[], to_delete=[1])
        assert diff.to_add == [entry]
        assert diff.to_update == []
        assert diff.to_delete == [1]

    def test_defaults_empty(self):
        diff = GlossaryDiff(to_add=[], to_update=[], to_delete=[])
        assert diff.to_add == []
        assert diff.to_update == []
        assert diff.to_delete == []


class TestCachedEntries:
    def test_construction_up_to_date(self, en_lang, ru_lang):
        entry = GlossaryEntry(
            1,
            frozenset({Term(en_lang, "flow meter"), Term(ru_lang, "расходомер")}),
        )
        cached = CachedEntries([entry], True)
        assert cached.entries == [entry]
        assert cached.are_up_to_date is True

    def test_construction_not_up_to_date(self):
        cached = CachedEntries([], False)
        assert cached.entries == []
        assert cached.are_up_to_date is False
