from iso639 import Lang

from src.glossary.models import GlossaryEntry, Term
from src.glossary.stringify import stringify_entries


def _make_term(lang_str: str, value: str, lemmatized: str | None = None) -> Term:
    return Term(language=Lang(lang_str), value=value, lemmatized=lemmatized)


def _make_entry(
    entry_id: int,
    en_value: str,
    ru_value: str,
    en_lemma: str | None = None,
    ru_lemma: str | None = None,
) -> GlossaryEntry:
    return GlossaryEntry(
        id=entry_id,
        terms=frozenset(
            [
                _make_term("en", en_value, en_lemma),
                _make_term("ru", ru_value, ru_lemma),
            ]
        ),
    )


class TestStringifyEntries:
    def test_matching_langs(self, en_lang: Lang, ru_lang: Lang):
        entries = [
            _make_entry(1, "flow meter", "расходомер", "flow meter", "расходомер"),
            _make_entry(
                2,
                "pressure valve",
                "клапан давления",
                "pressure valve",
                "клапан давления",
            ),
        ]
        result = stringify_entries(entries, en_lang, ru_lang)
        assert "flow meter = расходомер" in result
        assert "pressure valve = клапан давления" in result
        assert "\n" in result

    def test_mismatched_langs_skipped(self, en_lang: Lang, ru_lang: Lang):
        entry = GlossaryEntry(
            id=1,
            terms=frozenset(
                [
                    _make_term("en", "hello"),
                    _make_term("fr", "bonjour"),
                ]
            ),
        )
        result = stringify_entries([entry], en_lang, ru_lang)
        assert result == ""

    def test_empty_list(self, en_lang: Lang, ru_lang: Lang):
        assert stringify_entries([], en_lang, ru_lang) == ""

    def test_multi_term_synonyms(self, en_lang: Lang, ru_lang: Lang):
        entry = GlossaryEntry(
            id=1,
            terms=frozenset(
                [
                    _make_term("en", "pressure valve"),
                    _make_term("en", "relief valve"),
                    _make_term("ru", "клапан давления"),
                    _make_term("ru", "предохранительный клапан"),
                ]
            ),
        )
        result = stringify_entries([entry], en_lang, ru_lang)
        assert "pressure valve" in result
        assert "relief valve" in result
        assert "клапан давления" in result
        assert "предохранительный клапан" in result
        assert " = " in result
        left, right = result.split(" = ", 1)
        assert all(t in left.split(" | ") for t in ("pressure valve", "relief valve"))
        assert all(
            t in right.split(" | ")
            for t in ("клапан давления", "предохранительный клапан")
        )
