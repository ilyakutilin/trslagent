from iso639 import Lang

from src.glossary.dedup import deduplicate_user_auto
from src.glossary.models import GlossaryEntry, Term


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


class TestDeduplicateUserAuto:
    def test_user_overrides_matched_auto(self, en_lang: Lang):
        user_entry = _make_entry(
            10, "flow meter", "расходомер", "flow meter", "расходомер"
        )
        auto_entry = _make_entry(
            1, "flow meter", "расходомер", "flow meter", "расходомер"
        )

        user_entries, auto_entries = deduplicate_user_auto(
            [auto_entry], [user_entry], en_lang
        )
        assert len(user_entries) == 1
        assert len(auto_entries) == 0

    def test_no_overlap(self, en_lang: Lang):
        user_entry = _make_entry(
            10, "flow meter", "расходомер", "flow meter", "расходомер"
        )
        auto_entry = _make_entry(
            1, "pressure valve", "клапан давления", "pressure valve", "клапан давления"
        )

        user_entries, auto_entries = deduplicate_user_auto(
            [auto_entry], [user_entry], en_lang
        )
        assert len(user_entries) == 1
        assert len(auto_entries) == 1

    def test_empty_inputs(self, en_lang: Lang):
        user_entries, auto_entries = deduplicate_user_auto([], [], en_lang)
        assert user_entries == []
        assert auto_entries == []

    def test_empty_user_entries(self, en_lang: Lang):
        auto_entry = _make_entry(
            1, "flow meter", "расходомер", "flow meter", "расходомер"
        )

        user_entries, auto_entries = deduplicate_user_auto([auto_entry], [], en_lang)
        assert user_entries == []
        assert len(auto_entries) == 1

    def test_different_values_same_lemma(self, en_lang: Lang):
        user_entry = _make_entry(10, "color", "цвет", "color", "цвет")
        auto_entry = _make_entry(1, "colour", "цвет", "color", "цвет")

        user_entries, auto_entries = deduplicate_user_auto(
            [auto_entry], [user_entry], en_lang
        )
        assert len(user_entries) == 1
        assert len(auto_entries) == 0
