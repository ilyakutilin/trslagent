from xml.etree import ElementTree as ET

import pytest
from iso639 import Lang

from src.glossary.models import GlossaryEntry, Term
from src.glossary.parser import (
    GlossaryUpdater,
    GlossaryXMLParser,
    UserGlossaryParser,
)
from src.lemmatizer import Lemmatizer


def _make_entry(
    entry_id: int, en_terms: list[str], ru_terms: list[str]
) -> GlossaryEntry:
    terms: set[Term] = set()
    for t in en_terms:
        terms.add(Term(Lang("en"), t))
    for t in ru_terms:
        terms.add(Term(Lang("ru"), t))
    return GlossaryEntry(entry_id, frozenset(terms))


class TestGlossaryXMLParser:
    def test_get_xml_root_valid(self, tmp_path):
        f = tmp_path / "gloss.xml"
        f.write_text("<mtf><conceptGrp/></mtf>")
        parser = GlossaryXMLParser(f)
        root = parser._get_xml_root()
        assert root.tag == "mtf"

    def test_get_xml_root_file_not_found(self, tmp_path):
        f = tmp_path / "nonexistent.xml"
        parser = GlossaryXMLParser(f)
        with pytest.raises(FileNotFoundError):
            parser._get_xml_root()

    def test_get_xml_root_malformed(self, tmp_path):
        f = tmp_path / "bad.xml"
        f.write_text("<mtf><unclosed>")
        parser = GlossaryXMLParser(f)
        with pytest.raises(ValueError):
            parser._get_xml_root()

    def test_get_xml_root_wrong_root_tag(self, tmp_path):
        f = tmp_path / "wrong.xml"
        f.write_text("<notmtf><conceptGrp/></notmtf>")
        parser = GlossaryXMLParser(f)
        with pytest.raises(ValueError):
            parser._get_xml_root()

    def test_get_entry_id_valid(self):
        cg = ET.fromstring("<conceptGrp><concept>42</concept></conceptGrp>")
        parser = GlossaryXMLParser()
        assert parser._get_entry_id(cg) == 42

    def test_get_entry_id_missing_concept(self):
        cg = ET.fromstring("<conceptGrp></conceptGrp>")
        parser = GlossaryXMLParser()
        with pytest.raises(ValueError, match="no <concept> tag"):
            parser._get_entry_id(cg)

    def test_get_entry_id_empty_concept(self):
        cg = ET.fromstring("<conceptGrp><concept></concept></conceptGrp>")
        parser = GlossaryXMLParser()
        with pytest.raises(ValueError, match="empty"):
            parser._get_entry_id(cg)

    def test_get_entry_id_non_numeric(self):
        cg = ET.fromstring("<conceptGrp><concept>abc</concept></conceptGrp>")
        parser = GlossaryXMLParser()
        with pytest.raises(ValueError):
            parser._get_entry_id(cg)

    def test_get_entry_id_below_one(self):
        cg = ET.fromstring("<conceptGrp><concept>0</concept></conceptGrp>")
        parser = GlossaryXMLParser()
        with pytest.raises(ValueError, match="invalid ID"):
            parser._get_entry_id(cg)

    def test_get_language_lang_attr(self):
        lg = ET.fromstring(
            '<languageGrp><language lang="en" type="English"/></languageGrp>'
        )
        parser = GlossaryXMLParser()
        result = parser._get_language(lg)
        assert result == Lang("en")

    def test_get_language_type_fallback(self):
        lg = ET.fromstring('<languageGrp><language type="English"/></languageGrp>')
        parser = GlossaryXMLParser()
        result = parser._get_language(lg)
        assert result == Lang("en")

    def test_get_language_missing(self):
        lg = ET.fromstring("<languageGrp></languageGrp>")
        parser = GlossaryXMLParser()
        with pytest.raises(ValueError, match="<language> tag is missing"):
            parser._get_language(lg)

    def test_get_language_invalid(self):
        lg = ET.fromstring('<languageGrp><language lang="zz"/></languageGrp>')
        parser = GlossaryXMLParser()
        with pytest.raises(ValueError):
            parser._get_language(lg)

    def test_get_entry_terms_single(self):
        lg = ET.fromstring(
            "<languageGrp><termGrp><term>flow meter</term></termGrp></languageGrp>"
        )
        parser = GlossaryXMLParser()
        result = parser._get_entry_terms_for_lang(lg)
        assert result == ["flow meter"]

    def test_get_entry_terms_semicolon_synonyms(self):
        lg = ET.fromstring(
            "<languageGrp><termGrp><term>a;b</term></termGrp></languageGrp>"
        )
        parser = GlossaryXMLParser()
        result = parser._get_entry_terms_for_lang(lg)
        assert result == ["a", "b"]

    def test_get_entry_terms_pipe_synonyms(self):
        lg = ET.fromstring(
            "<languageGrp><termGrp><term>x|y</term></termGrp></languageGrp>"
        )
        parser = GlossaryXMLParser()
        result = parser._get_entry_terms_for_lang(lg)
        assert result == ["x", "y"]

    def test_get_entry_terms_empty(self):
        lg = ET.fromstring(
            "<languageGrp><termGrp><term></term></termGrp></languageGrp>"
        )
        parser = GlossaryXMLParser()
        with pytest.raises(ValueError, match="all the terms in termGrp are empty"):
            parser._get_entry_terms_for_lang(lg)

    def test_parse_integration(self, tmp_path, sample_xml_glossary, en_lang, ru_lang):
        f = tmp_path / "gloss.xml"
        f.write_text(sample_xml_glossary)
        parser = GlossaryXMLParser(f)
        entries = parser.parse()
        assert len(entries) == 2

        ids = {e.id for e in entries}
        assert ids == {1, 2}

        en_terms_all = {(t.value, t.language) for e in entries for t in e.terms}
        assert en_terms_all == {
            ("flow meter", en_lang),
            ("pressure valve", en_lang),
            ("расходомер", ru_lang),
            ("клапан давления", ru_lang),
        }

        entry_1 = next(e for e in entries if e.id == 1)
        lang_1 = {t.language for t in entry_1.terms}
        assert lang_1 == {en_lang, ru_lang}

        entry_2 = next(e for e in entries if e.id == 2)
        assert len([t for t in entry_2.terms if t.language == en_lang]) == 1
        assert len([t for t in entry_2.terms if t.language == ru_lang]) == 1


class TestGlossaryUpdater:
    def test_diff_glossary_empty_old(self):
        old: list[GlossaryEntry] = []
        new = [
            _make_entry(1, ["flow meter"], ["расходомер"]),
            _make_entry(2, ["pressure valve"], ["клапан давления"]),
        ]
        updater = GlossaryUpdater(new=new, old=old)
        diff = updater._diff_glossary()
        assert diff.to_add == new
        assert diff.to_update == []
        assert diff.to_delete == []

    def test_diff_glossary_identical(self):
        entries = [_make_entry(1, ["flow meter"], ["расходомер"])]
        updater = GlossaryUpdater(new=entries, old=entries)
        diff = updater._diff_glossary()
        assert diff.to_add == []
        assert diff.to_update == []
        assert diff.to_delete == []

    def test_diff_glossary_entry_added(self):
        old = [_make_entry(1, ["flow meter"], ["расходомер"])]
        new_entry = _make_entry(2, ["pressure valve"], ["клапан давления"])
        new = [_make_entry(1, ["flow meter"], ["расходомер"]), new_entry]
        updater = GlossaryUpdater(new=new, old=old)
        diff = updater._diff_glossary()
        assert diff.to_add == [new_entry]
        assert diff.to_update == []
        assert diff.to_delete == []

    def test_diff_glossary_entry_removed(self):
        old = [
            _make_entry(1, ["flow meter"], ["расходомер"]),
            _make_entry(2, ["pressure valve"], ["клапан давления"]),
        ]
        new = [_make_entry(1, ["flow meter"], ["расходомер"])]
        updater = GlossaryUpdater(new=new, old=old)
        diff = updater._diff_glossary()
        assert diff.to_add == []
        assert diff.to_update == []
        assert diff.to_delete == [2]

    def test_diff_glossary_entry_updated(self):
        old = [_make_entry(1, ["flow meter"], ["расходомер"])]
        new = [_make_entry(1, ["flow gauge"], ["расходомер"])]
        expected_updated = [_make_entry(1, ["flow gauge"], ["расходомер"])]
        updater = GlossaryUpdater(new=new, old=old)
        diff = updater._diff_glossary()
        assert diff.to_add == []
        assert diff.to_update == expected_updated
        assert diff.to_delete == []

    def test_update_no_old_entries(self, mocker, en_lang, ru_lang):
        new = [
            _make_entry(1, ["flow meter"], ["расходомер"]),
            _make_entry(2, ["pressure valve"], ["клапан давления"]),
        ]

        def fake_lemmatize(text, lang):
            return [f"lemmatized_{text}"]

        mocker.patch.object(Lemmatizer, "lemmatize", side_effect=fake_lemmatize)
        updater = GlossaryUpdater(new=new, old=[])
        result = updater.update()
        assert len(result) == 2

        result_ids = {e.id for e in result}
        assert result_ids == {1, 2}

        for entry in result:
            for term in entry.terms:
                assert term.lemmatized == f"lemmatized_{term.value}"


class TestUserGlossaryParser:
    def test_parse_simple(self, mocker, en_lang, ru_lang):
        mocker.patch.object(Lemmatizer, "lemmatize", return_value=["dummy"])
        parser = UserGlossaryParser(["flow meter = расходомер"], en_lang, ru_lang)
        entries = parser.parse()
        assert len(entries) == 1
        entry = entries[0]
        en_terms = [t.value for t in entry.terms if t.language == en_lang]
        ru_terms = [t.value for t in entry.terms if t.language == ru_lang]
        assert en_terms == ["flow meter"]
        assert ru_terms == ["расходомер"]

    def test_parse_synonyms(self, mocker, en_lang, ru_lang):
        mocker.patch.object(Lemmatizer, "lemmatize", return_value=["dummy"])
        parser = UserGlossaryParser(["a;b = x|y"], en_lang, ru_lang)
        entries = parser.parse()
        assert len(entries) == 1
        entry = entries[0]
        en_terms = {t.value for t in entry.terms if t.language == en_lang}
        ru_terms = {t.value for t in entry.terms if t.language == ru_lang}
        assert en_terms == {"a", "b"}
        assert ru_terms == {"x", "y"}

    def test_parse_comment_lines_skipped(self, mocker, en_lang, ru_lang):
        mocker.patch.object(Lemmatizer, "lemmatize", return_value=["dummy"])
        parser = UserGlossaryParser(
            [
                "# this is a comment",
                "flow meter = расходомер",
            ],
            en_lang,
            ru_lang,
        )
        entries = parser.parse()
        assert len(entries) == 1

    def test_parse_empty_lines_skipped(self, mocker, en_lang, ru_lang):
        mocker.patch.object(Lemmatizer, "lemmatize", return_value=["dummy"])
        parser = UserGlossaryParser(
            [
                "",
                "   ",
                "flow meter = расходомер",
            ],
            en_lang,
            ru_lang,
        )
        entries = parser.parse()
        assert len(entries) == 1

    def test_parse_malformed_line_skipped(self, mocker, en_lang, ru_lang):
        mocker.patch.object(Lemmatizer, "lemmatize", return_value=["dummy"])
        parser = UserGlossaryParser(
            [
                "flow meter = расходомер",
                "no equals sign",
                "pressure valve = клапан давления",
            ],
            en_lang,
            ru_lang,
        )
        entries = parser.parse()
        assert len(entries) == 2

    def test_parse_empty_input(self, en_lang, ru_lang):
        parser = UserGlossaryParser([], en_lang, ru_lang)
        entries = parser.parse()
        assert entries == []

    def test_parse_all_comments(self, mocker, en_lang, ru_lang):
        mocker.patch.object(Lemmatizer, "lemmatize", return_value=["dummy"])
        parser = UserGlossaryParser(
            [
                "# comment 1",
                "  # indented comment",
                "",
                "# another comment",
            ],
            en_lang,
            ru_lang,
        )
        entries = parser.parse()
        assert entries == []
