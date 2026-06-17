import pickle
from pathlib import Path

import pytest
from iso639 import Lang

from src.glossary.cache import GlossaryCache
from src.glossary.models import CachedEntries, GlossaryEntry, GlossaryFile, Term


def _make_entry(entry_id: int, en_val: str, ru_val: str) -> GlossaryEntry:
    return GlossaryEntry(
        entry_id,
        frozenset({Term(Lang("en"), en_val), Term(Lang("ru"), ru_val)}),
    )


def _make_xml_file(tmp_path: Path, name: str, content: str = "<mtf></mtf>") -> Path:
    f = tmp_path / name
    f.write_text(content)
    return f


class TestGlossaryCache:
    def test_get_cache_missing_file(self, tmp_path):
        xml = _make_xml_file(tmp_path, "test.xml")
        gf = GlossaryFile(str(xml))
        cache = GlossaryCache(gf, tmp_path / "cache")
        result = cache.get_cache()
        assert result.are_up_to_date is False
        assert result.entries == []

    def test_get_cache_up_to_date(self, tmp_path):
        xml = _make_xml_file(tmp_path, "test.xml")
        gf = GlossaryFile(str(xml))
        entry = _make_entry(1, "flow meter", "расходомер")

        cache_dir = tmp_path / "cache"
        cache = GlossaryCache(gf, cache_dir)

        cache_file = cache_dir / f"{gf.glossary_name}.pickle"
        payload = {"hash": gf.hash, "entries": [entry]}
        with cache_file.open("wb") as f:
            pickle.dump(payload, f)

        result = cache.get_cache()
        assert result.are_up_to_date is True
        assert result.entries == [entry]

    def test_get_cache_stale(self, tmp_path):
        xml = _make_xml_file(tmp_path, "test.xml")
        gf = GlossaryFile(str(xml))
        entry = _make_entry(1, "flow meter", "расходомер")

        cache_dir = tmp_path / "cache"
        cache = GlossaryCache(gf, cache_dir)

        cache_file = cache_dir / f"{gf.glossary_name}.pickle"
        payload = {"hash": "wronghash", "entries": [entry]}
        with cache_file.open("wb") as f:
            pickle.dump(payload, f)

        result = cache.get_cache()
        assert result.are_up_to_date is False
        assert result.entries == [entry]

    def test_get_cache_corrupt_pickle(self, tmp_path):
        xml = _make_xml_file(tmp_path, "test.xml")
        gf = GlossaryFile(str(xml))

        cache_dir = tmp_path / "cache"
        cache = GlossaryCache(gf, cache_dir)

        cache_file = cache_dir / f"{gf.glossary_name}.pickle"
        cache_file.write_bytes(b"not a valid pickle")

        result = cache.get_cache()
        assert result.are_up_to_date is False
        assert result.entries == []

    def test_write_cache_then_read_back(self, tmp_path):
        xml = _make_xml_file(tmp_path, "test.xml")
        gf = GlossaryFile(str(xml))
        entry = _make_entry(1, "flow meter", "расходомер")

        cache_dir = tmp_path / "cache"
        cache = GlossaryCache(gf, cache_dir)
        cache.write_cache([entry])

        result = cache.get_cache()
        assert result.are_up_to_date is True
        assert result.entries == [entry]

    def test_write_cache_overwrites_existing(self, tmp_path):
        xml = _make_xml_file(tmp_path, "test.xml")
        gf = GlossaryFile(str(xml))
        entry1 = _make_entry(1, "flow meter", "расходомер")
        entry2 = _make_entry(2, "pressure valve", "клапан давления")

        cache_dir = tmp_path / "cache"
        cache = GlossaryCache(gf, cache_dir)
        cache.write_cache([entry1])
        cache.write_cache([entry2])

        result = cache.get_cache()
        assert result.are_up_to_date is True
        assert len(result.entries) == 1
        assert result.entries == [entry2]

    def test_get_cache_file_path_creates_directory(self, tmp_path):
        cache_dir = tmp_path / "nonexistent" / "cache"
        xml = _make_xml_file(tmp_path, "test.xml")
        gf = GlossaryFile(str(xml))

        cache = GlossaryCache(gf, cache_dir)
        assert cache_dir.is_dir()
        assert cache.cache_file == cache_dir / f"{gf.glossary_name}.pickle"

    def test_get_cache_file_path_raises_not_a_directory(self, tmp_path):
        cache_dir = tmp_path / "notadir"
        cache_dir.write_text("i am a file")
        xml = _make_xml_file(tmp_path, "test.xml")
        gf = GlossaryFile(str(xml))

        with pytest.raises(FileExistsError):
            GlossaryCache(gf, cache_dir)
