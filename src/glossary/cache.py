import pickle
from pathlib import Path

from src.config import get_settings, logger
from src.glossary.models import CachedEntries, GlossaryEntry, GlossaryFile

settings = get_settings()


class GlossaryCache:
    def __init__(
        self,
        glossary_file: GlossaryFile,
        cache_dir: str | Path = settings.glossary.xml_dir,
    ) -> None:
        self.glossary_file = glossary_file
        self.cache_file = self._get_cache_file_path(Path(cache_dir))

    def get_cache(self) -> CachedEntries:
        """Returns all GlossaryEntry objects for a particular glossary from cache."""

        empty_cache = CachedEntries(entries=[], are_up_to_date=False)

        if not self.cache_file.exists():
            logger.info("Glossary cache file does not exist")
            return empty_cache

        try:
            with self.cache_file.open("rb") as f:
                payload = pickle.load(f)

            are_up_to_date = payload.get("hash") == self.glossary_file.hash
            cached_entries = CachedEntries(
                entries=payload["entries"], are_up_to_date=are_up_to_date
            )
            logger.info(
                f"Cache hit — loaded {len(cached_entries.entries)} entries from pickle."
            )
            if not are_up_to_date:
                logger.warning("Cache is stale. Re-parsing is required.")
            else:
                logger.info("Cache is up to date and is going to be used.")

            return cached_entries

        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Cache read failed ({exc})")

        return empty_cache

    def write_cache(self, entries: list[GlossaryEntry]) -> None:
        payload = {
            "hash": self.glossary_file.hash,
            "entries": entries,
        }
        tmp = self.cache_file.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.cache_file)  # atomic on POSIX
        logger.info(f"Cache written ({len(entries)} entries).")

    def _get_cache_file_path(self, cache_dir: Path) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)

        if not cache_dir.is_dir():
            raise NotADirectoryError(f"Cache dir is not a directory: {cache_dir}")

        return cache_dir / f"{self.glossary_file.glossary_name}.pickle"
