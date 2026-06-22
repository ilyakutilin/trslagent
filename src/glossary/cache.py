"""Cache module for reading and writing serialized glossary entries via pickle."""

import pickle
from pathlib import Path

from src.config import logger
from src.glossary.models import CachedEntries, GlossaryEntry, GlossaryFile


class GlossaryCache:
    """Reads and writes glossary entries as pickled cache files.

    Attributes:
        glossary_file: The GlossaryFile object describing the source XML.
        cache_file: Path to the corresponding pickle cache file.
    """

    def __init__(
        self,
        glossary_file: GlossaryFile,
        cache_dir: str | Path,
    ) -> None:
        """Initialize a GlossaryCache.

        Args:
            glossary_file: The GlossaryFile object describing the source XML.
            cache_dir: Directory where pickle cache files are stored.
        """
        self.glossary_file = glossary_file
        self.cache_file = self._get_cache_file_path(Path(cache_dir))

    def get_cache(self) -> CachedEntries:
        """Returns all GlossaryEntry objects for a particular glossary from cache.

        Returns:
            A CachedEntries object. If the cache file is missing, corrupt, or
            stale, the returned object will have are_up_to_date=False.
        """

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
            logger.warning(f"Cache read failed ({exc}), removing corrupt file")
            self.cache_file.unlink(missing_ok=True)

        return empty_cache

    def write_cache(self, entries: list[GlossaryEntry]) -> None:
        """Write glossary entries to the pickle cache file.

        The cache is written atomically via a temporary file that is renamed.

        Args:
            entries: The list of GlossaryEntry objects to cache.
        """
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
        """Generate the cache file path, creating the directory if needed.

        Args:
            cache_dir: The directory where cache files are stored.

        Returns:
            The path to the pickle cache file for this glossary.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{self.glossary_file.glossary_name}.pickle"
