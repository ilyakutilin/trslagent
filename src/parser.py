from pathlib import Path
from xml.etree import ElementTree

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue

from src.config import get_settings, logger
from src.glossary_cache import GlossaryCache
from src.lemmatizer import GlossaryLemmatizer
from src.models import CachedEntries, GlossaryDiff, GlossaryEntry, GlossaryFile, Term

settings = get_settings()


class ParserError(Exception):
    pass


class GlossaryXMLParser:
    def __init__(self, file_path: str | Path) -> None:
        self.file_path = file_path

    def _get_xml_root(self) -> ElementTree.Element:
        fp = Path(self.file_path)

        logger.info(f"Loading glossary terms from XML: {fp}")

        if not fp.is_file():
            logger.error(f"XML file not found: {fp}")
            raise FileNotFoundError(f"Glossary XML file not found: {fp}")

        try:
            tree = ElementTree.parse(fp)
        except ElementTree.ParseError as exc:
            logger.error(f"Malformed XML in {fp}: {exc}")
            raise ValueError(f"Failed to parse XML file {fp}: {exc}") from exc
        except OSError as exc:
            logger.error(f"Cannot read file {fp}: {exc}")
            raise RuntimeError(f"Cannot read glossary file {fp}: {exc}") from exc

        root = tree.getroot()
        if root.tag != "mtf":
            logger.error(f"Unexpected root tag {root.tag} in {fp} (expected 'mtf')")
            raise ValueError(
                f"Invalid glossary XML: root tag is {root.tag}, expected 'mtf'."
            )
        return root

    def _get_entry_id(self, concept_group: ElementTree.Element) -> int:
        concept_el = concept_group.find("concept")
        if concept_el is None:
            raise ValueError("conceptGrp has no <concept> tag, therefore no ID")

        concept_id = (concept_el.text or "").strip()

        if not concept_id:
            raise ValueError("conceptGrp <concept> tag is empty, therefore no ID")

        try:
            id_ = int(concept_id)
        except ValueError:
            raise ValueError(f"failed to convert {concept_id} to integer")

        if id_ < 1:
            raise ValueError(f"{id_} is an invalid ID")

        return id_

    def _get_language(self, language_group: ElementTree.Element) -> Lang:
        lang_el = language_group.find("language")
        if lang_el is None:
            raise ValueError("<language> tag is missing")

        lang_code = lang_el.get("lang", "").strip().lower()
        lang_name = lang_el.get("type", "").strip().lower()
        try:
            return Lang(lang_code or lang_name)
        except InvalidLanguageValue as e:
            raise ValueError(f"failed to parse language: {e}")

    def _get_entry_terms_for_lang(
        self, language_group: ElementTree.Element
    ) -> list[str]:
        term_grp = language_group.findall("termGrp")
        if not term_grp:
            raise ValueError("<termGrp> tag is missing")

        terms: list[str] = []

        for term in term_grp:
            term_el = term.find("term")
            if term_el is None:
                continue

            term_text = (term_el.text or "").strip()
            if not term_text:
                continue

            terms.append(term_text)

        if len(terms) == 0:
            raise ValueError("all the terms in termGrp are empty")

        return terms

    def _get_glossary_entries(
        self, xml_root: ElementTree.Element
    ) -> list[GlossaryEntry]:
        fp = Path(self.file_path)

        concept_groups = xml_root.findall("conceptGrp")
        if not concept_groups:
            logger.warning(
                f"No <conceptGrp> elements found in {fp.name}; "
                f"skipping {fp.name} entirely."
            )
            return []

        entries: list[GlossaryEntry] = []
        self._entry_ids = set()

        for cg_idx, cg in enumerate(concept_groups):
            try:
                entry_id = self._get_entry_id(cg)
            except ValueError as e:
                logger.warning(f"Skipping conceptGrp with seq No. {cg_idx}: {e}")
                continue

            language_groups = cg.findall("languageGrp")
            if not language_groups:
                logger.warning(
                    f"Skipping conceptGrp with seq No. {cg_idx}: no <languageGrp> tags"
                )

            terms = set()

            for lg_idx, lg in enumerate(language_groups):
                try:
                    lang = self._get_language(lg)
                except ValueError as e:
                    logger.warning(
                        f"Skipping languageGrp with seq No. {lg_idx} "
                        f"of conceptGrp with entry ID {entry_id}: {e}"
                    )
                    continue

                try:
                    lang_terms = self._get_entry_terms_for_lang(lg)
                except ValueError as e:
                    logger.warning(
                        f"Skipping languageGrp with seq No. {lg_idx} "
                        f"of conceptGrp with entry ID {entry_id}: {e}"
                    )
                    continue

                for lt in lang_terms:
                    terms.add(Term(language=lang, value=lt))

            entries.append(GlossaryEntry(id=entry_id, terms=frozenset(terms)))

        len_concept_groups = len(concept_groups)
        len_entries = len(entries)
        logger.info(
            f"Glossary entries: Overall qty of entries in XML: {len_concept_groups}, "
            f"successfully parsed: {len_entries}, "
            f"skipped: {len_concept_groups - len_entries}"
        )

        return entries

    def parse(self) -> list[GlossaryEntry]:
        """
        Read a MultiTerm XML file into a list of Glossary Entries.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        If the XML structure is invalid or inconsistent.
            RuntimeError:      For unexpected errors during parsing.
        """
        logger.info(f"Parsing {Path(self.file_path).name}...")
        root = self._get_xml_root()

        glossary_entries = self._get_glossary_entries(root)

        if len(glossary_entries) == 0:
            raise ValueError(f"No entries have been parsed from {self.file_path}")

        logger.info(
            f"Finished loading {len(glossary_entries)} entries from {self.file_path}"
        )
        return glossary_entries


class GlossaryUpdater:
    def __init__(self, old: list[GlossaryEntry], new: list[GlossaryEntry]) -> None:
        self.old = old
        self.new = new

    def diff_glossary(self) -> GlossaryDiff:
        """
        Compare old (cached, lemmatized) entries against new (parsed, unlemmatized).
        Returns what needs to be added, updated, and deleted.
        """
        old_by_id: dict[int, GlossaryEntry] = {e.id: e for e in self.old}
        new_by_id: dict[int, GlossaryEntry] = {e.id: e for e in self.new}

        old_ids = old_by_id.keys()
        new_ids = new_by_id.keys()

        added_ids = new_ids - old_ids
        deleted_ids = old_ids - new_ids
        common_ids = old_ids & new_ids

        to_add = [new_by_id[i] for i in added_ids]
        to_delete = list(deleted_ids)
        to_update = [new_by_id[i] for i in common_ids if new_by_id[i] != old_by_id[i]]

        return GlossaryDiff(to_add=to_add, to_update=to_update, to_delete=to_delete)

    def apply_glossary_diff(self, diff: GlossaryDiff) -> list[GlossaryEntry]:
        entries_by_id: dict[int, GlossaryEntry] = {e.id: e for e in self.old}

        # Deletions
        for entry_id in diff.to_delete:
            entries_by_id.pop(entry_id, None)

        # Additions + updates
        needs_lemmatization = diff.to_add + diff.to_update
        if needs_lemmatization:
            glossary_lemmatizer = GlossaryLemmatizer(needs_lemmatization)
            lemmatized: list[GlossaryEntry] = glossary_lemmatizer.lemmatize_entries()
            for entry in lemmatized:
                entries_by_id[entry.id] = entry  # insert or replace

        return list(entries_by_id.values())

    def update(self) -> list[GlossaryEntry]:
        logger.info(
            f"Comparing {len(self.old)} old (cached, lemmatized) entries "
            f"against {len(self.new)} new (parsed, unlemmatized)"
        )
        diff: GlossaryDiff = self.diff_glossary()
        logger.debug(f"{len(diff.to_add)} glossary entries to be added")
        logger.debug(f"{len(diff.to_delete)} glossary entries to be deleted")
        logger.debug(f"{len(diff.to_update)} glossary entries to be updated")

        updated_entries = self.apply_glossary_diff(diff)
        logger.info(
            f"All {len(updated_entries)} entries have been successfully updated"
        )
        return self.apply_glossary_diff(diff)


class GlossaryParser:
    def __init__(self, dir_path: str | None = None) -> None:
        self.dir_path = dir_path or settings.glossary.xml_dir

    def parse(self) -> list[GlossaryEntry]:
        # TODO: Update docstring
        """
        Process all XML files in a directory, parse them with GlossaryXMLParser,
        and return a combined list of GlossaryEntry objects.

        Args:
            dir_path: Path to the directory containing XML files.

        Returns:
            List of GlossaryEntry objects from all successfully parsed files.

        Raises:
            ParserError, if:
                - the directory does not exist
                - the directory is not actually a directory
                - the directory is empty
                - there are no .xml files in the directory
                - parsing all xml files failed


        Behavior:
            - Checks that dir_path exists and is a directory.
            - Checks that the directory is not empty.
            - Finds all files with .xml extension (case-sensitive) in the directory.
            - For each XML file, creates a GlossaryXMLParser and calls .parse().
            - If .parse() raises any exception, logs a warning and skips the file.
            - Collects the list of GlossaryEntry objects from each file.
            - Returns the concatenated list.
        """
        xml_files = self._get_xml_files_in_dir()
        logger.info(
            f"The following glossary XML files found in {self.dir_path}:\n"
            f"{'\n'.join([xmlf.name for xmlf in xml_files])}\n"
        )

        removed_cache_files = self._remove_non_matching_cache_files(xml_files)
        if removed_cache_files:
            logger.warning(
                f"Non-matching pickle cache files removed from {self.dir_path}:\n"
                f"{'\n'.join([cf.name for cf in removed_cache_files])}"
            )

        all_entries = []

        failed_count = 0
        for xml_file in xml_files:
            logger.info(f"Processing {xml_file.name}...")
            glossary_file = GlossaryFile(xml_file)
            cached_entries: CachedEntries = self._get_cached_entries(glossary_file)

            if cached_entries.are_up_to_date:
                all_entries.extend(cached_entries.entries)
                continue

            try:
                parsed_entries = self._parse_xml_file(xml_file)
            except Exception as e:
                failed_count += 1
                logger.warning(f"Error parsing {xml_file}: {e}")
                continue

            updated_entries = self._get_updated_entries(
                old=cached_entries.entries, new=parsed_entries
            )

            all_entries.extend(updated_entries)

            self._update_cache(glossary_file, updated_entries)

        if not all_entries:
            if failed_count == len(xml_files):
                raise ParserError("All XML files failed to parse")
            raise ParserError("No glossary entries found across all XML files")

        return all_entries

    def _get_xml_files_in_dir(self) -> list[Path]:
        dir_path = Path(self.dir_path)

        if not dir_path.exists():
            raise ParserError(f"Directory does not exist: {dir_path}")

        if not dir_path.is_dir():
            raise ParserError(f"Path is not a directory: {dir_path}")

        if not any(dir_path.iterdir()):
            raise ParserError(f"Directory is empty: {dir_path}")

        xml_files = list(dir_path.glob("*.xml"))
        if not xml_files:
            raise ParserError(f"No XML files found in directory: {dir_path}")

        return xml_files

    def _remove_non_matching_cache_files(self, xml_files: list[Path]) -> list[Path]:
        removed: list[Path] = []
        for file in Path(self.dir_path).glob("*.pickle"):
            if file.stem not in [xmlf.stem for xmlf in xml_files]:
                file.unlink()
                removed.append(file)

        return removed

    def _get_cached_entries(self, glossary_file: GlossaryFile) -> CachedEntries:
        return GlossaryCache(glossary_file).get_cache()

    def _parse_xml_file(self, xml_file: str | Path) -> list[GlossaryEntry]:
        parser = GlossaryXMLParser(xml_file)
        return parser.parse()

    def _get_updated_entries(
        self, old: list[GlossaryEntry], new: list[GlossaryEntry]
    ) -> list[GlossaryEntry]:
        updater = GlossaryUpdater(old, new)
        return updater.update()

    def _update_cache(
        self, glossary_file: GlossaryFile, entries: list[GlossaryEntry]
    ) -> None:
        GlossaryCache(glossary_file).write_cache(entries)
