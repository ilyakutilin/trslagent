import re
from pathlib import Path
from xml.etree import ElementTree

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue

from src.config import get_settings, logger
from src.glossary.cache import GlossaryCache
from src.glossary.models import (
    CachedEntries,
    GlossaryDiff,
    GlossaryEntry,
    GlossaryFile,
    Term,
)
from src.lemmatizer import GlossaryLemmatizer, Lemmatizer

settings = get_settings()


class ParserError(Exception):
    pass


class GlossaryXMLParser:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def _get_xml_root(self) -> ElementTree.Element:
        fp = self.file_path

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
        fp = self.file_path

        concept_groups = xml_root.findall("conceptGrp")
        if not concept_groups:
            logger.warning(
                f"No <conceptGrp> elements found in {fp.name}; "
                f"skipping {fp.name} entirely."
            )
            return []

        entries: list[GlossaryEntry] = []

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
        logger.info(f"Parsing {self.file_path.name}...")
        root = self._get_xml_root()

        glossary_entries = self._get_glossary_entries(root)

        if len(glossary_entries) == 0:
            raise ValueError(f"No entries have been parsed from {self.file_path}")

        logger.info(
            f"Finished loading {len(glossary_entries)} entries from {self.file_path}"
        )
        return glossary_entries


class GlossaryUpdater:
    def __init__(
        self,
        new: list[GlossaryEntry],
        old: list[GlossaryEntry] = [],
        glossary_lemmatizer: GlossaryLemmatizer | None = None,
    ) -> None:
        self.old = old
        self.new = new
        self.glossary_lemmatizer = (
            glossary_lemmatizer if glossary_lemmatizer else GlossaryLemmatizer()
        )

    def _diff_glossary(self) -> GlossaryDiff:
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

    def _lemmatize_entries(self, entries: list[GlossaryEntry]) -> list[GlossaryEntry]:
        logger.info(f"Launching lemmatization for {len(entries)} entries...")
        lemmatized_entries: list[GlossaryEntry] = (
            self.glossary_lemmatizer.lemmatize_entries(entries)
        )
        actually_lemmatized = [
            e
            for e in lemmatized_entries
            if any([t.lemmatized is not None for t in e.terms])
        ]
        logger.info(
            f"{len(actually_lemmatized)} entries have been lemmatized out of "
            f"{len(entries)}"
        )
        return lemmatized_entries

    def _apply_glossary_diff(self, diff: GlossaryDiff) -> list[GlossaryEntry]:
        entries_by_id: dict[int, GlossaryEntry] = {e.id: e for e in self.old}

        # Deletions
        for entry_id in diff.to_delete:
            entries_by_id.pop(entry_id, None)

        # Additions + updates
        needs_lemmatization = diff.to_add + diff.to_update
        if needs_lemmatization:
            lemmatized_entries = self._lemmatize_entries(needs_lemmatization)
            for entry in lemmatized_entries:
                entries_by_id[entry.id] = entry  # insert or replace
        else:
            logger.info("Nothing to lemmatize")

        return list(entries_by_id.values())

    def update(self) -> list[GlossaryEntry]:
        if self.old == self.new:
            logger.warning("")

        if not self.old:
            return self._lemmatize_entries(self.new)

        diff: GlossaryDiff = self._diff_glossary()
        logger.info(f"{len(diff.to_add)} glossary entries to be added")
        logger.info(f"{len(diff.to_delete)} glossary entries to be deleted")
        logger.info(f"{len(diff.to_update)} glossary entries to be updated")

        updated_entries = self._apply_glossary_diff(diff)
        return updated_entries


class ProjectGlossaryParser:
    def __init__(
        self,
        project_glossary_file_path: Path,
        source_lang: Lang,
        target_lang: Lang,
        lemmatizer: Lemmatizer | None = None,
    ) -> None:
        self.file = project_glossary_file_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.lemmatizer = lemmatizer if lemmatizer else Lemmatizer()

    def parse(self) -> list[GlossaryEntry]:
        logger.info(f"Parsing the project-specific glossary from {self.file}...")
        project_glossary: list[GlossaryEntry] = []
        try:
            with open(self.file, "r", encoding="utf-8") as f:
                next_id = 1
                for line in f:
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue

                    # Parse "term = translation"
                    parts = [part.strip() for part in line.split("=", 1)]
                    if len(parts) != 2 or not all(parts):
                        logger.warning(
                            "Failed to parse malformed line "
                            f"in the project glossary file: {line}"
                        )
                        continue

                    # Parse separate terms accounting for synonyms
                    terms: list[Term] = []
                    for i, part in enumerate(parts):
                        synonyms = [s.strip() for s in re.split(r"[;|]", part)]
                        for syn in synonyms:
                            lg = self.source_lang if i == 0 else self.target_lang
                            terms.append(Term(language=lg, value=syn))

                    # Construct the entry and add to the final list
                    entry = GlossaryEntry(id=next_id, terms=frozenset(terms))
                    next_id += 1
                    project_glossary.append(entry)

        except FileNotFoundError:
            logger.warning(
                f"Project glossary file not found: {self.file}. "
                "Translation will continue, but there will be no project glossary."
            )
            return []

        except Exception as e:
            logger.warning(
                f"Failed to parse the project glossary from {self.file}: "
                f"{e}. Translation will continue, but there will be no project glossary"
            )

        return GlossaryUpdater(
            new=project_glossary,
            glossary_lemmatizer=GlossaryLemmatizer(lemmatizer=self.lemmatizer),
        ).update()


class MainGlossaryParser:
    def __init__(
        self, dir_path: str | Path, lemmatizer: Lemmatizer | None = None
    ) -> None:
        self.dir_path = dir_path
        self.lemmatizer = lemmatizer if lemmatizer else Lemmatizer()

    def parse(self) -> list[GlossaryEntry]:
        logger.info("Parsing the main glossary...")
        xml_files = self._get_xml_files_in_dir()
        if not xml_files:
            logger.warning(
                f"No XML files have been found in {self.dir_path}. "
                "Translation will continue, but there will be no main glossary."
            )
            return []

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
            glossary_file = GlossaryFile(xml_file)
            logger.info(f"Processing {xml_file.name}. Looking for cache...")
            cached_entries: CachedEntries = self._get_cached_entries(glossary_file)

            if cached_entries.are_up_to_date:
                logger.info(
                    f"Cache for {xml_file.name} is up to date: "
                    f"{len(cached_entries.entries)} valid entries"
                )
                all_entries.extend(cached_entries.entries)
                continue

            cache_missing = bool(cached_entries.entries)
            logger.info(
                f"Cache for {xml_file.name} is "
                f"{'missing' if cache_missing else 'outdated'}. Parsing the XML file..."
            )

            try:
                parsed_entries = self._parse_xml_file(xml_file)
                logger.info(
                    f"{len(parsed_entries)} entries parsed from {xml_file.name}"
                )
            except Exception as e:
                failed_count += 1
                logger.warning(f"Error parsing {xml_file}: {e}")
                continue

            logger.info(
                f"Updating the entries: comparing {len(cached_entries.entries)} "
                f"old (cached, lemmatized) entries against {len(parsed_entries)} "
                "new (parsed, unlemmatized) entries..."
            )
            updated_entries = self._get_updated_entries(
                old=cached_entries.entries, new=parsed_entries
            )
            logger.info(f"Entries for {xml_file.name} updated.")

            all_entries.extend(updated_entries)

            cache_file = self._update_cache(glossary_file, updated_entries)
            logger.info(f"Pickle cache has been updated and saved to {cache_file}")

        if not all_entries:
            if failed_count == len(xml_files):
                raise ParserError("All XML files failed to parse")
            raise ParserError("No glossary entries found across all XML files")

        logger.info(f"There are {len(all_entries)} entries in the main glossary.")
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

    def _parse_xml_file(self, xml_file: Path) -> list[GlossaryEntry]:
        parser = GlossaryXMLParser(xml_file)
        return parser.parse()

    def _get_updated_entries(
        self, new: list[GlossaryEntry], old: list[GlossaryEntry]
    ) -> list[GlossaryEntry]:
        glossary_lemmatizer = GlossaryLemmatizer(lemmatizer=self.lemmatizer)
        updater = GlossaryUpdater(old, new, glossary_lemmatizer)
        return updater.update()

    def _update_cache(
        self, glossary_file: GlossaryFile, entries: list[GlossaryEntry]
    ) -> Path:
        gc = GlossaryCache(glossary_file)
        gc.write_cache(entries)
        return gc.cache_file
