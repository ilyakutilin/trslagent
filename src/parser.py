from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue

from src.config import get_settings, logger
from src.lemmatizer import Lemmatizer

settings = get_settings()


class ParserError(Exception):
    pass


@dataclass
class Term:
    language: Lang
    value: str
    lemmatized: str | None = None


@dataclass
class GlossaryEntry:
    id: str
    terms: list[Term]

    def str_synonyms(self, lang: Lang) -> str:
        return " | ".join([t.value for t in self.terms if t.language == lang])


class GlossaryXMLParser:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._entry_ids: set[str] = set()

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

    def _get_entry_id(self, concept_group: ElementTree.Element) -> str:
        concept_el = concept_group.find("concept")
        if concept_el is None:
            raise ValueError("conceptGrp has no <concept> tag, therefore no ID")
        concept_id = (concept_el.text or "").strip()
        if not concept_id:
            raise ValueError("conceptGrp <concept> tag is empty, therefore no ID")

        stem = Path(self.file_path).stem
        entry_id = f"{stem}_{concept_id}"

        previous_ids_len = len(self._entry_ids)
        self._entry_ids.add(entry_id)
        if len(self._entry_ids) == previous_ids_len:
            raise ValueError(f"entry id {entry_id} is not unique")

        return entry_id

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

    def _lemmatize_term(self, term: str, lang: Lang) -> str | None:
        return Lemmatizer(term, lang).lemmatize()

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

            entry = GlossaryEntry(id=entry_id, terms=[])

            language_groups = cg.findall("languageGrp")
            if not language_groups:
                logger.warning(
                    f"Skipping conceptGrp with seq No. {cg_idx}: no <languageGrp> tags"
                )

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
                    lemmatized_lt = self._lemmatize_term(lt, lang)
                    entry.terms.append(
                        Term(language=lang, value=lt, lemmatized=lemmatized_lt)
                    )

            entries.append(entry)

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
        root = self._get_xml_root()

        glossary_entries = self._get_glossary_entries(root)

        logger.info(
            f"Finished loading {len(glossary_entries)} entries from {self.file_path}"
        )
        return glossary_entries


class GlossaryParser:
    def __init__(self, dir_path: str | None = None) -> None:
        self.dir_path = dir_path or settings.glossary.xml_dir

    def parse(self) -> list[GlossaryEntry]:
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
        dir_path = self.dir_path
        dir_path_obj = Path(dir_path)

        # 1. Check existence and directory type
        if not dir_path_obj.exists():
            raise ParserError(f"Directory does not exist: {dir_path}")

        if not dir_path_obj.is_dir():
            raise ParserError(f"Path is not a directory: {dir_path}")

        # 2. Check that directory is not empty (contains any files or subdirs)
        if not any(dir_path_obj.iterdir()):
            raise ParserError(f"Directory is empty: {dir_path}")

        # 3. Find all XML files in the directory (non-recursive)
        xml_files = list(dir_path_obj.glob("*.xml"))
        if not xml_files:
            raise ParserError(f"No XML files found in directory: {dir_path}")

        all_entries = []

        # 4. Process each XML file
        failed_count = 0
        for xml_file in xml_files:
            file_path_str = str(xml_file)
            try:
                parser = GlossaryXMLParser(file_path_str)
                entries = parser.parse()
                all_entries.extend(entries)
            except Exception as e:
                failed_count += 1
                logger.warning(f"Error parsing {file_path_str}: {e}")
                continue

        if not all_entries:
            if failed_count == len(xml_files):
                raise ParserError("All XML files failed to parse")
            raise ParserError("No glossary entries found across all XML files")

        return all_entries
