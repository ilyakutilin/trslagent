"""
Parser of the document format.

Doc format
----------
The watched document must follow this template (whitespace-tolerant):

    === Translation Request ===

    Status: Draft        ← Only for Google Docs; change to "Ready" to trigger processing

    --- Required ---
    Source Language: en
    Target Language: de
    Text:
    <<<
    Multi-line text here.
    >>>

    --- Optional ---
    Model:
    Specialized In:
    Document Type:
    Document Title:
    Use Main Glossary: true
    Print Prompt Only: false

    --- Glossary ---
    term1 = translation1
    term2 = translation2
"""

from __future__ import annotations

import re

from src.models import InputData


class DocParser:
    def __init__(self, raw: str) -> None:
        self.raw = raw

    def _parse_key_value(self, line: str) -> tuple[str, str] | None:
        """Return (normalised_key, value) from 'Key: value', or None."""
        if ":" not in line:
            return None

        key, _, value = line.partition(":")

        return key.strip().lower().replace(" ", "_"), value.strip()

    def _parse_doc_text(self) -> dict:
        """
        Parse the plain-text document into a dict ready for InputData construction.
        Raises ValueError with a descriptive message on structural problems.
        """
        # ── multi-line text block ────────────────────────────────────────────────
        text_match = re.search(r"<<<\n(.*?)>>>", self.raw, re.DOTALL)

        if not text_match:
            raise ValueError(
                "Could not find the text block. "
                "Make sure it is wrapped in <<< and >>> on their own lines."
            )

        translation_text = text_match.group(1).strip()

        if not translation_text:
            raise ValueError("The text block (between <<< and >>>) is empty.")

        # ── glossary section ─────────────────────────────────────────────────────
        glossary_lines: list[str] | None = None
        glossary_match = re.search(
            r"--- Glossary ---\n(.*?)(?:\n---|$)", self.raw, re.DOTALL
        )

        if glossary_match:
            raw_glossary = glossary_match.group(1).strip()
            if raw_glossary:
                glossary_lines = [
                    ln.strip() for ln in raw_glossary.splitlines() if ln.strip()
                ]

        # ── key/value fields (everything outside the text block & glossary) ──────
        # Strip the text block so its contents don't confuse the kv parser
        kv_section = re.sub(r"<<<.*?>>>", "", self.raw, flags=re.DOTALL)

        # Strip the glossary section too
        kv_section = re.sub(r"--- Glossary ---.*", "", kv_section, flags=re.DOTALL)

        kv: dict[str, str] = {}
        for line in kv_section.splitlines():
            parsed = self._parse_key_value(line)
            if parsed:
                key, value = parsed
                kv[key] = value

        # ── required fields ───────────────────────────────────────────────────────
        for required in ("source_language", "target_language", "status"):
            if not kv.get(required):
                raise ValueError(f"Required field '{required}' is missing or empty.")

        def _bool(val: str, field: str) -> bool:
            if val.lower() in ("true", "yes", "1"):
                return True

            if val.lower() in ("false", "no", "0", ""):
                return False

            raise ValueError(f"Field '{field}' must be true/false, got: '{val}'")

        def _optional(val: str) -> str | None:
            return val if val else None

        return dict(
            status=kv["status"],
            text=translation_text,
            source_lang=kv["source_language"],
            target_lang=kv["target_language"],
            project_glossary_lines=glossary_lines,
            model=_optional(kv.get("model", "")),
            specialized_in=_optional(kv.get("specialized_in", "")),
            doc_type=_optional(kv.get("document_type", "")),
            doc_title=_optional(kv.get("document_title", "")),
            use_main_glossary=_bool(
                kv.get("use_main_glossary", "true"), "use_main_glossary"
            ),
            print_prompt_only=_bool(
                kv.get("print_prompt_only", "false"), "print_prompt_only"
            ),
        )

    def parse(self) -> InputData:
        parsed = self._parse_doc_text()
        parsed.pop("status")  # not part of InputData
        return InputData(**parsed)


if __name__ == "__main__":
    RAW = """
=== Translation Request ===


Status: Ready


--- Required ---
Source Language: en
Target Language: ru
Text:
<<<
Text for translation line one.
Text for translation line two.
Text for translation line three.
>>>


--- Optional ---
Model: google/gemini-3.5-flash
Specialized In: technical translation
Document Type: letter
Document Title: very important letter
Use Main Glossary: true
Print Prompt Only: false


--- Glossary ---
contract = договор | контракт
letter = письмо
"""

    parser = DocParser(RAW)
    input_data = parser.parse()
    print(input_data)
