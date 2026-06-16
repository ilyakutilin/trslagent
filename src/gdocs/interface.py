"""
Google Docs interface for the translation agent.

Doc format
----------
The watched document must follow this template (whitespace-tolerant):

    === Translation Request ===

    Status: Draft          ← change to "Ready" to trigger processing

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

Authentication
--------------
Uses a Google service account. Share the doc with the service account email
(editor access so the script can write Status back).

Required scopes:
    https://www.googleapis.com/auth/documents
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.config import GDocsSettings, logger
from src.doc_parser import DocParser
from src.main import main
from src.models import InputData

SCOPES = ["https://www.googleapis.com/auth/documents"]
TRANSLATION_HEADING = "=== Translation ==="


class DocsService(Protocol):
    def documents(self) -> DocumentsResource: ...


class DocumentsResource(Protocol):
    def get(self, documentId: str) -> GetRequest: ...
    def batchUpdate(self, documentId: str, body: dict) -> BatchUpdateRequest: ...


class GetRequest(Protocol):
    def execute(self) -> dict: ...


class BatchUpdateRequest(Protocol):
    def execute(self) -> dict: ...


class GoogleDocHandler:
    def __init__(
        self,
        cfg: GDocsSettings,
        service: DocsService | None = None,
        main_fn: Callable[[InputData], str | None] | None = None,
        scopes: list[str] | None = None,
        translation_heading: str | None = None,
    ) -> None:
        if not cfg.document_id:
            raise ValueError("Google doc ID must be provided")

        self.cfg = cfg
        self.scopes = scopes or SCOPES
        self.service = service or self._build_service()
        self.main_fn = main_fn or main
        self.translation_heading = translation_heading or TRANSLATION_HEADING

    def _build_service(self) -> DocsService:
        creds = Credentials.from_service_account_file(
            self.cfg.credentials_path, scopes=self.scopes
        )
        return build("docs", "v1", credentials=creds)

    @staticmethod
    def _extract_plain_text(doc: dict) -> str:
        """Concatenate all text runs in the document body into a single string."""
        chunks: list[str] = []
        for element in doc.get("body", {}).get("content", []):
            paragraph = element.get("paragraph")
            if not paragraph:
                continue

            for part in paragraph.get("elements", []):
                text_run = part.get("textRun")
                if text_run:
                    chunks.append(text_run.get("content", ""))

        return "".join(chunks)

    @staticmethod
    def _find_text_range(doc: dict, search: str) -> tuple[int, int]:
        """Return the (startIndex, endIndex) of the first occurrence of *search*."""
        for element in doc.get("body", {}).get("content", []):
            paragraph = element.get("paragraph")
            if not paragraph:
                continue

            for part in paragraph.get("elements", []):
                text_run = part.get("textRun")
                if not text_run:
                    continue

                content = text_run.get("content", "")

                if search in content:
                    start = part["startIndex"] + content.index(search)
                    end = start + len(search)
                    return start, end

        return 0, 0

    def _replace_text(self, old_text: str, new_text: str) -> None:
        """Replace ALL occurrences of old_text with new_text."""
        self.service.documents().batchUpdate(
            documentId=self.cfg.document_id,
            body={
                "requests": [
                    {
                        "replaceAllText": {
                            "containsText": {"text": old_text, "matchCase": True},
                            "replaceText": new_text,
                        }
                    }
                ]
            },
        ).execute()

    def _set_status(self, current_status: str, new_status: str) -> None:
        """Overwrite the Status line in the doc."""
        try:
            self._replace_text(
                old_text=f"Status: {current_status}", new_text=f"Status: {new_status}"
            )
        except HttpError as exc:
            logger.error(f"Failed to update Status in doc: {exc}")

    def _append_translation_to_doc(self, translation: str):
        """
        Appends translation to the appropriate place in the document.

        Args:
            translation: The translated text.
        """
        try:
            document = (
                self.service.documents().get(documentId=self.cfg.document_id).execute()
            )

            end_index = document["body"]["content"][-1]["endIndex"]
            requests: list[dict] = []
            _, insert_after = self._find_text_range(
                doc=document, search=self.translation_heading
            )
            if not insert_after:
                insert_after = end_index
                translation = f"{self.translation_heading}\n\n{translation}"
            else:
                delete_request = {
                    "deleteContentRange": {
                        "range": {"startIndex": insert_after + 1, "endIndex": end_index}
                    }
                }
                requests.append(delete_request)

            insert_request = {
                "insertText": {
                    "location": {"index": insert_after + 1},
                    "text": f"\n\n{translation}",
                }
            }
            requests.append(insert_request)

            result = (
                self.service.documents()
                .batchUpdate(
                    documentId=self.cfg.document_id, body={"requests": requests}
                )
                .execute()
            )

            logger.info(
                f"Successfully appended translation to document {self.cfg.document_id}"
            )

            return result

        except HttpError as err:
            logger.error(f"An error occurred: {err}")

    def run_polling_loop(self) -> None:
        """
        Poll the Google Doc every `poll_interval_seconds` seconds.
        When Status == ready_trigger, parse the doc and call main_fn(InputData).
        Writes Status back to done_marker on success, error_marker on failure.
        """
        logger.info(
            f"Polling document {self.cfg.document_id} "
            f"every {self.cfg.poll_interval_seconds}s. "
            f"Waiting for Status: {self.cfg.ready_trigger} …"
        )

        while True:
            try:
                doc = (
                    self.service.documents()
                    .get(documentId=self.cfg.document_id)
                    .execute()
                )
                raw = self._extract_plain_text(doc)
                try:
                    input_data, status = DocParser(raw=raw).parse()
                except ValueError as exc:
                    # Doc may simply not be filled in yet; log at DEBUG level
                    logger.debug(f"Doc not ready / parse error: {exc}")
                    time.sleep(self.cfg.poll_interval_seconds)
                    continue

                if status is None:
                    raise ValueError(
                        "Status cannot be None for the Google Docs workflow"
                    )

                if status.lower() != self.cfg.ready_trigger.lower():
                    logger.debug(
                        f"Status is '{status}', not '{self.cfg.ready_trigger}'. "
                        "Skipping."
                    )
                    time.sleep(self.cfg.poll_interval_seconds)
                    continue

                logger.info(
                    f"Status is '{self.cfg.ready_trigger}' — starting translation."
                )
                self._set_status(current_status=status, new_status="Processing…")

                try:
                    translation = self.main_fn(input_data)
                    if not translation:
                        raise ValueError("Translation is empty")

                    self._append_translation_to_doc(translation=translation)

                    self._set_status(
                        current_status="Processing…",
                        new_status=self.cfg.done_marker,
                    )
                    logger.info(
                        f"Translation complete. Status set to '{self.cfg.done_marker}'."
                    )

                except Exception as exc:
                    logger.exception(f"Translation failed: {exc}")
                    self._set_status(
                        current_status="Processing…",
                        new_status=self.cfg.error_marker,
                    )

            except HttpError as exc:
                logger.error(f"Google API error: {exc}")

            time.sleep(self.cfg.poll_interval_seconds)
