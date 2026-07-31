"""Email processing helpers for the Resend inbound email pipeline.

Handles fetching email content, downloading attachments, sending replies,
and constructing Settings objects from email attachments and body text.
"""

import tempfile
import tomllib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from loguru import logger

from src.config import ProxySettings, Settings, get_settings
from src.http_client import create_client

MAX_INDIVIDUAL_ATTACHMENT = 10 * 1024 * 1024  # 10 MB
RESEND_API_BASE = "https://api.resend.com"
RESEND_EMAILS_RECEIVING_BASE = f"{RESEND_API_BASE}/emails/receiving"


@asynccontextmanager
async def _resolve_client(
    client: httpx.AsyncClient | None,
    *,
    timeout: float,
    proxy_settings: ProxySettings | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    """Yield an injected client or a short-lived owned one.

    When *client* is None, a client is created via ``create_client`` with
    the given *timeout* and *proxy_settings* and closed when the context
    exits. An injected client is yielded as-is and is never closed by this
    helper.

    Args:
        client: Optional pre-configured client injected by the caller.
        timeout: Timeout in seconds used for the owned client.
        proxy_settings: Proxy settings for the owned short-lived client;
            an injected client is unaffected.

    Yields:
        The httpx.AsyncClient to use for requests.
    """
    if client is None:
        async with create_client(
            timeout=timeout, proxy_settings=proxy_settings
        ) as owned:
            yield owned
    else:
        yield client


async def fetch_email_content(
    email_id: str,
    api_key: str,
    *,
    client: httpx.AsyncClient | None = None,
    proxy_settings: ProxySettings | None = None,
) -> dict:
    """Retrieve the raw email content from the Resend Receiving API.

    Args:
        email_id: The Resend email ID to fetch.
        api_key: Resend API key for authentication.
        client: Optional pre-configured client to reuse; it is not closed
            by this function. When None, a short-lived client is created
            via ``create_client`` and closed automatically.
        proxy_settings: Proxy settings for the owned short-lived client;
            an injected *client* is unaffected.

    Returns:
        The full email JSON payload, including text, html, and headers.
    """
    async with _resolve_client(
        client, timeout=30, proxy_settings=proxy_settings
    ) as client:
        resp = await client.get(
            f"{RESEND_EMAILS_RECEIVING_BASE}/{email_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        return resp.json()


async def fetch_attachments(
    email_id: str,
    api_key: str,
    *,
    client: httpx.AsyncClient | None = None,
    proxy_settings: ProxySettings | None = None,
) -> list[dict]:
    """List all attachments for a received email via the Resend API.

    Args:
        email_id: The Resend email ID.
        api_key: Resend API key for authentication.
        client: Optional pre-configured client to reuse; it is not closed
            by this function. When None, a short-lived client is created
            via ``create_client`` and closed automatically.
        proxy_settings: Proxy settings for the owned short-lived client;
            an injected *client* is unaffected.

    Returns:
        A list of attachment metadata dicts (filename, download_url, etc.).
        Returns an empty list if no attachments are found.
    """
    async with _resolve_client(
        client, timeout=30, proxy_settings=proxy_settings
    ) as client:
        resp = await client.get(
            f"{RESEND_EMAILS_RECEIVING_BASE}/{email_id}/attachments",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        return resp.json().get("data", [])


async def download_attachment(
    download_url: str,
    *,
    client: httpx.AsyncClient | None = None,
    proxy_settings: ProxySettings | None = None,
) -> bytes:
    """Download a single attachment from its temporary download URL.

    Args:
        download_url: The temporary URL provided by the Resend API
            for downloading the attachment content.
        client: Optional pre-configured client to reuse; it is not closed
            by this function. When None, a short-lived client is created
            via ``create_client`` and closed automatically.
        proxy_settings: Proxy settings for the owned short-lived client;
            an injected *client* is unaffected.

    Returns:
        The raw bytes of the attachment.

    The request is sent with a 60-second timeout regardless of whether an
    injected client or an owned one is used.

    Raises:
        ValueError: If the attachment exceeds ``MAX_INDIVIDUAL_ATTACHMENT``
            (10 MB).
        httpx.HTTPStatusError: If the download request fails.
    """
    async with _resolve_client(
        client, timeout=60, proxy_settings=proxy_settings
    ) as client:
        resp = await client.get(download_url, timeout=60)
        resp.raise_for_status()
        content = resp.content
        if len(content) > MAX_INDIVIDUAL_ATTACHMENT:
            raise ValueError(
                f"Attachment too large: {len(content)} bytes "
                f"(max {MAX_INDIVIDUAL_ATTACHMENT})"
            )
        return content


async def send_reply(
    *,
    to: str,
    subject: str,
    body: str,
    message_id: str,
    from_address: str,
    api_key: str,
    client: httpx.AsyncClient | None = None,
    proxy_settings: ProxySettings | None = None,
) -> None:
    """Send a threaded reply email via the Resend send API.

    Adds a ``Re:`` prefix to the subject if not already present and links
    the reply to the original message via the ``In-Reply-To`` header.

    Args:
        to: Recipient email address.
        subject: Original email subject (will be prefixed with ``Re:``).
        body: Plain-text reply body.
        message_id: The Message-ID of the original email for threading.
        from_address: Sender email address.
        api_key: Resend API key for authentication.
        client: Optional pre-configured client to reuse; it is not closed
            by this function. When None, a short-lived client is created
            via ``create_client`` and closed automatically.
        proxy_settings: Proxy settings for the owned short-lived client;
            an injected *client* is unaffected.
    """
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"

    async with _resolve_client(
        client, timeout=30, proxy_settings=proxy_settings
    ) as client:
        resp = await client.post(
            f"{RESEND_API_BASE}/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": from_address,
                "to": [to],
                "subject": subject,
                "html": (
                    '<pre style="white-space:pre-wrap;font-family:monospace">'
                    f"{body}"
                    "</pre>"
                ),
                "text": body,
                "headers": {
                    "In-Reply-To": message_id,
                },
            },
        )
        resp.raise_for_status()
        logger.info(
            "Reply sent to {} (Resend id: {})",
            to,
            resp.json().get("id", "unknown"),
        )


def _patch_toml_for_attachments(
    toml_content: str,
    source_text: str | None,
    target_text: str | None,
    glossary_lines: list[str] | None,
) -> str:
    """Remove file-path keys from TOML content and reconstruct it.

    The ``InputData`` validator would try to read disk files for
    ``source_file_path``, ``target_file_path``, and
    ``user_glossary_file_path``, so they are stripped out. The actual
    text data is supplied directly after Settings construction.

    Args:
        toml_content: Raw TOML string from the attached ``config.toml``.
        source_text: Source text content (unused; preserved for call
            signature compatibility).
        target_text: Target text content (unused; preserved for call
            signature compatibility).
        glossary_lines: Glossary lines (unused; preserved for call
            signature compatibility).

    Returns:
        A reconstructed TOML string with file-path keys removed from the
        ``[input_data]`` section.
    """
    data = tomllib.loads(toml_content)
    id_section = data.get("input_data", {})
    id_section.pop("source_file_path", None)
    id_section.pop("target_file_path", None)
    id_section.pop("user_glossary_file_path", None)
    id_section.pop("ref_source_file_path", None)
    id_section.pop("ref_target_file_path", None)
    data["input_data"] = id_section

    lines: list[str] = []
    for section, values in data.items():
        lines.append(f"[{section}]")
        for k, v in values.items():
            if isinstance(v, str):
                escaped = v.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{k} = "{escaped}"')
            elif isinstance(v, bool):
                lines.append(f"{k} = {str(v).lower()}")
            else:
                lines.append(f"{k} = {v}")
        lines.append("")
    return "\n".join(lines)


def build_settings_from_email(
    *,
    attachment_bodies: dict[str, bytes],
    email_body: str,
    default_cfg: Settings,
) -> Settings:
    """Construct a Settings object from email attachments and body.

    Loads settings from an attached ``config.toml`` (if present) or the
    server's default config. Source / target / glossary texts are supplied
    directly, bypassing file reads. If no ``source.txt`` is provided, the
    plain-text email body is used as the source text.

    If the email body starts with ``@match-glossary`` (case-insensitive)
    as the first non-empty line, that line is stripped and
    ``match_glossary_only`` is set on ``InputData``, triggering
    glossary-term matching instead of translation or review.

    Args:
        attachment_bodies: Mapping of attachment filenames to their raw
            bytes content.
        email_body: Plain-text email body (used as fallback source text).
        default_cfg: The server's default Settings, used when no
            ``config.toml`` is attached.

    Returns:
        A Settings object ready for use in the translation/review
        pipeline.
    """
    source_text: str | None = None
    target_text: str | None = None
    glossary_lines: list[str] | None = None
    ref_source_text: str | None = None
    ref_target_text: str | None = None

    MATCH_TRIGGER = "@match-glossary"
    match_glossary_only = False
    clean_body = email_body.strip()
    if clean_body:
        first_line = clean_body.split("\n", 1)[0].strip()
        if first_line.casefold() == MATCH_TRIGGER.casefold():
            match_glossary_only = True
            _, sep, rest = clean_body.partition("\n")
            clean_body = rest.strip()
            logger.info("Match-glossary mode triggered via email body prefix")

    if "source.txt" in attachment_bodies:
        source_text = attachment_bodies["source.txt"].decode("utf-8")

    if "target.txt" in attachment_bodies:
        target_text = attachment_bodies["target.txt"].decode("utf-8")

    if "glossary.txt" in attachment_bodies:
        glossary_lines = attachment_bodies["glossary.txt"].decode("utf-8").splitlines()

    if "ref_source.txt" in attachment_bodies:
        ref_source_text = attachment_bodies["ref_source.txt"].decode("utf-8")

    if "ref_target.txt" in attachment_bodies:
        ref_target_text = attachment_bodies["ref_target.txt"].decode("utf-8")

    if "config.toml" in attachment_bodies:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_toml = attachment_bodies["config.toml"].decode("utf-8")
            raw_toml = _patch_toml_for_attachments(
                raw_toml, source_text, target_text, glossary_lines
            )
            config_path = tmp / "config.toml"
            config_path.write_text(raw_toml, encoding="utf-8")
            settings = get_settings(toml_path=config_path)

        if source_text is not None:
            settings.input_data.source_text = source_text
        elif clean_body:
            settings.input_data.source_text = clean_body

        if target_text is not None:
            settings.input_data.target_text = target_text

        if glossary_lines is not None:
            settings.input_data.user_glossary_lines = glossary_lines

        if ref_source_text is not None:
            settings.input_data.ref_source_text = ref_source_text

        if ref_target_text is not None:
            settings.input_data.ref_target_text = ref_target_text

        settings.input_data.match_glossary_only = match_glossary_only

        return settings

    settings = default_cfg.model_copy(deep=True)
    settings.output_data.print_prompt_only = False
    settings.input_data.source_text = None
    settings.input_data.target_text = None
    settings.input_data.user_glossary_lines = None
    settings.input_data.ref_source_text = None
    settings.input_data.ref_target_text = None

    if source_text is not None:
        settings.input_data.source_text = source_text
    elif clean_body:
        settings.input_data.source_text = clean_body

    if target_text is not None:
        settings.input_data.target_text = target_text

    if glossary_lines is not None:
        settings.input_data.user_glossary_lines = glossary_lines

    if ref_source_text is not None:
        settings.input_data.ref_source_text = ref_source_text

    if ref_target_text is not None:
        settings.input_data.ref_target_text = ref_target_text

    settings.input_data.match_glossary_only = match_glossary_only

    return settings


def check_sender(
    sender: str,
    allowed_senders: list[str],
    whitelist_enabled: bool,
) -> bool:
    """Check whether the sender is authorized to submit requests.

    If the whitelist is disabled or the allowed senders list is empty,
    all senders are permitted.

    Args:
        sender: The sender's email address (may include display name).
        allowed_senders: List of permitted email addresses.
        whitelist_enabled: Whether sender whitelisting is active.

    Returns:
        True if the sender is allowed or whitelisting is disabled.
    """
    if not whitelist_enabled or not allowed_senders:
        return True
    sender_addr = sender.strip().lower()
    return any(allowed.strip().lower() == sender_addr for allowed in allowed_senders)
