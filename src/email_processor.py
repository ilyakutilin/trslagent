import tempfile
import tomllib
from pathlib import Path

import httpx
from loguru import logger

from src.config import Settings, get_settings

MAX_INDIVIDUAL_ATTACHMENT = 10 * 1024 * 1024  # 10 MB
RESEND_API_BASE = "https://api.resend.com"
RESEND_EMAILS_RECEIVING_BASE = f"{RESEND_API_BASE}/emails/receiving"


async def fetch_email_content(email_id: str, api_key: str) -> dict:
    """Retrieve the raw email (text, html, headers) from Resend."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{RESEND_EMAILS_RECEIVING_BASE}/{email_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


async def fetch_attachments(email_id: str, api_key: str) -> list[dict]:
    """List all attachments for a received email."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{RESEND_EMAILS_RECEIVING_BASE}/{email_id}/attachments",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])


async def download_attachment(download_url: str) -> bytes:
    """Download a single attachment via its temporary URL."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(download_url)
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
) -> None:
    """Send a threaded reply via the Resend send API."""
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"

    async with httpx.AsyncClient(timeout=30) as client:
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
    """Remove file-path keys from TOML content so InputData validator
    does not attempt to read missing files.  We supply the texts directly
    after Settings construction."""
    data = tomllib.loads(toml_content)
    id_section = data.get("input_data", {})
    id_section.pop("source_file_path", None)
    id_section.pop("target_file_path", None)
    id_section.pop("user_glossary_file_path", None)
    data["input_data"] = id_section

    lines: list[str] = []
    for section, values in data.items():
        lines.append(f"[{section}]")
        for k, v in values.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
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

    Loads settings from an attached *config.toml* (if present) or the
    server's default config.  Source / target / glossary texts are supplied
    directly, bypassing file reads.
    """
    source_text: str | None = None
    target_text: str | None = None
    glossary_lines: list[str] | None = None

    if "source.txt" in attachment_bodies:
        source_text = attachment_bodies["source.txt"].decode("utf-8")

    if "target.txt" in attachment_bodies:
        target_text = attachment_bodies["target.txt"].decode("utf-8")

    if "glossary.txt" in attachment_bodies:
        glossary_lines = attachment_bodies["glossary.txt"].decode("utf-8").splitlines()

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
        elif email_body.strip():
            settings.input_data.source_text = email_body.strip()

        if target_text is not None:
            settings.input_data.target_text = target_text

        if glossary_lines is not None:
            settings.input_data.user_glossary_lines = glossary_lines

        return settings

    settings = default_cfg.model_copy(deep=True)
    settings.output_data.print_prompt_only = False
    settings.input_data.source_text = None
    settings.input_data.target_text = None
    settings.input_data.user_glossary_lines = None

    if source_text is not None:
        settings.input_data.source_text = source_text
    elif email_body.strip():
        settings.input_data.source_text = email_body.strip()

    if target_text is not None:
        settings.input_data.target_text = target_text

    if glossary_lines is not None:
        settings.input_data.user_glossary_lines = glossary_lines

    return settings


def check_sender(
    sender: str,
    allowed_senders: list[str],
    whitelist_enabled: bool,
) -> bool:
    """Return True if the sender is allowed to submit requests."""
    if not whitelist_enabled or not allowed_senders:
        return True
    sender_addr = sender.strip().lower()
    return any(allowed.strip().lower() == sender_addr for allowed in allowed_senders)
