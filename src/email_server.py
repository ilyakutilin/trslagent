import asyncio
import base64
import hashlib
import hmac

from aiohttp import web

from src.config import Settings, logger
from src.email_processor import (
    build_settings_from_email,
    check_sender,
    download_attachment,
    fetch_attachments,
    fetch_email_content,
    send_reply,
)
from src.main import main


SIGNATURE_TOLERANCE = 300  # 5 minutes


def _verify_svix(payload: bytes, headers: dict, secret: str) -> bool:
    if not secret:
        logger.warning("No webhook secret configured — skipping signature verification")
        return True

    svix_id = headers.get("svix-id", "")
    svix_timestamp = headers.get("svix-timestamp", "")
    svix_signature = headers.get("svix-signature", "")

    if not svix_id or not svix_timestamp or not svix_signature:
        logger.warning("Missing svix headers")
        return False

    try:
        ts = int(svix_timestamp)
    except ValueError:
        logger.warning("Invalid svix-timestamp: {}", svix_timestamp)
        return False

    import time

    if abs(time.time() - ts) > SIGNATURE_TOLERANCE:
        logger.warning("Svix timestamp outside tolerance window")
        return False

    signed_content = f"{svix_id}.{svix_timestamp}".encode() + b"." + payload

    secret_key = secret
    if secret_key.startswith("whsec_"):
        secret_key = secret_key[6:]

    try:
        secret_bytes = base64.b64decode(secret_key)
    except Exception:
        logger.warning("Failed to base64-decode webhook secret")
        return False

    expected = hmac.new(secret_bytes, signed_content, hashlib.sha256).digest()
    expected_b64 = base64.b64encode(expected).decode()

    for sig in svix_signature.split(" "):
        parts = sig.split(",", 1)
        if len(parts) != 2:
            continue
        version, sig_b64 = parts
        if version == "v1" and hmac.compare_digest(expected_b64, sig_b64):
            return True

    return False


def _extract_sender(from_header: str) -> str:
    """Parse bare email address from 'Name <addr>' or 'addr'."""
    if "<" in from_header and ">" in from_header:
        return from_header.split("<")[1].split(">")[0].strip().lower()
    return from_header.strip().lower()


async def _handle_webhook(request: web.Request, cfg: Settings) -> web.Response:
    raw_body = await request.read()

    email_settings = cfg.email
    if not _verify_svix(raw_body, dict(request.headers), email_settings.resend_webhook_secret):
        return web.Response(status=401, text="Invalid signature")

    try:
        payload = await request.json()
    except Exception:
        logger.warning("Failed to parse webhook JSON")
        return web.Response(status=400, text="Invalid JSON")

    event_type = payload.get("type", "")
    if event_type != "email.received":
        return web.Response(status=200, text="Ignored")

    data = payload.get("data", {})
    email_id = data.get("email_id", "")
    sender_raw = data.get("from", "")
    subject = data.get("subject", "(no subject)")
    message_id = data.get("message_id", "")

    if not email_id:
        return web.Response(status=200, text="Missing email_id")

    sender = _extract_sender(sender_raw)

    asyncio.ensure_future(
        _process_inbound(
            email_id=email_id,
            sender=sender,
            subject=subject,
            message_id=message_id,
            cfg=cfg,
        )
    )

    return web.Response(status=200, text="OK")


async def _process_inbound(
    *,
    email_id: str,
    sender: str,
    subject: str,
    message_id: str,
    cfg: Settings,
) -> None:
    email_settings = cfg.email
    api_key = email_settings.resend_api_key.get_secret_value()

    if not check_sender(
        sender,
        email_settings.allowed_senders,
        email_settings.sender_whitelist_enabled,
    ):
        logger.info("Rejected request from unauthorized sender: {}", sender)
        try:
            await send_reply(
                to=sender,
                subject=subject,
                body=(
                    "Your email address is not authorized to use this translation service.\n\n"
                    "Please contact the administrator for access."
                ),
                message_id=message_id,
                from_address=email_settings.from_address,
                api_key=api_key,
            )
        except Exception:
            logger.exception("Failed to send rejection reply to {}", sender)
        return

    try:
        email_content = await fetch_email_content(email_id, api_key)
        raw_attachments = await fetch_attachments(email_id, api_key)

        attachment_bodies: dict[str, bytes] = {}
        for att in raw_attachments:
            fname = att.get("filename", "")
            dl_url = att.get("download_url", "")
            if not fname or not dl_url:
                continue
            try:
                content = await download_attachment(dl_url)
                attachment_bodies[fname] = content
            except ValueError as e:
                await send_reply(
                    to=sender,
                    subject=subject,
                    body=f"Attachment '{fname}' is too large: {e}",
                    message_id=message_id,
                    from_address=email_settings.from_address,
                    api_key=api_key,
                )
                return
            except Exception:
                logger.exception("Failed to download attachment {}", fname)
                continue

        email_body = email_content.get("text", "") or ""
    except Exception:
        logger.exception("Failed to fetch email content for {}", email_id)
        try:
            await send_reply(
                to=sender,
                subject=subject,
                body="Failed to retrieve your email. Please try again later.",
                message_id=message_id,
                from_address=email_settings.from_address,
                api_key=api_key,
            )
        except Exception:
            logger.exception("Failed to send error reply to {}", sender)
        return

    try:
        translation_settings = build_settings_from_email(
            attachment_bodies=attachment_bodies,
            email_body=email_body,
            default_config_path=email_settings.default_config_path,
        )
    except Exception:
        logger.exception("Failed to build settings from email")
        await _send_error_reply(
            sender=sender,
            subject=subject,
            message_id=message_id,
            email_settings=email_settings,
            api_key=api_key,
            detail="Failed to parse the request. Please check your attachments and try again.",
        )
        return

    try:
        result = await main(cfg=translation_settings)
    except Exception:
        logger.exception("Translation/review failed")
        await _send_error_reply(
            sender=sender,
            subject=subject,
            message_id=message_id,
            email_settings=email_settings,
            api_key=api_key,
            detail="An error occurred during processing. Please try again.",
        )
        return

    if not result:
        await _send_error_reply(
            sender=sender,
            subject=subject,
            message_id=message_id,
            email_settings=email_settings,
            api_key=api_key,
            detail="No output was produced. Check your input and try again.",
        )
        return

    try:
        await send_reply(
            to=sender,
            subject=subject,
            body=result,
            message_id=message_id,
            from_address=email_settings.from_address,
            api_key=api_key,
        )
    except Exception:
        logger.exception("Failed to send result reply to {}", sender)


async def _send_error_reply(
    *,
    sender: str,
    subject: str,
    message_id: str,
    email_settings,
    api_key: str,
    detail: str,
) -> None:
    try:
        await send_reply(
            to=sender,
            subject=subject,
            body=detail,
            message_id=message_id,
            from_address=email_settings.from_address,
            api_key=api_key,
        )
    except Exception:
        logger.exception("Failed to send error reply to {}", sender)


async def serve(cfg: Settings) -> None:
    email_cfg = cfg.email
    app = web.Application()
    app["cfg"] = cfg
    app.router.add_post(
        "/webhook/email",
        lambda req: _handle_webhook(req, req.app["cfg"]),
    )

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, email_cfg.listen_host, email_cfg.listen_port)
    await site.start()

    logger.info(
        "Email webhook server listening on {}:{}",
        email_cfg.listen_host,
        email_cfg.listen_port,
    )

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
