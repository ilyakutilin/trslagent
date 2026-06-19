import asyncio
import base64
import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from src.config import EmailSettings, InputData, Settings
from src.email_server import (
    SIGNATURE_TOLERANCE,
    _extract_sender,
    _handle_webhook,
    _process_inbound,
    _send_error_reply,
    _verify_svix,
    serve,
)

WSEC = "whsec_c2VjcmV0LWtleS1mb3ItdGVzdGluZw=="


def _make_svix_signature(
    payload: bytes, svix_id: str, svix_timestamp: str, secret: str = WSEC
) -> str:
    signed = f"{svix_id}.{svix_timestamp}".encode() + b"." + payload
    key = (
        base64.b64decode(secret[6:]) if secret.startswith("whsec_") else secret.encode()
    )
    sig = hmac.new(key, signed, hashlib.sha256).digest()
    return f"v1,{base64.b64encode(sig).decode()}"


def _make_svix_headers(
    payload: bytes,
    svix_id: str = "msg_123",
    svix_timestamp: str | None = None,
    secret: str = WSEC,
) -> dict:
    ts = svix_timestamp or str(int(time.time()))
    return {
        "svix-id": svix_id,
        "svix-timestamp": ts,
        "svix-signature": _make_svix_signature(payload, svix_id, ts, secret),
    }


def _make_email_settings(**kwargs) -> EmailSettings:
    defaults: dict = {
        "resend_api_key": SecretStr("re_key"),
        "resend_webhook_secret": "",
    }
    defaults.update(kwargs)
    return EmailSettings(**defaults)


def _make_cfg(email_overrides: dict | None = None) -> Settings:
    email = _make_email_settings(**(email_overrides or {}))
    return Settings(
        input_data=InputData(
            source_lang="en",  # type: ignore[arg-type]
            target_lang="ru",  # type: ignore[arg-type]
            source_text="dummy",
        ),
        email=email,
    )


class FakeRequest:
    def __init__(self, headers: dict, body: bytes):
        self.headers = headers
        self._body = body

    async def read(self) -> bytes:
        return self._body

    async def json(self) -> dict:
        return json.loads(self._body.decode())


class TestVerifySvix:
    def test_returns_true_when_secret_empty(self):
        assert _verify_svix(b"body", {}, "") is True

    def test_returns_false_when_headers_missing(self):
        assert _verify_svix(b"body", {}, "whsec_test") is False

    def test_returns_false_when_svix_id_missing(self):
        headers = {
            "svix-timestamp": str(int(time.time())),
            "svix-signature": "v1,abc",
        }
        assert _verify_svix(b"body", headers, WSEC) is False

    def test_returns_false_when_timestamp_not_integer(self):
        payload = b'{"type":"email.received"}'
        headers = _make_svix_headers(payload, svix_timestamp="not-a-number")
        assert _verify_svix(payload, headers, WSEC) is False

    def test_returns_false_when_timestamp_outside_tolerance(self):
        payload = b"{}"
        past_ts = str(int(time.time()) - SIGNATURE_TOLERANCE - 10)
        headers = _make_svix_headers(payload, svix_timestamp=past_ts)
        assert _verify_svix(payload, headers, WSEC) is False

    def test_returns_true_for_valid_signature(self):
        payload = b'{"type":"email.received"}'
        headers = _make_svix_headers(payload)
        assert _verify_svix(payload, headers, WSEC) is True

    def test_returns_false_for_wrong_secret(self):
        payload = b'{"type":"email.received"}'
        wrong_secret = "whsec_YW5vdGhlci1zZWNyZXQ="
        headers = _make_svix_headers(payload, secret=wrong_secret)
        assert _verify_svix(payload, headers, WSEC) is False

    def test_returns_false_for_wrong_payload(self):
        headers = _make_svix_headers(b'{"type":"email.received"}')
        assert _verify_svix(b"different payload", headers, WSEC) is False

    def test_strips_whsec_prefix(self):
        payload = b"body"
        headers = _make_svix_headers(payload, secret=WSEC)
        assert _verify_svix(payload, headers, WSEC) is True

    def test_returns_false_for_invalid_base64_secret(self):
        assert _verify_svix(b"body", {}, "whsec_!!!not-valid-base64!!!") is False

    def test_picks_correct_signature_when_multiple_versions(self):
        payload = b"body"
        ts = str(int(time.time()))
        valid_sig = _make_svix_signature(payload, "msg_1", ts, WSEC)
        fake_sig = "v2," + base64.b64encode(b"wrong").decode()
        headers = {
            "svix-id": "msg_1",
            "svix-timestamp": ts,
            "svix-signature": f"{fake_sig} {valid_sig}",
        }
        assert _verify_svix(payload, headers, WSEC) is True

    def test_returns_false_when_no_v1_signature(self):
        payload = b"body"
        ts = str(int(time.time()))
        sig_v2 = f"v2,{base64.b64encode(b'wrong').decode()}"
        headers = {
            "svix-id": "msg_1",
            "svix-timestamp": ts,
            "svix-signature": sig_v2,
        }
        assert _verify_svix(payload, headers, WSEC) is False


class TestExtractSender:
    def test_name_and_angle_brackets(self):
        assert _extract_sender("John Doe <john@example.com>") == "john@example.com"

    def test_bare_address(self):
        assert _extract_sender("john@example.com") == "john@example.com"

    def test_lowercases(self):
        assert _extract_sender("John@Example.COM") == "john@example.com"

    def test_trims_whitespace(self):
        assert _extract_sender("  john@example.com  ") == "john@example.com"

    def test_with_display_name_having_special_chars(self):
        result = _extract_sender("=?UTF-8?B?TmFtZQ==?= <user@domain.com>")
        assert result == "user@domain.com"


class TestHandleWebhook:
    def _make_payload(self, overrides: dict | None = None) -> dict:
        base = {
            "type": "email.received",
            "data": {
                "email_id": "email-1",
                "from": "user@example.com",
                "subject": "Test subject",
                "message_id": "mid-1",
                "to": ["trsl@mydomain.com"],
            },
        }
        if overrides:
            base.update(overrides)
        return base

    def _make_req(
        self,
        payload: dict | None = None,
        raw_body: bytes | None = None,
        headers: dict | None = None,
        with_svix: bool = False,
    ) -> FakeRequest:
        body = raw_body or json.dumps(payload or self._make_payload()).encode()
        hdrs = headers or (_make_svix_headers(body) if with_svix else {})
        return FakeRequest(headers=hdrs, body=body)

    async def test_returns_401_for_invalid_signature(self):
        payload = self._make_payload()
        body = json.dumps(payload).encode()
        req = FakeRequest(
            headers={
                "svix-id": "x",
                "svix-timestamp": str(int(time.time())),
                "svix-signature": "v1,bad",
            },
            body=body,
        )
        cfg = _make_cfg({"resend_webhook_secret": WSEC})
        resp = await _handle_webhook(req, cfg)  # type: ignore[arg-type]
        assert resp.status == 401

    async def test_returns_401_when_svix_headers_absent(self):
        payload = self._make_payload()
        body = json.dumps(payload).encode()
        req = FakeRequest(headers={}, body=body)
        cfg = _make_cfg({"resend_webhook_secret": WSEC})
        resp = await _handle_webhook(req, cfg)  # type: ignore[arg-type]
        assert resp.status == 401

    async def test_returns_200_when_no_secret_configured(self):
        payload = self._make_payload()
        body = json.dumps(payload).encode()
        req = FakeRequest(headers={}, body=body)
        cfg = _make_cfg({"resend_webhook_secret": ""})
        with patch(
            "src.email_server._process_inbound", new_callable=AsyncMock
        ) as mock_proc:
            resp = await _handle_webhook(req, cfg)  # type: ignore[arg-type]
            assert resp.status == 200
            await asyncio.sleep(0)
            mock_proc.assert_called_once()

    async def test_returns_400_for_invalid_json(self):
        body = b"not-json"
        headers = _make_svix_headers(body)
        req = FakeRequest(headers=headers, body=body)
        cfg = _make_cfg({"resend_webhook_secret": WSEC})
        resp = await _handle_webhook(req, cfg)  # type: ignore[arg-type]
        assert resp.status == 400

    async def test_returns_200_for_non_email_received_event(self):
        payload = {"type": "email.sent", "data": {}}
        body = json.dumps(payload).encode()
        headers = _make_svix_headers(body)
        req = FakeRequest(headers=headers, body=body)
        cfg = _make_cfg({"resend_webhook_secret": WSEC})
        resp = await _handle_webhook(req, cfg)  # type: ignore[arg-type]
        assert resp.status == 200
        assert resp.text is not None
        assert "Ignored" in resp.text

    async def test_returns_200_missing_email_id(self):
        payload = {"type": "email.received", "data": {"from": "u@x.com"}}
        body = json.dumps(payload).encode()
        headers = _make_svix_headers(body)
        req = FakeRequest(headers=headers, body=body)
        cfg = _make_cfg({"resend_webhook_secret": WSEC})
        resp = await _handle_webhook(req, cfg)  # type: ignore[arg-type]
        assert resp.status == 200
        assert resp.text is not None
        assert "Missing email_id" in resp.text

    async def test_ignores_when_allowed_recipient_set_and_not_matching(self):
        payload = {
            "type": "email.received",
            "data": {
                "email_id": "email-1",
                "from": "u@x.com",
                "to": ["other@domain.com"],
            },
        }
        body = json.dumps(payload).encode()
        headers = _make_svix_headers(body)
        req = FakeRequest(headers=headers, body=body)
        cfg = _make_cfg(
            {
                "resend_webhook_secret": WSEC,
                "allowed_recipient": "trsl@mydomain.com",
            }
        )
        resp = await _handle_webhook(req, cfg)  # type: ignore[arg-type]
        assert resp.status == 200
        assert resp.text is not None
        assert "Ignored" in resp.text

    async def test_processes_when_allowed_recipient_matches(self):
        payload = self._make_payload()
        body = json.dumps(payload).encode()
        headers = _make_svix_headers(body)
        req = FakeRequest(headers=headers, body=body)
        cfg = _make_cfg(
            {
                "resend_webhook_secret": WSEC,
                "allowed_recipient": "trsl@mydomain.com",
                "sender_whitelist_enabled": False,
            }
        )
        with patch(
            "src.email_server._process_inbound", new_callable=AsyncMock
        ) as mock_proc:
            resp = await _handle_webhook(req, cfg)  # type: ignore[arg-type]
            assert resp.status == 200
            assert resp.text is not None
            assert "OK" in resp.text
            await asyncio.sleep(0)
            mock_proc.assert_called_once()
            kwargs = mock_proc.call_args.kwargs
            assert kwargs["email_id"] == "email-1"
            assert kwargs["sender"] == "user@example.com"

    async def test_spawns_process_inbound_for_valid_request(self):
        payload = self._make_payload()
        body = json.dumps(payload).encode()
        headers = _make_svix_headers(body)
        req = FakeRequest(headers=headers, body=body)
        cfg = _make_cfg(
            {
                "resend_webhook_secret": WSEC,
                "sender_whitelist_enabled": False,
            }
        )
        with patch(
            "src.email_server._process_inbound", new_callable=AsyncMock
        ) as mock_proc:
            resp = await _handle_webhook(req, cfg)  # type: ignore[arg-type]
            assert resp.status == 200
            await asyncio.sleep(0)
            mock_proc.assert_called_once()


class TestProcessInbound:
    def _setup_mocks(
        self,
        mocker,
        *,
        fetch_email_return=None,
        fetch_email_side_effect=None,
        fetch_attach_return=None,
        build_settings_return=None,
        build_settings_side_effect=None,
        main_return: str | None = "result",
        main_side_effect=None,
        send_reply_side_effect=None,
        download_side_effect=None,
    ):
        mocks = {}
        if fetch_email_side_effect:
            mocks["fetch_email_content"] = mocker.patch(
                "src.email_server.fetch_email_content",
                side_effect=fetch_email_side_effect,
            )
        elif fetch_email_return is not None:
            mocks["fetch_email_content"] = mocker.patch(
                "src.email_server.fetch_email_content",
                new_callable=AsyncMock,
                return_value=fetch_email_return,
            )
        if fetch_attach_return is not None:
            mocks["fetch_attachments"] = mocker.patch(
                "src.email_server.fetch_attachments",
                new_callable=AsyncMock,
                return_value=fetch_attach_return,
            )
        if build_settings_side_effect:
            mocks["build_settings_from_email"] = mocker.patch(
                "src.email_server.build_settings_from_email",
                side_effect=build_settings_side_effect,
            )
        elif build_settings_return is not None:
            mocks["build_settings_from_email"] = mocker.patch(
                "src.email_server.build_settings_from_email",
                return_value=build_settings_return,
            )
        if main_side_effect:
            mocks["main"] = mocker.patch(
                "src.email_server.main",
                side_effect=main_side_effect,
            )
        elif main_return is not None:
            mocks["main"] = mocker.patch(
                "src.email_server.main",
                new_callable=AsyncMock,
                return_value=main_return,
            )
        mock_reply_kwargs = {"new_callable": AsyncMock}
        if send_reply_side_effect:
            mock_reply_kwargs["side_effect"] = send_reply_side_effect
        mocks["send_reply"] = mocker.patch(
            "src.email_server.send_reply", **mock_reply_kwargs
        )
        if download_side_effect is not None:
            mocks["download_attachment"] = mocker.patch(
                "src.email_server.download_attachment",
                new_callable=AsyncMock,
                side_effect=download_side_effect,
            )
        return mocks

    async def test_rejects_unauthorized_sender_and_no_reply(self, mocker):
        mocks = self._setup_mocks(mocker, main_return=None)
        cfg = _make_cfg(
            {
                "resend_api_key": SecretStr("re_key"),
                "allowed_senders": ["allowed@x.com"],
                "sender_whitelist_enabled": True,
            }
        )
        await _process_inbound(
            email_id="e1",
            sender="blocked@x.com",
            subject="Hi",
            message_id="mid",
            cfg=cfg,
        )
        mocks["send_reply"].assert_not_called()

    async def test_sends_error_reply_on_fetch_failure(self, mocker):
        mocks = self._setup_mocks(
            mocker,
            fetch_email_side_effect=Exception("network error"),
            main_return=None,
        )
        cfg = _make_cfg(
            {
                "resend_api_key": SecretStr("re_key"),
                "sender_whitelist_enabled": False,
            }
        )
        await _process_inbound(
            email_id="e1",
            sender="u@x.com",
            subject="Hi",
            message_id="mid",
            cfg=cfg,
        )
        mocks["send_reply"].assert_called_once()
        assert "Failed to retrieve" in mocks["send_reply"].call_args.kwargs["body"]

    async def test_sends_reply_when_attachment_too_large(self, mocker):
        mocks = self._setup_mocks(
            mocker,
            fetch_email_return={"text": "body"},
            fetch_attach_return=[
                {"filename": "big.txt", "download_url": "http://dl/big"},
            ],
            main_return=None,
            download_side_effect=ValueError("Attachment too large: ..."),
        )
        cfg = _make_cfg(
            {
                "resend_api_key": SecretStr("re_key"),
                "sender_whitelist_enabled": False,
            }
        )
        await _process_inbound(
            email_id="e1",
            sender="u@x.com",
            subject="Hi",
            message_id="mid",
            cfg=cfg,
        )
        mocks["send_reply"].assert_called_once()
        assert "too large" in mocks["send_reply"].call_args.kwargs["body"]

    async def test_skips_attachments_without_filename_or_url(self, mocker):
        mocks = self._setup_mocks(
            mocker,
            fetch_email_return={"text": "body"},
            fetch_attach_return=[
                {"filename": "", "download_url": "http://dl/a"},
                {"filename": "ok.txt", "download_url": ""},
            ],
            build_settings_return=_make_cfg(),
        )
        mock_dl = mocker.patch(
            "src.email_server.download_attachment", new_callable=AsyncMock
        )
        cfg = _make_cfg({"sender_whitelist_enabled": False})
        await _process_inbound(
            email_id="e1",
            sender="u@x.com",
            subject="Hi",
            message_id="mid",
            cfg=cfg,
        )
        mock_dl.assert_not_called()
        assert mocks["send_reply"].call_args.kwargs["body"] == "result"

    async def test_sends_error_reply_when_build_settings_fails(self, mocker):
        mocks = self._setup_mocks(
            mocker,
            fetch_email_return={"text": "body"},
            fetch_attach_return=[],
            build_settings_side_effect=Exception("bad config"),
            main_return=None,
        )
        cfg = _make_cfg({"sender_whitelist_enabled": False})
        await _process_inbound(
            email_id="e1",
            sender="u@x.com",
            subject="Hi",
            message_id="mid",
            cfg=cfg,
        )
        mocks["send_reply"].assert_called_once()
        assert "Failed to parse" in mocks["send_reply"].call_args.kwargs["body"]

    async def test_sends_error_reply_when_translation_fails(self, mocker):
        mocks = self._setup_mocks(
            mocker,
            fetch_email_return={"text": "body"},
            fetch_attach_return=[],
            build_settings_return=_make_cfg(),
            main_side_effect=Exception("LLM error"),
        )
        cfg = _make_cfg({"sender_whitelist_enabled": False})
        await _process_inbound(
            email_id="e1",
            sender="u@x.com",
            subject="Hi",
            message_id="mid",
            cfg=cfg,
        )
        mocks["send_reply"].assert_called_once()
        assert (
            "An error occurred during processing"
            in mocks["send_reply"].call_args.kwargs["body"]
        )

    async def test_sends_error_reply_when_main_returns_none(self, mocker):
        mocks = self._setup_mocks(
            mocker,
            fetch_email_return={"text": "body"},
            fetch_attach_return=[],
            build_settings_return=_make_cfg(),
            main_return=None,
        )
        cfg = _make_cfg({"sender_whitelist_enabled": False})
        await _process_inbound(
            email_id="e1",
            sender="u@x.com",
            subject="Hi",
            message_id="mid",
            cfg=cfg,
        )
        mocks["send_reply"].assert_called_once()
        assert "No output was produced" in mocks["send_reply"].call_args.kwargs["body"]

    async def test_sends_result_reply_on_success(self, mocker):
        mocks = self._setup_mocks(
            mocker,
            fetch_email_return={"text": "body"},
            fetch_attach_return=[],
            build_settings_return=_make_cfg(),
            main_return="Translation result",
        )
        cfg = _make_cfg({"sender_whitelist_enabled": False})
        await _process_inbound(
            email_id="e1",
            sender="u@x.com",
            subject="Hi",
            message_id="mid",
            cfg=cfg,
        )
        mocks["send_reply"].assert_called_once()
        kwargs = mocks["send_reply"].call_args.kwargs
        assert kwargs["body"] == "Translation result"
        assert kwargs["to"] == "u@x.com"
        assert kwargs["message_id"] == "mid"

    async def test_continues_after_download_exception_for_one_attachment(self, mocker):
        call_count = 0

        async def dl_side_effect(url):
            nonlocal call_count
            call_count += 1
            if "bad" in url:
                raise Exception("download failed")
            return b"good content"

        mocks = self._setup_mocks(
            mocker,
            fetch_email_return={"text": "body"},
            fetch_attach_return=[
                {"filename": "bad.txt", "download_url": "http://dl/bad"},
                {"filename": "good.txt", "download_url": "http://dl/good"},
            ],
            build_settings_return=_make_cfg(),
            download_side_effect=dl_side_effect,
        )
        cfg = _make_cfg({"sender_whitelist_enabled": False})
        await _process_inbound(
            email_id="e1",
            sender="u@x.com",
            subject="Hi",
            message_id="mid",
            cfg=cfg,
        )
        assert call_count == 2
        assert mocks["send_reply"].call_args.kwargs["body"] == "result"

    async def test_send_reply_failure_is_silent(self, mocker):
        mocks = self._setup_mocks(
            mocker,
            fetch_email_return={"text": "body"},
            fetch_attach_return=[],
            build_settings_return=_make_cfg(),
            send_reply_side_effect=Exception("send failed"),
        )
        cfg = _make_cfg({"sender_whitelist_enabled": False})
        await _process_inbound(
            email_id="e1",
            sender="u@x.com",
            subject="Hi",
            message_id="mid",
            cfg=cfg,
        )
        mocks["send_reply"].assert_called_once()


# Serves as a container for serve() tests; kept as a class to allow
# future serve-related tests to be added without restructuring.
class TestServe:
    async def test_creates_app_with_webhook_route(self, mocker):
        mock_app = mocker.MagicMock()
        mocker.patch("src.email_server.web.Application", return_value=mock_app)
        mock_runner = mocker.patch("aiohttp.web.AppRunner", autospec=True)
        mock_site = mocker.patch("aiohttp.web.TCPSite", autospec=True)
        runner_instance = mock_runner.return_value
        runner_instance.setup = AsyncMock()
        runner_instance.cleanup = AsyncMock()
        site_instance = mock_site.return_value
        site_instance.start = AsyncMock()

        event_mock = mocker.patch("asyncio.Event")
        event_instance = event_mock.return_value
        event_instance.wait = AsyncMock(side_effect=KeyboardInterrupt)

        cfg = _make_cfg({"listen_host": "127.0.0.1", "listen_port": 9999})
        with pytest.raises(KeyboardInterrupt):
            await serve(cfg)

        mock_app.router.add_post.assert_called_once_with("/webhook/email", mocker.ANY)
        runner_instance.setup.assert_called_once()
        runner_instance.cleanup.assert_called_once()
        mock_site.assert_called_once_with(mocker.ANY, "127.0.0.1", 9999)
        site_instance.start.assert_called_once()


class TestSendErrorReply:
    async def test_calls_send_reply_with_correct_params(self, mocker):
        mock_reply = mocker.patch("src.email_server.send_reply", new_callable=AsyncMock)
        mock_email_settings = mocker.MagicMock()
        mock_email_settings.from_address = "bot@example.com"

        await _send_error_reply(
            sender="user@x.com",
            subject="Test",
            message_id="mid",
            email_settings=mock_email_settings,
            api_key="key",
            detail="Something went wrong",
        )

        mock_reply.assert_called_once()
        kwargs = mock_reply.call_args.kwargs
        assert kwargs["to"] == "user@x.com"
        assert kwargs["subject"] == "Test"
        assert kwargs["body"] == "Something went wrong"
        assert kwargs["message_id"] == "mid"
        assert kwargs["from_address"] == "bot@example.com"
        assert kwargs["api_key"] == "key"

    async def test_swallows_send_reply_exception(self, mocker):
        mocker.patch(
            "src.email_server.send_reply",
            new_callable=AsyncMock,
            side_effect=Exception("send failed"),
        )
        mock_email_settings = mocker.MagicMock()

        await _send_error_reply(
            sender="user@x.com",
            subject="Test",
            message_id="mid",
            email_settings=mock_email_settings,
            api_key="key",
            detail="Something went wrong",
        )
