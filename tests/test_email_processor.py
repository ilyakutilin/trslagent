import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from iso639 import Lang
from pydantic import SecretStr

from src.config import EmailSettings, InputData, ProxySettings, Settings
from src.email_processor import (
    MAX_INDIVIDUAL_ATTACHMENT,
    RESEND_API_BASE,
    RESEND_EMAILS_RECEIVING_BASE,
    _patch_toml_for_attachments,
    build_settings_from_email,
    check_sender,
    download_attachment,
    fetch_attachments,
    fetch_email_content,
    send_reply,
)
from src.http_client import create_client

SRC_TRG_TOML = 'source_lang = "en"\ntarget_lang = "ru"\n'


def _make_default_settings() -> Settings:
    """Minimal Settings with valid input_data for translation."""
    return Settings(
        input_data=InputData(
            source_lang=Lang("en"),
            target_lang=Lang("ru"),
            source_text="default source",
        ),
        email=EmailSettings(resend_api_key=SecretStr("re_key")),
    )


class TestCheckSender:
    def test_returns_true_when_whitelist_disabled(self):
        assert check_sender("any@x.com", [], False) is True

    def test_returns_true_when_allowed_senders_empty(self):
        assert check_sender("any@x.com", [], True) is True

    def test_returns_true_when_sender_matches_exactly(self):
        assert check_sender("a@b.com", ["a@b.com"], True) is True

    def test_case_insensitive_match(self):
        assert check_sender("A@B.COM", ["a@b.com"], True) is True

    def test_returns_false_when_sender_not_in_list(self):
        assert check_sender("c@d.com", ["a@b.com"], True) is False

    def test_strips_whitespace_from_sender(self):
        assert check_sender("  a@b.com  ", ["a@b.com"], True) is True

    def test_strips_whitespace_from_allowed(self):
        assert check_sender("a@b.com", ["  a@b.com  "], True) is True

    def test_empty_sender_matches_empty_allowed(self):
        assert check_sender("", [""], True) is True

    def test_empty_sender_not_matched_when_allowed_has_real_email(self):
        assert check_sender("", ["real@x.com"], True) is False


class TestFetchEmailContent:
    @respx.mock
    async def test_returns_json_on_success(self):
        route = respx.get(f"{RESEND_EMAILS_RECEIVING_BASE}/email-123").mock(
            return_value=httpx.Response(
                200, json={"text": "Hello", "html": "<p>Hello</p>"}
            )
        )

        result = await fetch_email_content("email-123", "api-key")
        assert result == {"text": "Hello", "html": "<p>Hello</p>"}
        assert route.called

    @respx.mock
    async def test_raises_on_http_error(self):
        respx.get(f"{RESEND_EMAILS_RECEIVING_BASE}/email-404").mock(
            return_value=httpx.Response(404, json={"error": "not found"})
        )
        with pytest.raises(httpx.HTTPStatusError):
            await fetch_email_content("email-404", "api-key")

    @respx.mock
    async def test_sends_auth_header(self):
        route = respx.get(
            f"{RESEND_EMAILS_RECEIVING_BASE}/email-1",
            headers={"Authorization": "Bearer secret-key"},
        ).mock(return_value=httpx.Response(200, json={}))
        await fetch_email_content("email-1", "secret-key")
        assert route.called

    @respx.mock
    async def test_injected_client_is_not_closed(self):
        respx.get(f"{RESEND_EMAILS_RECEIVING_BASE}/email-1").mock(
            return_value=httpx.Response(200, json={"text": "Hello"})
        )
        client = create_client()
        try:
            result = await fetch_email_content("email-1", "key", client=client)
            assert result == {"text": "Hello"}
            assert not client.is_closed
        finally:
            await client.aclose()


class TestFetchAttachments:
    @respx.mock
    async def test_returns_data_list(self):
        respx.get(f"{RESEND_EMAILS_RECEIVING_BASE}/email-1/attachments").mock(
            return_value=httpx.Response(200, json={"data": [{"filename": "a.txt"}]})
        )

        result = await fetch_attachments("email-1", "key")
        assert result == [{"filename": "a.txt"}]

    @respx.mock
    async def test_returns_empty_list_when_no_data_key(self):
        respx.get(f"{RESEND_EMAILS_RECEIVING_BASE}/email-1/attachments").mock(
            return_value=httpx.Response(200, json={})
        )

        result = await fetch_attachments("email-1", "key")
        assert result == []


class TestDownloadAttachment:
    @respx.mock
    async def test_returns_content_on_success(self):
        respx.get("https://storage.example.com/file.bin").mock(
            return_value=httpx.Response(200, content=b"hello world")
        )
        result = await download_attachment("https://storage.example.com/file.bin")
        assert result == b"hello world"

    @respx.mock
    async def test_raises_valueerror_when_content_too_large(self):
        large_content = b"x" * (MAX_INDIVIDUAL_ATTACHMENT + 1)
        respx.get("https://storage.example.com/huge.bin").mock(
            return_value=httpx.Response(200, content=large_content)
        )
        with pytest.raises(ValueError, match="Attachment too large"):
            await download_attachment("https://storage.example.com/huge.bin")

    @respx.mock
    async def test_content_at_limit_is_ok(self):
        content = b"x" * MAX_INDIVIDUAL_ATTACHMENT
        respx.get("https://storage.example.com/ok.bin").mock(
            return_value=httpx.Response(200, content=content)
        )
        result = await download_attachment("https://storage.example.com/ok.bin")
        assert len(result) == MAX_INDIVIDUAL_ATTACHMENT

    @respx.mock
    async def test_raises_on_http_error(self):
        respx.get("https://storage.example.com/err.bin").mock(
            return_value=httpx.Response(500)
        )
        with pytest.raises(httpx.HTTPStatusError):
            await download_attachment("https://storage.example.com/err.bin")

    @respx.mock
    async def test_injected_client_is_not_closed(self):
        respx.get("https://storage.example.com/file.bin").mock(
            return_value=httpx.Response(200, content=b"hello world")
        )
        client = create_client()
        try:
            result = await download_attachment(
                "https://storage.example.com/file.bin", client=client
            )
            assert result == b"hello world"
            assert not client.is_closed
        finally:
            await client.aclose()

    async def test_injected_client_slow_response_still_succeeds(self):
        async def slow_handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.05)
            return httpx.Response(200, content=b"hello world")

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(slow_handler), timeout=0.01
        )
        try:
            result = await download_attachment(
                "https://storage.example.com/slow.bin", client=client
            )
            assert result == b"hello world"
        finally:
            await client.aclose()


class TestSendReply:
    @respx.mock
    async def test_sends_correct_json_payload(self):
        route = respx.post(f"{RESEND_API_BASE}/emails").mock(
            return_value=httpx.Response(200, json={"id": "msg-456"})
        )
        await send_reply(
            to="user@x.com",
            subject="Test",
            body="Result body",
            message_id="mid-123",
            from_address="Agent <a@b.com>",
            api_key="key",
        )
        assert route.called
        req = route.calls.last.request
        payload = json.loads(req.content)
        assert payload["from"] == "Agent <a@b.com>"
        assert payload["to"] == ["user@x.com"]
        assert payload["subject"] == "Re: Test"
        assert payload["text"] == "Result body"
        assert "In-Reply-To" in payload["headers"]
        assert payload["headers"]["In-Reply-To"] == "mid-123"

    @respx.mock
    async def test_does_not_double_prefix_re(self):
        route = respx.post(f"{RESEND_API_BASE}/emails").mock(
            return_value=httpx.Response(200, json={"id": "msg-456"})
        )
        await send_reply(
            to="user@x.com",
            subject="Re: Test",
            body="Result",
            message_id="mid",
            from_address="a@b.com",
            api_key="key",
        )
        payload = json.loads(route.calls.last.request.content)
        assert payload["subject"] == "Re: Test"

    @respx.mock
    async def test_case_insensitive_re_prefix(self):
        route = respx.post(f"{RESEND_API_BASE}/emails").mock(
            return_value=httpx.Response(200, json={"id": "msg-456"})
        )
        await send_reply(
            to="user@x.com",
            subject="re: Test",
            body="Result",
            message_id="mid",
            from_address="a@b.com",
            api_key="key",
        )
        payload = json.loads(route.calls.last.request.content)
        assert payload["subject"] == "re: Test"

    @respx.mock
    async def test_raises_on_non_200(self):
        respx.post(f"{RESEND_API_BASE}/emails").mock(
            return_value=httpx.Response(400, json={"error": "bad"})
        )
        with pytest.raises(httpx.HTTPStatusError):
            await send_reply(
                to="user@x.com",
                subject="Test",
                body="Body",
                message_id="mid",
                from_address="a@b.com",
                api_key="key",
            )

    @pytest.mark.asyncio
    async def test_proxy_settings_forwarded_to_owned_client(self, mocker):
        proxy = ProxySettings(enabled=False)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = httpx.Response(
            200,
            json={"id": "msg-1"},
            request=httpx.Request("POST", f"{RESEND_API_BASE}/emails"),
        )
        mock_create = mocker.patch(
            "src.email_processor.create_client", return_value=mock_client
        )
        await send_reply(
            to="user@x.com",
            subject="Test",
            body="Body",
            message_id="mid",
            from_address="a@b.com",
            api_key="key",
            proxy_settings=proxy,
        )
        mock_create.assert_called_once_with(timeout=30, proxy_settings=proxy)
        mock_client.__aexit__.assert_awaited_once()

    @respx.mock
    async def test_html_wraps_body_in_pre_tag(self):
        # NOTE: body is interpolated directly into HTML without escaping,
        # which is a known XSS vector.
        route = respx.post(f"{RESEND_API_BASE}/emails").mock(
            return_value=httpx.Response(200, json={"id": "msg"})
        )
        await send_reply(
            to="user@x.com",
            subject="Test",
            body="<unsafe>",
            message_id="mid",
            from_address="a@b.com",
            api_key="key",
        )
        payload = json.loads(route.calls.last.request.content)
        expected_html = (
            '<pre style="white-space:pre-wrap;font-family:monospace"><unsafe></pre>'
        )
        assert payload["html"] == expected_html


class TestPatchTomlForAttachments:
    # NOTE: _patch_toml_for_attachments accepts source_text, target_text,
    # glossary_lines parameters but the implementation ignores them.
    # Tests faithfully pass None for these unused params.
    def test_removes_file_path_keys(self):
        toml = (
            "[input_data]\n" + SRC_TRG_TOML + 'source_file_path = "/some/path"\n'
            'target_file_path = "/other/path"\n'
            'user_glossary_file_path = "/glossary/path"\n'
            'ref_source_file_path = "/ref/src/path"\n'
            'ref_target_file_path = "/ref/tgt/path"\n'
        )
        result = _patch_toml_for_attachments(toml, None, None, None)
        assert "source_file_path" not in result
        assert "target_file_path" not in result
        assert "user_glossary_file_path" not in result
        assert "ref_source_file_path" not in result
        assert "ref_target_file_path" not in result

    def test_preserves_other_sections(self):
        toml = (
            "[llm]\n"
            'model = "my-model"\n'
            "[input_data]\n" + SRC_TRG_TOML + 'source_file_path = "x.txt"\n'
        )
        result = _patch_toml_for_attachments(toml, None, None, None)
        assert 'model = "my-model"' in result

    def test_handles_non_string_values(self):
        toml = "[chunk]\nsize = 5000\nmax_concurrent = 5\n[input_data]\n" + SRC_TRG_TOML
        result = _patch_toml_for_attachments(toml, None, None, None)
        assert "size = 5000" in result
        assert "max_concurrent = 5" in result

    def test_handles_boolean_values(self):
        toml = "[input_data]\n" + SRC_TRG_TOML + "auto_glossary = true\n"
        result = _patch_toml_for_attachments(toml, None, None, None)
        assert "auto_glossary = true" in result

    def test_handles_boolean_false_value(self):
        toml = "[input_data]\n" + SRC_TRG_TOML + "auto_glossary = false\n"
        result = _patch_toml_for_attachments(toml, None, None, None)
        assert "auto_glossary = false" in result


class TestBuildSettingsFromEmail:
    def test_with_source_txt_attachment(self):
        cfg = _make_default_settings()
        result = build_settings_from_email(
            attachment_bodies={"source.txt": b"Hello world"},
            email_body="",
            default_cfg=cfg,
        )
        assert result.input_data.source_text == "Hello world"

    def test_with_target_txt_attachment(self):
        cfg = _make_default_settings()
        result = build_settings_from_email(
            attachment_bodies={
                "source.txt": b"source",
                "target.txt": b"target text",
            },
            email_body="",
            default_cfg=cfg,
        )
        assert result.input_data.target_text == "target text"

    def test_with_glossary_txt_attachment(self):
        cfg = _make_default_settings()
        result = build_settings_from_email(
            attachment_bodies={
                "source.txt": b"source",
                "glossary.txt": b"term = translation\nterm2 = translation2\n",
            },
            email_body="",
            default_cfg=cfg,
        )
        assert result.input_data.user_glossary_lines == [
            "term = translation",
            "term2 = translation2",
        ]

    def test_email_body_as_source_when_no_source_txt(self):
        cfg = _make_default_settings()
        result = build_settings_from_email(
            attachment_bodies={},
            email_body="Email body text",
            default_cfg=cfg,
        )
        assert result.input_data.source_text == "Email body text"

    def test_source_txt_takes_precedence_over_email_body(self):
        cfg = _make_default_settings()
        result = build_settings_from_email(
            attachment_bodies={"source.txt": b"From attachment"},
            email_body="From email body",
            default_cfg=cfg,
        )
        assert result.input_data.source_text == "From attachment"

    def test_empty_email_body_not_used_as_source(self):
        cfg = _make_default_settings()
        result = build_settings_from_email(
            attachment_bodies={},
            email_body="   \n  ",
            default_cfg=cfg,
        )
        assert result.input_data.source_text is None

    def test_uses_default_config_deep_copy(self):
        cfg = _make_default_settings()
        result = build_settings_from_email(
            attachment_bodies={},
            email_body="body",
            default_cfg=cfg,
        )
        assert result is not cfg
        assert result.email.resend_api_key.get_secret_value() == "re_key"

    def test_print_prompt_only_set_to_false(self):
        cfg = _make_default_settings()
        cfg.output_data.print_prompt_only = True
        result = build_settings_from_email(
            attachment_bodies={},
            email_body="body",
            default_cfg=cfg,
        )
        assert result.output_data.print_prompt_only is False

    def test_with_config_toml_attachment(self):
        cfg = _make_default_settings()
        config_toml = (
            "[llm]\n"
            'model = "custom-model"\n'
            "[input_data]\n" + SRC_TRG_TOML + 'source_file_path = "/fake/source.txt"\n'
        )
        result = build_settings_from_email(
            attachment_bodies={
                "config.toml": config_toml.encode("utf-8"),
                "source.txt": b"From source attachment",
            },
            email_body="",
            default_cfg=cfg,
        )
        assert result.llm.model == "custom-model"
        assert result.input_data.source_text == "From source attachment"

    def test_with_config_toml_and_no_source_falls_back_to_email_body(
        self,
    ):
        cfg = _make_default_settings()
        config_toml = "[input_data]\n" + SRC_TRG_TOML
        result = build_settings_from_email(
            attachment_bodies={"config.toml": config_toml.encode("utf-8")},
            email_body="Fallback body",
            default_cfg=cfg,
        )
        assert result.input_data.source_text == "Fallback body"

    def test_config_toml_with_source_txt_ignores_email_body(self):
        cfg = _make_default_settings()
        config_toml = "[input_data]\n" + SRC_TRG_TOML
        result = build_settings_from_email(
            attachment_bodies={
                "config.toml": config_toml.encode("utf-8"),
                "source.txt": b"Attachment source",
            },
            email_body="Email body fallback",
            default_cfg=cfg,
        )
        assert result.input_data.source_text == "Attachment source"

    def test_with_ref_source_txt_attachment(self):
        cfg = _make_default_settings()
        result = build_settings_from_email(
            attachment_bodies={
                "source.txt": b"source",
                "ref_source.txt": b"Reference source text",
            },
            email_body="",
            default_cfg=cfg,
        )
        assert result.input_data.ref_source_text == "Reference source text"

    def test_with_ref_target_txt_attachment(self):
        cfg = _make_default_settings()
        result = build_settings_from_email(
            attachment_bodies={
                "source.txt": b"source",
                "ref_target.txt": b"Reference target text",
            },
            email_body="",
            default_cfg=cfg,
        )
        assert result.input_data.ref_target_text == "Reference target text"

    def test_with_both_ref_attachments(self):
        cfg = _make_default_settings()
        result = build_settings_from_email(
            attachment_bodies={
                "source.txt": b"source",
                "ref_source.txt": b"Source ref",
                "ref_target.txt": b"Target ref",
            },
            email_body="",
            default_cfg=cfg,
        )
        assert result.input_data.ref_source_text == "Source ref"
        assert result.input_data.ref_target_text == "Target ref"

    def test_ref_text_none_when_no_ref_attachments(self):
        cfg = _make_default_settings()
        result = build_settings_from_email(
            attachment_bodies={"source.txt": b"source"},
            email_body="",
            default_cfg=cfg,
        )
        assert result.input_data.ref_source_text is None
        assert result.input_data.ref_target_text is None

    def test_with_config_toml_and_ref_attachments(self):
        cfg = _make_default_settings()
        config_toml = "[input_data]\n" + SRC_TRG_TOML
        result = build_settings_from_email(
            attachment_bodies={
                "config.toml": config_toml.encode("utf-8"),
                "source.txt": b"source",
                "ref_source.txt": b"Config ref source",
                "ref_target.txt": b"Config ref target",
            },
            email_body="",
            default_cfg=cfg,
        )
        assert result.input_data.ref_source_text == "Config ref source"
        assert result.input_data.ref_target_text == "Config ref target"
