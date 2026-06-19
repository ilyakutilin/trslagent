import pytest
from pydantic_settings import SettingsConfigDict

from src.config import EmailSettings, InputData, Settings, get_settings


class TestEmailSettingsDefaults:
    def test_resend_api_key_defaults_to_empty_secret(self):
        es = EmailSettings()
        assert es.resend_api_key.get_secret_value() == ""

    def test_resend_webhook_secret_defaults_to_empty_string(self):
        es = EmailSettings()
        assert es.resend_webhook_secret == ""

    def test_from_address_default(self):
        es = EmailSettings()
        assert es.from_address == "Translation Agent <trsl@resend.dev>"

    def test_allowed_senders_defaults_to_empty_list(self):
        es = EmailSettings()
        assert es.allowed_senders == []

    def test_sender_whitelist_enabled_defaults_to_true(self):
        es = EmailSettings()
        assert es.sender_whitelist_enabled is True

    def test_allowed_recipient_defaults_to_none(self):
        es = EmailSettings()
        assert es.allowed_recipient is None

    def test_listen_host_default(self):
        es = EmailSettings()
        assert es.listen_host == "0.0.0.0"

    def test_listen_port_default(self):
        es = EmailSettings()
        assert es.listen_port == 8025

    def test_max_attachment_size_mb_default(self):
        es = EmailSettings()
        assert es.max_attachment_size_mb == 10


def _setup_email_settings_test(tmp_path, monkeypatch, toml_extra="", env_vars=None):
    source_fp = tmp_path / "source.txt"
    source_fp.write_text("dummy")
    toml_path = tmp_path / "config.toml"
    toml_path.write_text(
        "[input_data]\n"
        'source_lang = "en"\n'
        'target_lang = "ru"\n'
        f'source_file_path = "{source_fp}"\n' + toml_extra
    )
    for k, v in (env_vars or {}).items():
        monkeypatch.setenv(k, v)
    return toml_path


_NO_ENV_FILE = SettingsConfigDict(
    env_file="/nonexistent/.env",
    env_file_encoding="utf-8",
    env_nested_delimiter="__",
    case_sensitive=False,
    env_ignore_empty=True,
)


class TestEmailSettingsInSettings:
    @pytest.fixture(autouse=True)
    def _clean_email_env(self, monkeypatch):
        for var in [
            "EMAIL__RESEND_API_KEY",
            "EMAIL__RESEND_WEBHOOK_SECRET",
            "EMAIL__FROM_ADDRESS",
            "EMAIL__ALLOWED_SENDERS",
            "EMAIL__ALLOWED_RECIPIENT",
            "EMAIL__LISTEN_HOST",
            "EMAIL__LISTEN_PORT",
            "EMAIL__SENDER_WHITELIST_ENABLED",
            "EMAIL__MAX_ATTACHMENT_SIZE_MB",
        ]:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(Settings, "model_config", _NO_ENV_FILE)

    def test_email_section_present_in_settings(self, tmp_path, en_lang, ru_lang):
        source_fp = tmp_path / "source.txt"
        source_fp.write_text("dummy")
        s = Settings(
            input_data=InputData(
                source_lang=en_lang,
                target_lang=ru_lang,
                source_file_path=source_fp,
            ),
        )
        assert isinstance(s.email, EmailSettings)

    def test_email_inherits_defaults_when_not_in_toml(self, tmp_path, monkeypatch):
        toml_path = _setup_email_settings_test(tmp_path, monkeypatch)
        s = get_settings(toml_path)
        assert s.email.resend_webhook_secret == ""
        assert s.email.from_address == "Translation Agent <trsl@resend.dev>"
        assert s.email.allowed_senders == []
        assert s.email.allowed_recipient is None
        assert s.email.sender_whitelist_enabled is True
        assert s.email.listen_host == "0.0.0.0"
        assert s.email.listen_port == 8025
        assert s.email.max_attachment_size_mb == 10
        assert s.email.resend_api_key.get_secret_value() == ""

    def test_email_overrides_from_toml(self, tmp_path, monkeypatch):
        toml_path = _setup_email_settings_test(
            tmp_path,
            monkeypatch,
            toml_extra=(
                "[email]\n"
                "listen_port = 9999\n"
                'from_address = "Agent <a@b.com>"\n'
                'allowed_senders = ["alice@x.com", "bob@x.com"]\n'
                "sender_whitelist_enabled = false\n"
                'allowed_recipient = "trsl@mydomain.com"\n'
                "max_attachment_size_mb = 20\n"
            ),
        )
        s = get_settings(toml_path)
        assert s.email.listen_port == 9999
        assert s.email.from_address == "Agent <a@b.com>"
        assert s.email.allowed_senders == ["alice@x.com", "bob@x.com"]
        assert s.email.sender_whitelist_enabled is False
        assert s.email.allowed_recipient == "trsl@mydomain.com"
        assert s.email.max_attachment_size_mb == 20

    def test_email_parsed_from_env(self, tmp_path, monkeypatch):
        toml_path = _setup_email_settings_test(
            tmp_path,
            monkeypatch,
            env_vars={"EMAIL__LISTEN_PORT": "7777"},
        )
        s = get_settings(toml_path)
        assert s.email.listen_port == 7777

    def test_toml_overrides_env_for_email(self, tmp_path, monkeypatch):
        toml_path = _setup_email_settings_test(
            tmp_path,
            monkeypatch,
            toml_extra=("[email]\nlisten_port = 9999\n"),
            env_vars={"EMAIL__LISTEN_PORT": "5555"},
        )
        s = get_settings(toml_path)
        assert s.email.listen_port == 9999

    def test_email_resend_api_key_from_env(self, tmp_path, monkeypatch):
        toml_path = _setup_email_settings_test(
            tmp_path,
            monkeypatch,
            env_vars={"EMAIL__RESEND_API_KEY": "re_test123"},
        )
        s = get_settings(toml_path)
        assert s.email.resend_api_key.get_secret_value() == "re_test123"

    def test_env_file_provides_email_defaults(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("EMAIL__LISTEN_PORT=5555\n")
        new_config = SettingsConfigDict(
            env_file=env_file,
            env_file_encoding="utf-8",
            env_nested_delimiter="__",
            case_sensitive=False,
            env_ignore_empty=True,
        )
        monkeypatch.setattr(Settings, "model_config", new_config)
        toml_path = _setup_email_settings_test(tmp_path, monkeypatch)
        s = get_settings(toml_path)
        assert s.email.listen_port == 5555

    def test_partial_override_toml_and_env(self, tmp_path, monkeypatch):
        toml_path = _setup_email_settings_test(
            tmp_path,
            monkeypatch,
            toml_extra=("[email]\nlisten_port = 9999\n"),
            env_vars={"EMAIL__FROM_ADDRESS": "env@example.com"},
        )
        s = get_settings(toml_path)
        assert s.email.listen_port == 9999
        assert s.email.from_address == "env@example.com"
