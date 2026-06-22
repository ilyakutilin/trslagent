"""
Centralized Configuration
=========================
Uses pydantic-settings for type-safe, environment-based configuration.
Settings are organized into logical classes and can be overridden
via environment variables or .env file.
"""

import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self, Type

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue
from loguru import logger
from openai.types import ReasoningEffort
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    SecretStr,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from src.utils import read_lines_from_file, read_str_from_file


class LLMSettings(BaseSettings):
    """LLM/OpenRouter API settings"""

    base_url: str = Field(
        default="https://openrouter.ai/api/v1", description="LLM Base URL"
    )
    api_key: SecretStr = Field(default=SecretStr(""), description="LLM API key")
    model: str = Field(
        default="anthropic/claude-3.5-sonnet",
        description="Model to use for translation",
    )
    temperature: float | None = Field(
        default=None,
        description="Temperature for translation (lower = more consistent/literal)",
    )
    reasoning_effort: ReasoningEffort = Field(
        default=None,
        description=(
            "Reasoning effort for the thinking models. "
            "If not provided, reasoning is not used."
        ),
    )


class ChunkSettings(BaseSettings):
    """Text chunking settings"""

    size: int = Field(default=6000, description="Maximum chunk size in characters")
    divider: str | None = Field(
        default=None,
        description=(
            "Character for manual chunk splitting. When set, the text is split on "
            "lines consisting of this character repeated 10+ times (e.g. '----------'). "
            "Overrides size-based chunking. Applied to both source and target in review mode."
        ),
    )
    max_concurrent: int = Field(default=3, description="Max simultaneous LLM API calls")
    delay_seconds: float = Field(
        default=1.5, description="Seconds between launching chunk tasks"
    )


class GlossarySettings(BaseSettings):
    """Glossary/RAG settings"""

    xml_dir_path: Path = Field(
        default=Path("files/glossary"),
        description="Directory with glossary XML files exported by Multiterm",
    )
    known_abbrs_file_path: Path | None = Field(
        default=None, description="Path to a file with a list of known abbreviations"
    )


class CostSettings(BaseSettings):
    """Cost tracking settings"""

    generation_info_url: str | None = Field(
        default=None,
        description="URL for retrieving generation cost info via completion ID",
    )
    cost_key: str = Field(
        default="total_cost",
        description="Key to look for in the generation info API response JSON",
    )
    cost_currency: str = Field(
        default="USD",
        description="Currency for displaying cost totals",
    )


class LogSettings(BaseModel):
    """Logging settings"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    format: str = Field(
        default=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> |"
            " <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        description="Log format string",
    )


class EmailSettings(BaseModel):
    """Email/webhook settings for receiving translation requests via Resend"""

    resend_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Resend API key for sending and receiving emails",
    )
    resend_webhook_secret: str = Field(
        default="", description="Resend webhook signing secret (whsec_...)"
    )
    from_address: str = Field(
        default="Translation Agent <trsl@resend.dev>",
        description="From address used for reply emails",
    )
    allowed_senders: list[str] = Field(
        default_factory=list,
        description="Whitelist of sender email addresses allowed to submit translation requests",
    )
    sender_whitelist_enabled: bool = Field(
        default=True,
        description="When true, only senders in allowed_senders may submit requests",
    )
    allowed_recipient: str | None = Field(
        default=None,
        description="If set, only webhooks for emails sent to this exact address are processed; others are logged and silently ignored",
    )
    listen_host: str = Field(
        default="0.0.0.0", description="Host for the webhook HTTP server"
    )
    listen_port: int = Field(
        default=8025, description="Port for the webhook HTTP server"
    )
    max_attachment_size_mb: int = Field(
        default=10, description="Maximum individual attachment size in MB"
    )


def parse_lang(v: Any) -> Lang:
    if isinstance(v, str):
        try:
            return Lang(v)
        except InvalidLanguageValue as e:
            raise ValueError(e)
    if isinstance(v, Lang):
        return v
    raise ValueError(
        f"Object {v} is of a wrong class {v.__class__.__name__} "
        "while only str and Lang are supported"
    )


LangField = Annotated[
    Lang,
    BeforeValidator(parse_lang),
    PlainSerializer(lambda x: x.pt1, return_type=str, when_used="always"),
]


class InputData(BaseModel):
    source_lang: LangField | None = Field(
        default=None,
        description=(
            "Source language. ISO 639-1 code or full name. "
            "If not set, detected automatically from the source text "
            "via langdetect."
        ),
    )
    target_lang: LangField | None = Field(
        default=None,
        description=(
            "Target language. ISO 639-1 code or full name. "
            "If not set in translation mode, defaults to 'ru' "
            "(or 'en' if source is Russian). "
            "In review mode, detected automatically from the target text."
        ),
    )
    specialized_in: str | None = Field(
        default=None,
        description="Specialization of the LLM translator. Fed into the system prompt",
    )
    doc_type: str | None = Field(
        default=None,
        description=(
            "Document type (letter, procedure, presentation etc.). "
            "Fed into the system prompt"
        ),
    )
    doc_title: str | None = Field(
        default=None, description="Document title. Fed into the system prompt"
    )
    source_file_path: Path = Field(
        default=Path("files/source.txt"),
        description=(
            "Path to a file containning the source text for translation or review"
        ),
    )
    target_file_path: Path | None = Field(
        default=None,
        description=(
            "Path to a file containning the target text for review. "
            "If None, the mode is 'translation'. If not None, the mode is 'review'."
        ),
    )
    source_text: str | None = Field(
        default=None, description="Source text", min_length=1
    )
    target_text: str | None = Field(
        default=None, description="Target text (for review mode)", min_length=1
    )
    auto_glossary: bool = Field(
        default=False,
        description=(
            "Whether to automatically extract glossary terms from the auto glossary"
        ),
    )
    user_glossary_file_path: str | None = Field(
        default=None,
        description=("Path to a file containing the user-supplied glossary"),
    )
    user_glossary_lines: list[str] | None = Field(
        default=None, description="String lines read from the user glossary file"
    )

    @model_validator(mode="after")
    def validate_input_data(self) -> Self:
        if (
            self.source_lang is not None
            and self.target_lang is not None
            and self.source_lang == self.target_lang
        ):
            raise ValueError("Please set the source and target langs correctly")

        if self.source_text is None:
            try:
                self.source_text = read_str_from_file(fp=self.source_file_path)
            except (IOError, OSError, UnicodeDecodeError) as e:
                logger.warning("Failed to read {}: {}", self.source_file_path, e)
                self.source_text = None

        if self.target_text is None and self.target_file_path is not None:
            try:
                self.target_text = read_str_from_file(fp=self.target_file_path)
            except (IOError, OSError, UnicodeDecodeError) as e:
                logger.warning("Failed to read {}: {}", self.target_file_path, e)
                self.target_text = None

        if (
            self.user_glossary_lines is None
            and self.user_glossary_file_path is not None
        ):
            try:
                self.user_glossary_lines = read_lines_from_file(
                    fp=self.user_glossary_file_path
                )
            except (IOError, OSError, UnicodeDecodeError) as e:
                logger.warning(
                    "Failed to read {}: {}",
                    self.user_glossary_file_path,
                    e,
                )
                self.user_glossary_lines = None

        return self

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OutputData(BaseModel):
    result_file_path: Path = Field(
        default=Path("files/result.md"), description="Path to a result file"
    )
    raw_result_file_path: Path = Field(
        default=Path("files/raw_result.json"),
        description="Path to a raw JSON result file",
    )
    timestamped_result_filenames: bool = Field(
        default=True,
        description="Whether a timestamp should be appended to the result file name",
    )
    print_prompt_only: bool = Field(
        default=False,
        description=(
            "Debug flag to not execute translation and only print the prompts "
            "that would be fed to LLM"
        ),
    )

    def get_result_file_path(self) -> Path:
        if not self.timestamped_result_filenames:
            return self.result_file_path
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return self.result_file_path.with_stem(f"{self.result_file_path.stem}_{ts}")


class TomlConfigSource(PydanticBaseSettingsSource):
    """
    Reads settings from a TOML file.

    Only keys that are *actually present* in the TOML are emitted, so any
    missing key naturally falls through to the next lower-priority source
    (env vars, .env file, or Pydantic defaults).
    """

    def __init__(self, settings_cls: Type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._data: dict[str, Any] = {}
        if path.is_file():
            with open(path, "rb") as fh:
                self._data = tomllib.load(fh)

    # Required by PydanticBaseSettingsSource

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        val = self._data.get(field_name)
        return val, field_name, self.field_is_complex(field)

    def __call__(self) -> dict[str, Any]:
        return {
            name: self._data[name]
            for name in self.settings_cls.model_fields
            if name in self._data
        }


_PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    cost: CostSettings = Field(default_factory=CostSettings)
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    glossary: GlossarySettings = Field(default_factory=GlossarySettings)
    log: LogSettings = Field(default_factory=LogSettings)
    email: EmailSettings = Field(default_factory=EmailSettings)
    input_data: InputData = Field(default_factory=InputData)
    output_data: OutputData = Field(default_factory=OutputData)

    # Class-level slot for the CLI-supplied TOML path
    # Set this *before* instantiating Settings:
    #
    #      Settings._toml_path = Path("config.toml")
    #      cfg = Settings()
    #
    _toml_path: ClassVar[Path | None] = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Returns the ordered tuple of sources.

        pydantic-settings v2 processes this tuple in *reverse* order using
        deep_update, so the first entry listed here wins. Sub-model dicts
        are merged field-by-field, not replaced wholesale – meaning TOML can
        override just `llm.model` while the env still provides `llm.api_key`.
        """
        sources: list[PydanticBaseSettingsSource] = [init_settings]
        if cls._toml_path is not None:
            sources.append(TomlConfigSource(settings_cls, cls._toml_path))
        sources += [env_settings, dotenv_settings, file_secret_settings]
        return tuple(sources)


def setup_logging(log_settings: LogSettings) -> None:
    """Configure loguru logger with level-based output routing.

    INFO and DEBUG messages go to stdout.
    WARNING and above go to stderr.
    """
    # Remove default handler
    logger.remove()

    # INFO and DEBUG to stdout
    logger.add(
        sys.stdout,
        format=log_settings.format,
        level=log_settings.level,
        filter=lambda record: record["level"].name in ("DEBUG", "INFO"),
        colorize=True,
    )

    # WARNING and above to stderr
    logger.add(
        sys.stderr,
        format=log_settings.format,
        level="WARNING",
        colorize=True,
    )


def get_settings(toml_path: Path) -> Settings:
    """Get the global settings instance."""
    # Thread the TOML path into the settings class BEFORE instantiation.
    # settings_customise_sources() is a class method that runs during model
    # construction, so the class variable must be set first.
    Settings._toml_path = toml_path

    settings = Settings()

    setup_logging(log_settings=settings.log)

    return settings


# Export logger for use in other modules
__all__ = ["logger", "get_settings", "setup_logging", "Settings"]
