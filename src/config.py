"""
Centralized Configuration
=========================
Uses pydantic-settings for type-safe, environment-based configuration.
Settings are organized into logical classes and can be overridden
via environment variables or .env file.
"""

import sys
from typing import Literal

from loguru import logger
from openai.types import ReasoningEffort
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
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


class LLMSettings(BaseSettings):
    """LLM/OpenRouter API settings"""

    base_url: str = Field(
        default="https://openrouter.ai/api/v1", description="LLM Base URL"
    )
    api_key: str = Field(default="", description="LLM API key")
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


class ChunkingSettings(BaseSettings):
    """Text chunking settings"""

    size: int = Field(default=6000, description="Maximum chunk size in characters")
    overlap: int = Field(
        default=600, description="Overlap between chunks in characters"
    )


class GlossarySettings(BaseSettings):
    """Glossary/RAG settings"""

    xml_dir: str = Field(
        default="./files/glossary",
        description="Directory with glossary XML files exported by Multiterm",
    )
    top_k: int = Field(
        default=20, description="Number of glossary terms to retrieve per chunk"
    )
    known_abbrs_file_path: str | None = Field(
        default=None, description="Path to a file with a list of known abbreviations"
    )


class GDocsSettings(BaseSettings):
    """Google Docs interface configuration"""

    credentials_path: str = Field(
        default="credentials.json",
        description="Path to the Google service account JSON credentials file",
    )

    document_id: str = Field(
        default="",
        description="ID of the Google Doc used as input (from the doc URL)",
    )

    poll_interval_seconds: int = Field(
        default=15,
        description="How often (seconds) to check the document for new requests",
    )

    ready_trigger: str = Field(
        default="Ready",
        description="Value of the Status field that triggers processing",
    )

    done_marker: str = Field(
        default="Done",
        description="Value written to Status after successful processing",
    )

    error_marker: str = Field(
        default="Error",
        description="Value written to Status if processing fails",
    )


class Settings(BaseSettings):
    """Main settings container"""

    log: LoggingSettings = Field(default_factory=LoggingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    chunk: ChunkingSettings = Field(default_factory=ChunkingSettings)
    glossary: GlossarySettings = Field(default_factory=GlossarySettings)
    gdocs: GDocsSettings = Field(default_factory=GDocsSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )


# Global settings instance
_settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return _settings


def setup_logging() -> None:
    """Configure loguru logger with level-based output routing.

    INFO and DEBUG messages go to stdout.
    WARNING and above go to stderr.
    """
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # INFO and DEBUG to stdout
    logger.add(
        sys.stdout,
        format=settings.log.format,
        level=settings.log.level,
        filter=lambda record: record["level"].name in ("DEBUG", "INFO"),
        colorize=True,
    )

    # WARNING and above to stderr
    logger.add(
        sys.stderr,
        format=settings.log.format,
        level="WARNING",
        colorize=True,
    )


# Export logger for use in other modules
__all__ = ["logger", "get_settings", "setup_logging", "Settings"]
