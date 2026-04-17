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
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    """Logging settings"""

    model_config = SettingsConfigDict(env_prefix="LOG_")

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

    model_config = SettingsConfigDict(env_prefix="LLM_")

    api_key: str = Field(default="", description="OpenRouter API key")
    model: str = Field(
        default="anthropic/claude-3.5-sonnet",
        description="Model to use for translation",
    )
    temperature: float = Field(
        default=0.1,
        description="Temperature for translation (lower = more consistent/literal)",
    )


class ChunkingSettings(BaseSettings):
    """Text chunking settings"""

    model_config = SettingsConfigDict(env_prefix="CHUNK_")

    size: int = Field(default=6000, description="Maximum chunk size in characters")
    overlap: int = Field(
        default=600, description="Overlap between chunks in characters"
    )


class GlossarySettings(BaseSettings):
    """Glossary/RAG settings"""

    model_config = SettingsConfigDict(env_prefix="GLOSSARY_")

    top_k: int = Field(
        default=20, description="Number of glossary terms to retrieve per chunk"
    )
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-large",
        description="Sentence transformer model for embeddings",
    )
    chroma_dir: str = Field(
        default="./chroma_db", description="Directory for ChromaDB persistence"
    )
    collection_name: str = Field(
        default="glossary", description="ChromaDB collection name"
    )


class Settings(BaseSettings):
    """Main settings container"""

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    glossary: GlossarySettings = Field(default_factory=GlossarySettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # Allows LLM__API_KEY in .env
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
        format=settings.logging.format,
        level=settings.logging.level,
        filter=lambda record: record["level"].name in ("DEBUG", "INFO"),
        colorize=True,
    )

    # WARNING and above to stderr
    logger.add(
        sys.stderr,
        format=settings.logging.format,
        level="WARNING",
        colorize=True,
    )


# Export logger for use in other modules
__all__ = ["logger", "get_settings", "setup_logging", "Settings"]
