"""Glossary package for parsing, caching, lemmatizing, and matching glossary entries."""

from src.glossary.dedup import deduplicate_user_auto
from src.glossary.stringify import stringify_entries

__all__ = ["deduplicate_user_auto", "stringify_entries"]
