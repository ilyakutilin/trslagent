"""Language auto-detection for translation and review workflows.

Uses langdetect_ to infer the language of source and target texts,
defaulting the target language when only the source is available.

.. _langdetect: https://pypi.org/project/langdetect/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue

from langdetect import DetectorFactory
from langdetect import LangDetectException
from langdetect import detect

if TYPE_CHECKING:
    from src.config import Settings

DetectorFactory.seed = 0

_DEFAULT_MAX_CHARS = 200


def detect_language(text: str, max_chars: int = _DEFAULT_MAX_CHARS) -> Lang:
    """Detect the language of a text sample using langdetect.

    Args:
        text: The text whose language to detect.
        max_chars: Maximum number of characters from the start of *text*
            to pass to the detector.  Defaults to 200.

    Returns:
        An :class:`iso639.Lang` object for the detected language.

    Raises:
        ValueError: If the text is empty, too short for reliable detection,
            or *langdetect* returned a code that *iso639* does not recognise.
    """
    sample = text[:max_chars]
    try:
        code = detect(sample)
    except LangDetectException:
        raise ValueError(
            "Cannot detect language: text has no detectable features. "
            "Ensure the source text contains enough content."
        )
    try:
        lang = Lang(code)
    except InvalidLanguageValue as exc:
        raise ValueError(
            f"Cannot detect language: langdetect returned '{code}', "
            f"which iso639 does not recognise."
        ) from exc
    return lang


def resolve_languages(cfg: Settings) -> None:
    """Resolve source and target languages for the pipeline run.

    Fills in ``cfg.input_data.source_lang`` and
    ``cfg.input_data.target_lang`` when they are ``None``:

    * If *source_lang* is ``None``, it is detected from the source text
      via :func:`detect_language`.
    * If *target_lang* is ``None``:
        * In **review** mode (when *target_text* is present) it is
          detected from the target text.
        * In **translation** mode it defaults to ``ru`` when the source
          is not Russian, and to ``en`` when the source is Russian.

    Args:
        cfg: The application settings.  ``input_data.source_lang`` and
            ``input_data.target_lang`` are mutated in place when they
            were ``None``.

    Raises:
        ValueError: If the resolved source and target languages are
            identical, or if language detection fails.
    """
    from loguru import logger

    source_text = cfg.input_data.source_text or ""
    target_text = cfg.input_data.target_text

    is_review = target_text is not None
    resolved_source = cfg.input_data.source_lang
    resolved_target = cfg.input_data.target_lang

    if resolved_source is None:
        resolved_source = detect_language(source_text)
        logger.info(
            "Auto-detected source language: {} ({})",
            resolved_source.name,
            resolved_source.pt1,
        )

    if resolved_target is None:
        if is_review:
            resolved_target = detect_language(target_text)
            logger.info(
                "Auto-detected target language: {} ({})",
                resolved_target.name,
                resolved_target.pt1,
            )
        else:
            if resolved_source == Lang("ru"):
                resolved_target = Lang("en")
            else:
                resolved_target = Lang("ru")
            logger.info(
                "Defaulted target language: {} ({}) (source is {} in translation mode)",
                resolved_target.name,
                resolved_target.pt1,
                resolved_source.pt1,
            )

    if resolved_source == resolved_target:
        raise ValueError(
            "Source and target languages are the same: "
            f"{resolved_source.name} ({resolved_source.pt1}). "
            "Set different languages or check the source/target texts."
        )

    cfg.input_data.source_lang = resolved_source
    cfg.input_data.target_lang = resolved_target
