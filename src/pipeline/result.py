"""Pipeline result data class and constructor helper."""

from dataclasses import dataclass

from iso639 import Lang

from src.config import InputData


@dataclass
class PipelineResult:
    """Holds the result of a translation or review pipeline run.

    Attributes:
        text: The final output text (translation or review feedback).
        source_lang: Source language of the original text.
        target_lang: Target language of the output.
        source_chars: Character count of source text.
        target_chars: Character count of output text.
        chunk_count: Number of chunks the text was split into.
        model: LLM model identifier used.
        cost_total: Total cost of the run, or None if unavailable.
        cost_currency: Currency code for the cost (e.g. "USD").
        cost_unknowns: Number of chunks whose cost could not be resolved.
        auto_glossary_entries_matched: Total auto-glossary entries matched across chunks.
        user_glossary_entries: Number of user-supplied glossary entries.
        specialized_in: Optional domain specialization.
        doc_type: Optional document type.
        doc_title: Optional document title.
        additional_instructions: Optional extra instructions for the LLM.
        auto_glossary_enabled: Whether auto glossary matching was active.
        user_glossary_enabled: Whether a user glossary was provided.
        mode: Pipeline mode — "translation" or "review".
    """

    text: str
    source_lang: Lang
    target_lang: Lang
    source_chars: int
    target_chars: int
    chunk_count: int
    model: str
    cost_total: float | None
    cost_currency: str
    cost_unknowns: int
    auto_glossary_entries_matched: int
    user_glossary_entries: int
    specialized_in: str | None
    doc_type: str | None
    doc_title: str | None
    additional_instructions: str | None
    auto_glossary_enabled: bool
    user_glossary_enabled: bool
    mode: str


def build_pipeline_result(
    *,
    text: str,
    source_lang: Lang,
    target_lang: Lang,
    source_chars: int,
    target_chars: int,
    chunk_count: int,
    model: str,
    cost_total: float | None,
    cost_currency: str,
    cost_unknowns: int,
    auto_glossary_entries_matched: int,
    user_glossary_entries: int,
    auto_glossary_enabled: bool,
    user_glossary_enabled: bool,
    mode: str,
    input_data: InputData,
) -> PipelineResult:
    """Build a PipelineResult, sourcing context fields from InputData.

    Centralizes the 22-arg PipelineResult construction that previously appeared
    three times in src/main.py.

    Args:
        text: The final output text.
        source_lang: Source language of the original text.
        target_lang: Target language of the output.
        source_chars: Character count of source text.
        target_chars: Character count of output text.
        chunk_count: Number of chunks the text was split into.
        model: LLM model identifier used.
        cost_total: Total cost of the run, or None if unavailable.
        cost_currency: Currency code for the cost.
        cost_unknowns: Number of chunks whose cost could not be resolved.
        auto_glossary_entries_matched: Total auto-glossary entries matched.
        user_glossary_entries: Number of user-supplied glossary entries.
        auto_glossary_enabled: Whether auto glossary matching was active.
        user_glossary_enabled: Whether a user glossary was provided.
        mode: Pipeline mode — "translation" or "review".
        input_data: InputData with specialized_in, doc_type, doc_title,
            additional_instructions.

    Returns:
        A fully populated PipelineResult.
    """
    return PipelineResult(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        source_chars=source_chars,
        target_chars=target_chars,
        chunk_count=chunk_count,
        model=model,
        cost_total=cost_total,
        cost_currency=cost_currency,
        cost_unknowns=cost_unknowns,
        auto_glossary_entries_matched=auto_glossary_entries_matched,
        user_glossary_entries=user_glossary_entries,
        specialized_in=input_data.specialized_in,
        doc_type=input_data.doc_type,
        doc_title=input_data.doc_title,
        additional_instructions=input_data.additional_instructions,
        auto_glossary_enabled=auto_glossary_enabled,
        user_glossary_enabled=user_glossary_enabled,
        mode=mode,
    )


__all__ = ["PipelineResult", "build_pipeline_result"]
