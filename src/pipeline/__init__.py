"""Pipeline orchestration: translation, review, and shared helpers.

Public API:
    PipelineResult, build_pipeline_result: result data class and constructor.
    GlossaryContext, prepare_glossary_context: glossary prep.
    gather_chunk_results: concurrency helper.
    run_translation_pipeline, run_review_pipeline: mode-specific pipelines.
"""

from src.pipeline.concurrency import gather_chunk_results
from src.pipeline.glossary_context import GlossaryContext, prepare_glossary_context
from src.pipeline.result import PipelineResult, build_pipeline_result
from src.pipeline.review import run_review_pipeline
from src.pipeline.translation import run_translation_pipeline

__all__ = [
    "PipelineResult",
    "build_pipeline_result",
    "GlossaryContext",
    "prepare_glossary_context",
    "gather_chunk_results",
    "run_translation_pipeline",
    "run_review_pipeline",
]
