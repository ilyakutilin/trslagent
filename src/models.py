from typing import Annotated, Any

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer


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


class BaseInput(BaseModel):
    source_lang: LangField = Field(default=..., description="Source language")
    target_lang: LangField = Field(default=..., description="Target language")
    model: str | None = Field(
        default=None, description="LLM model used for translation"
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
    use_main_glossary: bool = Field(
        default=True, description="Whether to use the main glossary during translation"
    )
    print_prompt_only: bool = Field(
        default=False,
        description=(
            "Debug flag to not execute translation and only print the prompts "
            "that would be fed to LLM"
        ),
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CLIInput(BaseInput):
    input_file_path: str = Field(
        default="files/source.md",
        description="Path to a file containning the source text for translation",
    )
    output_file_path: str = Field(
        default="files/result.md", description="Path to a translated result file"
    )
    project_glossary_file_path: str | None = Field(
        default=None,
        description=(
            "Path to a file containing the glossary specific to this translation"
        ),
    )


class InputData(BaseInput):
    text: str = Field(default=..., description="Text for translation", min_length=1)
    project_glossary_lines: list[str] | None = Field(
        default=None, description="String lines read from the project glossary file"
    )
