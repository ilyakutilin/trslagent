from pathlib import Path

from iso639 import Lang

from src.config import get_settings
from src.glossary.parser import MainGlossaryParser, ProjectGlossaryParser
from src.lemmatizer import Lemmatizer
from src.llm import LLM
from src.translator import Translator


def main(
    input_file_path: Path,
    source_lang: Lang,
    target_lang: Lang,
    output_file_path: Path,
    print_prompt_only: bool,
    model: str | None,
    specialized_in: str | None,
    doc_type: str | None,
    doc_title: str | None,
    use_main_glossary: bool,
    project_glossary_file_path: Path | None = None,
):
    settings = get_settings()
    lemmatizer = Lemmatizer()

    # Read input file
    with open(input_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Get all glossary entries
    main_glossary_entries = []
    if use_main_glossary:
        main_glossary_entries = MainGlossaryParser(
            dir_path=settings.glossary.xml_dir,
            lemmatizer=lemmatizer,
        ).parse()

    project_glossary_entries = []
    if project_glossary_file_path:
        project_glossary_entries = ProjectGlossaryParser(
            project_glossary_file_path=project_glossary_file_path,
            source_lang=source_lang,
            target_lang=target_lang,
            lemmatizer=lemmatizer,
        ).parse()

    llm = None
    if not print_prompt_only:
        llm = LLM(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            model=model or settings.llm.model,
            temperature=settings.llm.temperature,
            reasoning_effort=settings.llm.reasoning_effort,
        )

    translator = Translator(
        source_lang=source_lang,
        target_lang=target_lang,
        specialized_in=specialized_in,
        text=text,
        doc_type=doc_type,
        doc_title=doc_title,
        llm=llm,
        lemmatizer=lemmatizer,
        main_glossary_entries=main_glossary_entries,
        project_glossary_entries=project_glossary_entries,
    )

    translation = translator.translate_document()

    if translation is None:
        return

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(translation)
