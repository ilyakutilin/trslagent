from src.config import get_settings
from src.glossary.parser import MainGlossaryParser, ProjectGlossaryParser
from src.lemmatizer import Lemmatizer
from src.llm import LLM
from src.models import InputData
from src.translator import Translator


def main(inp: InputData) -> str | None:
    settings = get_settings()
    lemmatizer = Lemmatizer()

    # Get all glossary entries
    main_glossary_entries = []
    if inp.use_main_glossary:
        main_glossary_entries = MainGlossaryParser(
            dir_path=settings.glossary.xml_dir,
            lemmatizer=lemmatizer,
        ).parse()

    project_glossary_entries = []
    if inp.project_glossary_lines:
        project_glossary_entries = ProjectGlossaryParser(
            project_glossary_lines=inp.project_glossary_lines,
            source_lang=inp.source_lang,
            target_lang=inp.target_lang,
            lemmatizer=lemmatizer,
        ).parse()

    llm = None
    if not inp.print_prompt_only:
        llm = LLM(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            model=inp.model or settings.llm.model,
            temperature=settings.llm.temperature,
            reasoning_effort=settings.llm.reasoning_effort,
        )

    translator = Translator(
        source_lang=inp.source_lang,
        target_lang=inp.target_lang,
        specialized_in=inp.specialized_in,
        text=inp.text,
        doc_type=inp.doc_type,
        doc_title=inp.doc_title,
        llm=llm,
        lemmatizer=lemmatizer,
        main_glossary_entries=main_glossary_entries,
        project_glossary_entries=project_glossary_entries,
    )

    return translator.translate_document()
