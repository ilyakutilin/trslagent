from src.config import Settings
from src.glossary.parser import MainGlossaryParser, ProjectGlossaryParser
from src.lemmatizer import Lemmatizer
from src.llm import LLM
from src.translator import Translator


def main(cfg: Settings) -> str | None:
    lemmatizer = Lemmatizer()

    # Get all glossary entries
    main_glossary_entries = []
    if cfg.input_data.auto_glossary:
        main_glossary_entries = MainGlossaryParser(
            dir_path=cfg.glossary.xml_dir_path,
            lemmatizer=lemmatizer,
        ).parse()

    project_glossary_entries = []
    if cfg.input_data.glossary_lines:
        project_glossary_entries = ProjectGlossaryParser(
            project_glossary_lines=cfg.input_data.glossary_lines,
            source_lang=cfg.input_data.source_lang,
            target_lang=cfg.input_data.target_lang,
            lemmatizer=lemmatizer,
        ).parse()

    llm = None
    if not cfg.output_data.print_prompt_only:
        llm = LLM(
            base_url=cfg.llm.base_url,
            api_key=cfg.llm.api_key.get_secret_value(),
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            reasoning_effort=cfg.llm.reasoning_effort,
        )

    if cfg.input_data.target_text:
        # TODO: Implement review functionality
        raise NotImplementedError("Review functionality is not yet implemented")
    else:
        translator = Translator(
            source_lang=cfg.input_data.source_lang,
            target_lang=cfg.input_data.target_lang,
            specialized_in=cfg.input_data.specialized_in,
            text=cfg.input_data.source_text or "",
            doc_type=cfg.input_data.doc_type,
            doc_title=cfg.input_data.doc_title,
            llm=llm,
            lemmatizer=lemmatizer,
            main_glossary_entries=main_glossary_entries,
            project_glossary_entries=project_glossary_entries,
        )

        return translator.translate_document()
