import argparse
import sys

from src.main import main
from src.models import CLIInput, InputData
from src.utils import (
    read_lines_from_file,
    read_str_from_file,
    validate_file,
    validate_lang,
)


def __input_or_none(inp: str) -> str | None:
    res = input(inp).strip()
    if not res:
        return None
    return res


def __input_or_default(inp: str, default: str) -> str:
    res = input(inp).strip()
    if not res:
        return default
    return res


def _get_cli_input_from_user(llm_required: bool) -> CLIInput:
    try:
        input_file_path = __input_or_default(
            "Path to the text file to translate (default: './files/source.md'): ",
            "./files/source.md",
        )
        validate_file(input_file_path)
        print(f"OK: {input_file_path}")
    except Exception as e:
        print(e)
        sys.exit(1)

    try:
        source_lang_raw = input("Source language (e.g. 'English' or 'en': ")
        source_lang = validate_lang(source_lang_raw)
        print(f"OK: {source_lang.name}")
    except Exception as e:
        print(e)
        sys.exit(1)

    try:
        target_lang_raw = input("Target language (e.g. 'Russian' or 'ru': ").strip()
        target_lang = validate_lang(target_lang_raw)
        print(f"OK: {target_lang.name}")
    except Exception as e:
        print(e)
        sys.exit(1)

    use_main_glossary = False
    if {source_lang.pt1, target_lang.pt1} == {"en", "ru"}:
        use_main_glossary = input(
            "Shall the main glossary be used? y/n (default = y): "
        ).strip().lower() in ("", "y", "yes")
    print(f"The main glossary {'WILL' if use_main_glossary else 'will NOT'} be used.")

    project_glossary_path = __input_or_none(
        "Is there a project glossary specific to this request? "
        "If yes, type the path to it. Otherwise leave empty: "
    )
    if project_glossary_path:
        try:
            validate_file(project_glossary_path)
            print(f"OK: {project_glossary_path}")
        except Exception as e:
            project_glossary_path = None
            proceed = input(
                f"Could not read project glossary file: {e}. "
                "Proceed without the project glossary? y/n (default y): "
            ).lower() in ("", "y", "yes")
            if not proceed:
                sys.exit(1)
    if not project_glossary_path:
        print("OK: No project glossary")

    llm_model = None
    if llm_required:
        llm_model = __input_or_none(
            "LLM model for translation "
            "(if empty, the model from the settings will be used): "
        )

    specialized_in = __input_or_none("The translator should be specialized in ...: ")
    doc_type = __input_or_none(
        "Type of the document (letter, contract, procedure etc.): "
    )
    doc_title = __input_or_none("Title of the document: ")

    try:
        output_file_path = __input_or_default(
            "Path to the translation result file (default: './files/result.md'): ",
            "./files/result.md",
        )
        print(f"OK: {output_file_path}")
    except Exception as e:
        print(e)
        sys.exit(1)

    return CLIInput(
        source_lang=source_lang,
        target_lang=target_lang,
        model=llm_model,
        specialized_in=specialized_in,
        doc_type=doc_type,
        doc_title=doc_title,
        use_main_glossary=use_main_glossary,
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        project_glossary_file_path=project_glossary_path,
    )


def _get_data_from_cli_input(cli_inp: CLIInput) -> InputData:
    cli_inp_dict = cli_inp.model_dump(exclude_defaults=True, exclude_none=True)
    cli_inp_dict["text"] = read_str_from_file(cli_inp.input_file_path)
    input_data = InputData.model_validate(cli_inp_dict, extra="ignore")
    if cli_inp.project_glossary_file_path:
        input_data.project_glossary_lines = read_lines_from_file(
            cli_inp.project_glossary_file_path
        )
    return input_data


def cli() -> None:
    parser = argparse.ArgumentParser(description="...")

    parser.add_argument(
        "--input-file",
        default=None,
        help=(
            "Path to the JSON input file instead of filling in the questions one by one"
        ),
    )

    parser.add_argument(
        "--print-prompt-only",
        action="store_true",
        default=False,
        help=(
            "Only print the prompts that would be sent to the LLM "
            "without actually calling it"
        ),
    )

    args = parser.parse_args()

    print_prompt_only = args.print_prompt_only

    if args.settings_file:
        try:
            json_str = read_str_from_file(args.settings_file)
            cli_inp = CLIInput.model_validate_json(json_str)
        except Exception:
            print("Failed to parse CLI input file, will proceed with questions")
            cli_inp = _get_cli_input_from_user(llm_required=not print_prompt_only)
    else:
        cli_inp = _get_cli_input_from_user(llm_required=not print_prompt_only)

    input_data = _get_data_from_cli_input(cli_inp)
    input_data.print_prompt_only = print_prompt_only

    translation = main(input_data)

    if translation:
        with open(cli_inp.output_file_path, "w", encoding="utf-8") as f:
            f.write(translation)


if __name__ == "__main__":
    cli()
