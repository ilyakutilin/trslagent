import argparse
import sys
from pathlib import Path

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue

from src.main import main


def validate_file(file_path: str) -> Path:
    try:
        file = Path(file_path)
        # Check if it is a regular file and readable
        with open(file, "r", encoding="utf-8") as f:
            f.read(1024)  # Attempt to read the first 1KB
        return file
    except (IOError, OSError, UnicodeDecodeError) as e:
        # IOError/OSError: File not found or no read permission
        # UnicodeDecodeError: File is binary or incompatible encoding
        raise e


def validate_lang(raw_lang: str) -> Lang:
    try:
        return Lang(raw_lang)
    except InvalidLanguageValue as e:
        raise e


def input_or_none(inp: str) -> str | None:
    res = input(inp).strip()
    if not res:
        return None
    return res


def input_or_default(inp: str, default: str) -> str:
    res = input(inp).strip()
    if not res:
        return default
    return res


if __name__ == "__main__":
    import argparse

    from src.main import main

    parser = argparse.ArgumentParser(description="...")

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

    try:
        input_file = validate_file(
            input_or_default(
                "Path to the text file to translate (default: './files/source.md'): ",
                "./files/source.md",
            )
        )
        print(f"OK: {input_file}")
    except Exception as e:
        print(e)
        sys.exit(1)

    try:
        source_lang = validate_lang(input("Source language (e.g. 'English' or 'en': "))
        print(f"OK: {source_lang.name}")
    except Exception as e:
        print(e)
        sys.exit(1)

    try:
        target_lang = validate_lang(
            input("Target language (e.g. 'Russian' or 'ru': ").strip()
        )
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

    project_glossary_path_raw = input_or_none(
        "Is there a project glossary specific to this request? "
        "If yes, type the path to it. Otherwise leave empty: "
    )
    if project_glossary_path_raw:
        try:
            project_glossary_path = validate_file(project_glossary_path_raw)
            print(f"OK: {project_glossary_path}")
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        print("OK: No project glossary")

    llm_model = None
    if not args.print_prompt_only:
        llm_model = input_or_none(
            "LLM model for translation "
            "(if empty, the model from the settings will be used): "
        )

    specialized_in = input_or_none("The translator should be specialized in ...: ")
    doc_type = input_or_none(
        "Type of the document (letter, contract, procedure etc.): "
    )
    doc_title = input_or_none("Title of the document: ")

    try:
        output_file = validate_file(
            input_or_default(
                "Path to the translation result file (default: './files/result.md'): ",
                "./files/result.md",
            )
        )
        print(f"OK: {output_file}")
    except Exception as e:
        print(e)
        sys.exit(1)

    main(
        input_file_path=input_file,
        source_lang=source_lang,
        target_lang=target_lang,
        output_file_path=output_file,
        print_prompt_only=args.print_prompt_only,
        model=llm_model,
        specialized_in=specialized_in,
        doc_type=doc_type,
        doc_title=doc_title,
        use_main_glossary=use_main_glossary,
        project_glossary_file_path=project_glossary_path,
    )
