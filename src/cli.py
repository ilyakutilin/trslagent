import sys
from pathlib import Path

from src.config import get_settings
from src.main import main, export_glossary_matches


USAGE = "Usage: python cli.py <path/to/config.toml> [--match-glossary --match-output <path>]"


def cli() -> None:
    args = sys.argv[1:]

    match_glossary = False
    match_output_path = None

    if "--match-glossary" in args:
        match_glossary = True
        args = [a for a in args if a != "--match-glossary"]

        if "--match-output" not in args:
            sys.exit(f"--match-glossary requires --match-output <path>\n{USAGE}")

        idx = args.index("--match-output")
        if idx + 1 >= len(args) or args[idx + 1].startswith("--"):
            sys.exit(f"--match-output requires a path argument\n{USAGE}")
        match_output_path = Path(args[idx + 1])
        args.pop(idx)       # --match-output
        args.pop(idx)       # the path value

    if len(args) != 1:
        sys.exit(USAGE)

    toml_path = Path(args[0]).resolve()
    if not toml_path.is_file():
        sys.exit(f"Config file not found: {toml_path}")

    settings = get_settings(toml_path=toml_path)

    if match_glossary:
        result = export_glossary_matches(cfg=settings)
        assert match_output_path is not None
        with open(match_output_path, "w", encoding="utf-8") as f:
            f.write(result)
        return

    translation = main(cfg=settings)
    if translation:
        with open(settings.output_data.result_file_path, "w", encoding="utf-8") as f:
            f.write(translation)


if __name__ == "__main__":
    cli()
