"""CLI entrypoint for the translation/review pipeline.

Supports translation, review, glossary matching, and email webhook server modes."""

import asyncio
import sys
from pathlib import Path

from src.config import get_settings
from src.main import main, export_glossary_matches


USAGE = (
    "Usage:\n"
    "  python cli.py <path/to/config.toml>                     # translate or review\n"
    "  python cli.py <path/to/config.toml> --match-glossary --match-output <path>\n"
    "  python cli.py serve-emails <path/to/config.toml>         # start email webhook server"
)


def _parse_glossary_args(args: list[str]) -> tuple[bool, Path | None, list[str]]:
    """Parse --match-glossary and --match-output flags from CLI args.

    Args:
        args: Command-line argument list (excluding the program name).

    Returns:
        A tuple of (match_glossary, match_output_path, remaining_args).
        Exits the process with an error message if --match-glossary is
        present but --match-output is missing or has no value.
    """
    match_glossary = False
    match_output_path: Path | None = None

    if "--match-glossary" in args:
        match_glossary = True
        args = [a for a in args if a != "--match-glossary"]

        if "--match-output" not in args:
            sys.exit(f"--match-glossary requires --match-output <path>\n{USAGE}")

        idx = args.index("--match-output")
        if idx + 1 >= len(args) or args[idx + 1].startswith("--"):
            sys.exit(f"--match-output requires a path argument\n{USAGE}")
        match_output_path = Path(args[idx + 1])
        args.pop(idx)
        args.pop(idx)

    return match_glossary, match_output_path, args


def cli() -> None:
    """Main CLI entrypoint.

    Dispatches to translation, review, glossary-matching, or email-server
    modes based on the provided arguments. Exits with usage info when no
    valid command is given.
    """
    args = sys.argv[1:]

    if not args:
        sys.exit(USAGE)

    if args[0] == "serve-emails":
        args.pop(0)
        if len(args) != 1:
            sys.exit(f"serve-emails requires a config.toml path\n{USAGE}")
        toml_path = Path(args[0]).resolve()
        if not toml_path.is_file():
            sys.exit(f"Config file not found: {toml_path}")

        from src.email_server import serve

        settings = get_settings(toml_path=toml_path)
        asyncio.run(serve(cfg=settings))
        return

    match_glossary, match_output_path, args = _parse_glossary_args(args)

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

    pipeline_result = asyncio.run(main(cfg=settings))
    if pipeline_result is not None:
        translation = pipeline_result.text
    else:
        translation = None
    if translation:
        result_path = settings.output_data.get_result_file_path()
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(translation)


if __name__ == "__main__":
    cli()
