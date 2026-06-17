import sys
from pathlib import Path

from src.config import get_settings
from src.main import main


def cli() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python cli.py <path/to/config.toml>")

    toml_path = Path(sys.argv[1]).resolve()
    if not toml_path.is_file():
        sys.exit(f"Config file not found: {toml_path}")

    settings = get_settings(toml_path=toml_path)
    translation = main(cfg=settings)

    if translation:
        with open(settings.output_data.result_file_path, "w", encoding="utf-8") as f:
            f.write(translation)


if __name__ == "__main__":
    cli()
