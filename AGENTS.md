# AGENTS.md

## Entrypoint
- The **only** CLI is `python src/cli.py <config.toml>` — a single config-driven entrypoint.
- The README references `src/translator.py` with positional args; that script does **not** exist. The README is outdated.

## Environment & Setup
- Python 3.13 (see `.python-version`).
- Package manager: `uv`. Install deps with `uv sync`.
- Copy `.env.example` to `.env` and set `LLM__API_KEY` (OpenRouter key).
- **SpaCy model must be installed manually**: `python -m spacy download en_core_web_sm`. It is not in `pyproject.toml` dependencies.

## Config System (Non-Obvious)
- Settings priority: **TOML > env vars > `.env` > defaults**.
- The TOML path is threaded via the **class variable** `Settings._toml_path`, which must be set *before* `Settings()` is instantiated. See `src/config.py:330`.
- Env vars use double-underscore nesting: `LLM__MODEL`, `CHUNK__SIZE`, `GLOSSARY__XML_DIR`, etc.
- Language fields accept either ISO 639-1 codes (`"en"`, `"ru"`) or full names (`"English"`, `"Russian"`).

## Lemmatizer Constraints
- Only **English** (spaCy `en_core_web_sm`) and **Russian** (pymorphy3) are supported. All other languages return `None`.
- Known abbreviations (loaded from `files/abbrs`) are skipped during lemmatization — kept as-is in matched text.

## Glossary
- **Main glossary**: Multiterm/Multitran XML export with `<mtf>` root → `<conceptGrp>` → `<languageGrp>` → `<termGrp>`. Non-standard format, handled by `src/glossary/parser.py`.
- **Project glossary**: Plain text file with `term = translation` lines. Used per-document, overrides main glossary where terms conflict.
- **Caching**: Parsed XML entries are cached as `.pickle` files alongside the XML source files. Cache validity is based on SHA-256 hash of the XML file. Stale/missing caches trigger automatic re-parse and re-lemmatization. Non-matching orphan pickle files are cleaned up automatically.
- **Abbreviation extraction**: Run `src/glossary/get_abbrs.py` (standalone) to scan the glossary for abbreviations and write `files/abbrs`. Used by the lemmatizer to preserve abbreviations.

## LLM
- Uses OpenRouter API. Default base URL: `https://openrouter.ai/api/v1`.
- 5 retries with exponential backoff for timeout/rate-limit/connection errors. Other errors raise immediately.
- `--print-prompt-only` / `print_prompt_only: true` skips LLM calls and prints the constructed prompts to stdout.

## Review Mode
- **Not implemented**. `src/main.py:39` raises `NotImplementedError` when `target_text` is set.

## Testing / Linting / CI
- No tests, no linting config, no typecheck config, no CI workflows. None exist in this repo.
