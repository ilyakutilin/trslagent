# AGENTS.md

## Entrypoint

- The **only** CLI is `python src/cli.py <config.toml>` — a single config-driven entrypoint.
- Additional subcommands:
  - `python src/cli.py <config.toml> --match-glossary --match-output <path>` — export glossary matches.
  - `python src/cli.py serve-emails <config.toml>` — start the email webhook server.

## Environment & Setup

- Python 3.13 (see `.python-version`).
- Package manager: `uv`. Install deps with `uv sync`.
- Copy `.env.example` to `.env` and set `LLM__API_KEY` (OpenRouter key).
- **SpaCy model must be installed manually**: `python -m spacy download en_core_web_sm`. It is not in `pyproject.toml` dependencies.

## Config System (Non-Obvious)

- Settings priority: **TOML > env vars > `.env` > defaults**.
- The TOML path is threaded via the **class variable** `Settings._toml_path`, which must be set _before_ `Settings()` is instantiated. See `src/config.py:330`.
- Env vars use double-underscore nesting: `LLM__MODEL`, `CHUNK__SIZE`, `GLOSSARY__XML_DIR`, etc.
- Language fields accept either ISO 639-1 codes (`"en"`, `"ru"`) or full names (`"English"`, `"Russian"`).
- Both `source_lang` and `target_lang` are **optional**. When not set, the source language is auto-detected from the source text via `langdetect` (first ~200 chars). In translation mode, the target language defaults to `ru` (or `en` if the source is Russian). In review mode, both languages are auto-detected from their respective texts. Detection logic lives in `src/language_detection.py` and is invoked at the start of `main()`.

## Lemmatizer Constraints

- Only **English** (spaCy `en_core_web_sm`) and **Russian** (pymorphy3) are supported. All other languages return `None`.
- Known abbreviations (loaded from `files/abbrs`) are skipped during lemmatization — kept as-is in matched text.

## Glossary

- **Auto glossary**: Multiterm/Multitran XML export with `<mtf>` root → `<conceptGrp>` → `<languageGrp>` → `<termGrp>`. Non-standard format, handled by `src/glossary/parser.py`.
- **User glossary**: Plain text file with `term = translation` lines. Used per-document, overrides auto glossary where terms conflict.
- **Caching**: Parsed XML entries are cached as `.pickle` files alongside the XML source files. Cache validity is based on SHA-256 hash of the XML file. Stale/missing caches trigger automatic re-parse and re-lemmatization. Non-matching orphan pickle files are cleaned up automatically.
- **Abbreviation extraction**: Run `src/glossary/get_abbrs.py` (standalone) to scan the glossary for abbreviations and write `files/abbrs`. Used by the lemmatizer to preserve abbreviations.

## LLM

- Uses OpenRouter API. Default base URL: `https://openrouter.ai/api/v1`.
- 5 retries with exponential backoff for timeout/rate-limit/connection errors. Other errors raise immediately.
- `--print-prompt-only` / `print_prompt_only: true` skips LLM calls and prints the constructed prompts to stdout.

## Review Mode

- Enabled by setting `target_file_path` (or `target_text`) in the config.
- Without `chunk.divider`: sends the full source + target text to the LLM for proofreading (no chunking).
- With `chunk.divider` set: splits both source and target on the divider character (repeated 10+ times). The number of source and target chunks must be equal — otherwise a `ValueError` is raised. Each source/target chunk pair is reviewed concurrently (same semaphore/gather/delay as translation).
- Returns a list of critical mistakes: missing translations, distortions, spelling, numbers, dictionary deviations.

## Email Server (Webhook)

- Started via `python src/cli.py serve-emails <config.toml>`.
- Runs an `aiohttp` HTTP server on `email.listen_host`:`email.listen_port` (default `0.0.0.0:8025`).
- Listens on `POST /webhook/email` for Resend inbound email webhooks.
- Uses `src/email_server.py` (server) + `src/email_processor.py` (Resend API client).
- **Svix signature verification**: webhook payloads are verified via HMAC-SHA256 against `email.resend_webhook_secret` (`whsec_...`), with a 5-minute timestamp tolerance window. If no secret is configured, verification is skipped (logged warning).
- **Sender whitelist**: if `email.sender_whitelist_enabled` is true (default), only senders in `email.allowed_senders` are accepted. An empty list disables enforcement.
- **Recipient filtering**: if `email.allowed_recipient` is set, only webhooks for that specific To address are processed; others are silently ignored.
- On receipt: fetches the email body and attachment list from the Resend Receiving API, downloads each attachment (cap: `email.max_attachment_size_mb`, default 10 MB per file).
- **Configuration from attachments**:
  - `config.toml` (optional) — per-request settings; file-path keys are patched out so `InputData` validation doesn't try to read missing files. Falls back to the server's default config loaded at startup.
  - `source.txt` — source text; if absent, the plain-text email body is used.
  - `target.txt` — existing translation (triggers review mode instead of translation).
  - `glossary.txt` — user glossary overrides (`term = translation` lines).
- Runs the full translation/review pipeline (`src/main.py:main`) with the constructed settings.
- Sends the result back as a threaded reply via the Resend Send API (`In-Reply-To` header set to the original `message_id`).
- Error replies are sent on attachment-too-large, fetch failure, parsing failure, pipeline failure, or empty output.
- Deployment helpers: `infra/nginx.conf` (TLS-terminating reverse proxy) and `infra/trslagent-email.service` (systemd unit).

## Translation (Async Chunks)

- Chunks are processed **concurrently** via `asyncio.gather()`.
- Two chunking modes: **size-based** (default, `chunk.size` controls max characters) and **divider-based** (set `chunk.divider` to a character like `"-"` — splits on lines of that character repeated 10+ times, overriding size).
- Concurrency is controlled by `ChunkSettings.max_concurrent` (default 3) and `delay_seconds` (default 1.5) to avoid 429 rate limits.
- One chunk failure does not kill the others (`return_exceptions=True`).
- Per-request retries with exponential backoff are preserved for 429/timeout/connection errors.

## Testing / Linting / CI

- No tests, no linting config, no typecheck config, no CI workflows. None exist in this repo.
