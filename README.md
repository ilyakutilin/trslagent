# trslagent — AI Translation Agent

Glossary-aware LLM document translation and review. Takes an `.xml` glossary (Multiterm export) or a plain-text user glossary, matches terms against source text with lemmatization and Aho-Corasick, then sends chunked text to an LLM (OpenRouter) for translation — or reviews an existing translation for critical mistakes.

## Architecture overview

```
Config TOML
    │
    ├── Glossary  ──► parse .xml (Multiterm) + .txt (user)  ──► lemmatize
    │                                                              │
    ├── Source text ──► chunk (RecursiveCharacterTextSplitter) ─┐  │
    │                                                           │  │
    │   For each chunk:                                         │  │
    │     match glossary terms (Aho-Corasick on lemmas)  ◄──────┘  │
    │     deduplicate (user overrides auto)                     │
    │     build LLM prompt (system + user + dictionary)            │
    │     call OpenRouter LLM                                      │
    │                                                              │
    └── Stitch ──► write result
```

## Project structure

```
src/
├── cli.py                    # entry point — the only CLI
├── config.py                 # configuration system (pydantic-settings)
├── main.py                   # pipeline orchestrator
├── llm.py                    # OpenRouter API wrapper (5-retry backoff)
├── translator.py             # per-chunk translation prompt builder
├── reviewer.py               # full-text review/proofread (no chunking)
├── splitter.py               # text → chunks → stitch
├── lemmatizer.py             # EN (spaCy) / RU (pymorphy3) lemmatizer
├── utils.py                  # file I/O helpers
├── glossary/
│   ├── parser.py             # XML & user-glossary parsers + cache logic
│   ├── cache.py              # .pickle cache (SHA-256 validated)
│   ├── matcher.py            # Aho-Corasick term matching
│   ├── models.py             # Term, GlossaryEntry, GlossaryFile
│   └── get_abbrs.py          # standalone abbreviation extractor
files/
├── config.toml               # working config
├── source.txt                # sample source
├── projgloss.txt             # sample user glossary
├── abbrs                     # known abbreviations (auto-extracted)
└── glossary/                 # auto glossary XMLs + .pickle caches
config.example.toml           # config reference
.env.example                  # env vars reference
```

## Setup

**Requirements:** Python 3.13, [uv](https://docs.astral.sh/uv/).

```bash
uv sync

# spaCy model — must be installed manually
python -m spacy download en_core_web_sm

# Configure secrets
cp .env.example .env
# Edit .env: set LLM__API_KEY=your-openrouter-key
chmod 600 .env
```

## Configuration

Settings are loaded with priority: **TOML > env vars > `.env` > defaults**.

Env vars use `__` as nesting delimiter: `LLM__MODEL`, `CHUNK__SIZE`, `GLOSSARY__XML_DIR_PATH`, `LOG__LEVEL`, etc.

### TOML config

Minimal config for translation:

```toml
[input_data]
source_lang = "en"
target_lang = "ru"
source_file_path = "files/source.txt"
auto_glossary = true

[output_data]
result_file_path = "files/result.md"
```

All sections and their keys:

| Section       | Key                            | Default                         | Description                                         |
| ------------- | ------------------------------ | ------------------------------- | --------------------------------------------------- |
| `input_data`  | `source_lang`                  | `"en"`                          | ISO 639-1 code or full name                         |
|               | `target_lang`                  | `"en"`                          | ISO 639-1 code or full name                         |
|               | `specialized_in`               | —                               | Domain expertise for system prompt                  |
|               | `doc_type`                     | —                               | e.g. `"letter"`, `"procedure"`                      |
|               | `doc_title`                    | —                               | Document title for prompt context                   |
|               | `source_file_path`             | `"files/source.txt"`            | Input text file                                     |
|               | `target_file_path`             | —                               | Set to enable **review mode**                       |
|               | `auto_glossary`                | `false`                         | Match auto glossary against source                  |
|               | `user_glossary_file_path`      | —                               | Path to user glossary `.txt`                     |
| `output_data` | `result_file_path`             | `"files/result.md"`             | Output file                                         |
|               | `raw_result_file_path`         | `"files/raw_result.json"`       | (unused)                                            |
|               | `print_prompt_only`            | `false`                         | Dry-run: print prompts, skip LLM                    |
|               | `timestamped_result_filenames` | `true`                          | Append `_YYYY-MM-DD_HH-MM-SS`                       |
| `llm`         | `base_url`                     | `https://openrouter.ai/api/v1`  |                                                     |
|               | `api_key`                      | from `.env`                     | OpenRouter key                                      |
|               | `model`                        | `"anthropic/claude-3.5-sonnet"` | Any OpenRouter model string                         |
|               | `temperature`                  | —                               | `float` or unset for model default                  |
|               | `reasoning_effort`             | —                               | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `chunk`       | `size`                         | `6000`                          | Max chunk size in characters                        |
|               | `max_concurrent`               | `3`                             | Max simultaneous LLM calls per task                 |
|               | `delay_seconds`                | `1.5`                           | Seconds between launching chunk tasks               |
| `glossary`    | `xml_dir_path`                 | `"files/glossary"`              | Dir with Multiterm `.xml` exports                   |
|               | `known_abbrs_file_path`        | —                               | Path to abbreviations list                          |
| `log`         | `level`                        | `"INFO"`                        | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`         |
|               | `format`                       | loguru default                  | Custom log format string                            |

### `.env` entries

```
LLM__API_KEY=sk-or-v1-...
LLM__MODEL=qwen/qwen3.7-max
LLM__TEMPERATURE=0.1
LLM__REASONING_EFFORT=high
CHUNK__SIZE=6000
CHUNK__MAX_CONCURRENT=3
CHUNK__DELAY_SECONDS=1.5
GLOSSARY__XML_DIR_PATH=files/glossary
GLOSSARY__KNOWN_ABBRS_FILE_PATH=files/abbrs
LOG__LEVEL=INFO
```

Note: `LLM__API_KEY` is stored as a `SecretStr` and must be set in `.env` or as an env var — not in TOML.

## Usage

### Translation

```bash
python src/cli.py files/config.toml
```

What happens:

1. Auto glossary XMLs are parsed (cached as `.pickle`, re-parsed only if XML changed)
2. User glossary (if configured) is parsed and lemmatized
3. Source text is split into chunks
4. For each chunk, glossary terms are matched via lemmatized Aho-Corasick, user glossary overrides auto where they conflict, matched terms are injected into the LLM system prompt
5. Chunks are translated **concurrently** (up to `max_concurrent` at a time, staggered by `delay_seconds`) and stitched back together
6. Result is written to `result_file_path`

### Review

Set `target_file_path` in your TOML to enable review mode:

```toml
[input_data]
source_lang = "en"
target_lang = "ru"
source_file_path = "files/source.md"
target_file_path = "files/result.md"    # existing translation
auto_glossary = true
```

This sends the full source + target text (no chunking) to the LLM and asks it to report critical mistakes: missing/incorrect translations, distortions, spelling errors, wrong numbers, dictionary deviations.

### Dry-run (print prompts only)

Either set in TOML:

```toml
[output_data]
print_prompt_only = true
```

Or via env: `LLM_OUTPUT_DATA__PRINT_PROMPT_ONLY=true`

This skips all LLM calls and prints the constructed system + user prompts to stdout.

### Export glossary matches

```bash
python src/cli.py files/config.toml --match-glossary --match-output matches.txt
```

Runs the full glossary matching pipeline and writes the matched term=translation pairs to the specified file. No LLM calls are made.

### Extract abbreviations

```bash
python src/glossary/get_abbrs.py files/config.toml
```

Scans the auto glossary for abbreviation-like terms (all-uppercase or mixed-case, ≤8 chars) and writes them to `files/abbrs`. These are then excluded from lemmatization during matching.

## Glossary

### Auto glossary (Multiterm XML)

Export from Multiterm with default settings. Place all `.xml` files in one directory (default: `files/glossary`).

Expected XML structure (`<mtf>` root):

```xml
<mtf>
  <conceptGrp>
    <concept>1234</concept>
    <languageGrp>
      <language lang="EN" type="English"/>
      <termGrp><term>work package</term></termGrp>
    </languageGrp>
    <languageGrp>
      <language lang="RU" type="Russian"/>
      <termGrp><term>комплекс работ</term></termGrp>
    </languageGrp>
  </conceptGrp>
</mtf>
```

- The glossary is **bidirectional** — entries are used regardless of which language is source vs target
- Synonyms are split by `;` or `|`
- Cache is stored as `.pickle` files alongside the XMLs, validated by SHA-256 hash

### User glossary (override)

Plain text file, one entry per line:

```
AGREEMENT = СОГЛАШЕНИЕ
mechanical completion = завершение механомонтажных работ
MOP | MOP test = испытание под максимальным рабочим давлением
commissioning = пусконаладка | ПНР
```

- `#` lines are comments, blank lines are ignored
- Format: `source_term = target_term`
- Synonyms separated by `;` or `|`
- **User glossary always overrides the auto glossary** on a per-term basis
- Terms in the user glossary that don't appear in the auto glossary are still included

## Lemmatization

- **English**: spaCy `en_core_web_sm` (must be manually installed)
- **Russian**: pymorphy3
- **Other languages**: not supported — terms are skipped
- Known abbreviations (from `files/abbrs`) are preserved as-is during lemmatization

## LLM

Uses the OpenRouter API (OpenAI-compatible endpoint). 5 retries with exponential backoff for timeouts, rate limits, and connection errors. Other errors raise immediately.

System prompt structure for translation:

```
You are a professional experienced translator specialized in <domain>.
Translate from <source> into <target>.
The text for translation is an extract from a <doc_type> titled '<doc_title>'.
The following terms were provided by the user. If a term appears in the source text,
its translation must be taken from this dictionary strictly.
<user dictionary start>
SOURCE_TERM = TARGET_TERM
<user dictionary end>

The following terms were automatically matched from a reference glossary.
Note that some terms may not be contextually relevant.
Consider and use them only if applicable; otherwise ignore.
<auto dictionary start>
SOURCE_TERM = TARGET_TERM
<auto dictionary end>
```

## Development notes

- **No tests, no linting, no CI.** This project has none.
- `src/models.py` is legacy/dead code — everything is in `src/config.py`.
- `pyproject.toml` defines `trslagent = "src.cli:cli"` as a console script (available after `uv sync`).
- The `tmp/` directory contains experimental scratch files and is not part of the production codebase.
