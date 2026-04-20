# AI Translation Agent

Translates large documents using:

- **LangChain** for sentence-safe text chunking
- **multilingual-e5-large** embeddings for cross-lingual glossary search
- **ChromaDB** for local persistent vector storage (glossary RAG)
- **OpenRouter** for LLM access (any model)

---

## Setup

```bash
# Install dependencies
uv sync
```

The first run will download the `multilingual-e5-large` model (~2GB). This is cached locally by `sentence-transformers` — subsequent runs are instant.

---

## Glossary format

A three-column file with a unique integer ID for each term pair. Header row is optional (auto-detected and skipped).
File name is important as it will be used in the IDs in the DB. So you should keep the file name consistent when updating.

Supported formats: **CSV** (`.csv`) and **XLSX** (`.xlsx`).

The three columns are:

1. **id** - A unique integer identifier for each term pair
2. **source_term** - Term in the source language
3. **target_term** - Term in the target language

All IDs must be unique integers. The system will validate this on loading.

### CSV example

```csv
id,source_term,target_term
1,contract termination,Vertragsauflösung
2,force majeure,höhere Gewalt
3,party of the first part,Partei des ersten Teils
```

### XLSX example

An Excel spreadsheet with three columns:

| id  | source_term             | target_term             |
| --- | ----------------------- | ----------------------- |
| 1   | contract termination    | Vertragsauflösung       |
| 2   | force majeure           | höhere Gewalt           |
| 3   | party of the first part | Partei des ersten Teils |

The glossary is **bidirectional** — you only need each pair once. The system embeds both directions so queries from either language find the right entry.

---

## Usage

### Command line

#### translator.py

```bash
python src/translator.py document.txt English German \
    --glossary glossary.csv \
    --output translated.txt \
    --model anthropic/claude-3.5-sonnet
```

**Positional arguments:**

| Argument      | Description                        |
| ------------- | ---------------------------------- |
| `input_file`  | Path to the text file to translate |
| `source_lang` | Source language (e.g., 'English')  |
| `target_lang` | Target language (e.g., 'German')   |

**Optional arguments:**

| Argument              | Description                                                             |
| --------------------- | ----------------------------------------------------------------------- |
| `--glossary`          | Path to glossary CSV or XLSX file (default: `glossary.csv`)             |
| `--glossary-override` | Path to glossary override file (format: `term = translation`)           |
| `--sync-glossary`     | Sync glossary (add/update/delete terms) instead of just loading         |
| `--output`            | Output file path (default: `translated.txt`)                            |
| `--model`             | OpenRouter model string (default: configured in `.env`)                 |
| `--print-prompt-only` | Print prompts that would be sent to the LLM without actually calling it |

**Glossary override:**

You can provide a glossary override file to use specific translations for certain terms in specific documents, overriding the embedded glossary.

Create a text file (e.g., `override.txt`) with one term per line in the format:

```
AGREEMENT = ДОГОВОР
contract = контракт
party = сторона
```

- Lines starting with `#` are treated as comments
- Blank lines are ignored
- Format: `source_term = target_term`

Use it with the `--glossary-override` flag:

```bash
python src/translator.py document.txt English German \
    --glossary glossary.csv \
    --glossary-override override.txt \
    --output translated.txt
```

The override glossary takes priority over the embedded glossary. If a term appears in both, the override translation is used. If a term only appears in the override (and in the document), it's added to the prompt.

This is useful when you have domain-specific documents where certain terms should be translated differently than in your main glossary (e.g., a contract from a different company project).

#### glossary_manager.py

```bash
python src/glossary_manager.py sync --glossary glossary.csv
```

**Subcommands:**

| Command | Description                                           |
| ------- | ----------------------------------------------------- |
| `sync`  | Sync glossary with ChromaDB (add/update/delete terms) |

**Arguments:**

| Argument     | Description                                                 |
| ------------ | ----------------------------------------------------------- |
| `--glossary` | Path to glossary CSV or XLSX file (default: `glossary.csv`) |

---

### In Python

```python
from src.glossary_manager import GlossaryManager
from src.translator import translate_document

gm = GlossaryManager()
gm.load_glossary("glossary.csv")   # embeds on first run, cached after

result = translate_document(
    text=open("document.txt").read(),
    source_lang="English",
    target_lang="German",
    glossary_manager=gm,
    model="anthropic/claude-3.5-sonnet",  # or any OpenRouter model string
)

print(result)
```

---

## How it works

```
Large text
    │
    ▼
[split_text()]  ← RecursiveCharacterTextSplitter
    │             Splits on: \n\n → \n → ". " → words
    │             Never cuts mid-sentence
    │
    ├── For each chunk:
    │       │
    │       ▼
    │   [Embed chunk] ──► query ChromaDB ──► Top-20 glossary terms
    │       │
    │       ▼
    │   [LLM prompt]:
    │     "Translate EN→DE.
    │      MANDATORY GLOSSARY: term1 → Term1, term2 → Term2 ...
    │      PREVIOUS SEGMENT: [tail of last translation]
    │      TEXT: [chunk]"
    │       │
    │       ▼
    │   Translated chunk
    │
    ▼
[stitch_chunks()]  ← deduplicates overlap at seams
    │
    ▼
Final translated document
```

---

## Configuration

```bash
cp .env.example .env
chmod 600 .env
```

Then edit the values.

---

## On the embedding model

`intfloat/multilingual-e5-large` is a strong choice for this use case because:

- Trained on 100+ languages — maps equivalent terms across languages to nearby vectors
- Optimised for retrieval (the "e5" = embed-everything architecture)
- Handles domain-specific vocabulary well (legal, medical, technical)
- The `large` variant is more accurate than `small`/`base`; since glossary embedding is a one-time cost, the extra size is fine

A lighter alternative is `intfloat/multilingual-e5-small` (~120MB vs ~2GB) — good for testing.

---

## Tips

- **Lower temperature** (`0.1`) in the config gives more consistent, literal output — appropriate for technical/legal documents.
- **Chunk size tuning**: larger chunks = more context per prompt = better translation quality, but higher cost. Start with 6000 characters and adjust.
- **OpenRouter model strings**: find them at https://openrouter.ai/models
