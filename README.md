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

## Glossary CSV format

A simple two-column CSV. Header row is optional (auto-detected and skipped).

```csv
source_term,target_term
contract termination,Vertragsauflösung
force majeure,höhere Gewalt
party of the first part,Partei des ersten Teils
```

The glossary is **bidirectional** — you only need each pair once. The system embeds both directions so queries from either language find the right entry.

---

## Usage

### Command line

```bash
python src/translator.py document.txt English German \
    --glossary glossary.csv \
    --output translated.txt \
    --model anthropic/claude-3.5-sonnet
```

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

- **Force re-embed** the glossary after updating the CSV:
  ```python
  gm.load_glossary("glossary.csv", force_reload=True)
  ```
- **Lower temperature** (`0.1`) in the config gives more consistent, literal output — appropriate for technical/legal documents.
- **Chunk size tuning**: larger chunks = more context per prompt = better translation quality, but higher cost. Start with 6000 characters and adjust.
- **OpenRouter model strings**: find them at https://openrouter.ai/models
