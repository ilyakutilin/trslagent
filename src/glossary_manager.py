"""
Glossary Manager
================
Loads a glossary CSV or XLSX, embeds all terms using multilingual-e5-large,
stores them in ChromaDB, and provides RAG-style retrieval per chunk.

Glossary file format (no header required, but header is fine):
  Three columns: id, source_term, target_term
  - id: unique integer identifier for each term pair
  - source_term: term in source language
  - target_term: term in target language
  Supported formats: .csv, .xlsx
  e.g.:
  1,contract termination,Vertragsauflösung
  2,force majeure,höhere Gewalt

The glossary is bidirectional: querying in either language will find the entry.
"""

import csv
import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from openpyxl import load_workbook
from sentence_transformers import SentenceTransformer

from src.config import get_settings, logger

# ── Configuration ─────────────────────────────────────────────────────────────

settings = get_settings()


# ── Why multilingual-e5-large? ────────────────────────────────────────────────
#
# - Trained on 100+ languages including all major European/Asian languages
# - 'e5' means it's an "embed everything" model optimised for retrieval tasks
# - The 'large' variant (560M params) gives better accuracy than 'small'/'base'
#   at the cost of ~2× slower embedding. For a one-time glossary embed, fine.
# - Crucially: it maps equivalent terms in different languages to nearby vectors,
#   so querying with an English chunk will still retrieve German glossary entries.
# - Important: e5 models expect a prefix on the text:
#     "query: <text>"  for things you're searching WITH
#     "passage: <text>" for things you're searching FOR (the glossary entries)
#   We handle this automatically below.


class GlossaryManager:
    def __init__(self, chroma_dir: str = settings.glossary.chroma_dir):
        self.chroma_dir = chroma_dir
        self.model: Optional[SentenceTransformer] = None
        self.collection = None
        self._terms: list[dict] = []  # in-memory copy for quick lookup by id

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        """Load the embedding model once, reuse thereafter."""
        if self.model is None:
            logger.info(
                f"Loading embedding model '{settings.glossary.embedding_model}' "
                "(first run downloads ~2GB)..."
            )
            self.model = SentenceTransformer(settings.glossary.embedding_model)
            logger.info(
                f"Embedding model '{settings.glossary.embedding_model}' is loaded."
            )
        return self.model

    # ── ChromaDB setup ────────────────────────────────────────────────────────

    def _get_collection(self):
        if self.collection is None:
            client = chromadb.PersistentClient(
                path=self.chroma_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = client.get_or_create_collection(
                name=settings.glossary.collection_name,
                metadata={"hnsw:space": "cosine"},  # cosine similarity for text
            )
            logger.info(f"ChromaDB collection '{self.collection.name}' is loaded.")
        return self.collection

    # ── Glossary loading & embedding ──────────────────────────────────────────

    def load_glossary(
        self, glossary_source_path: str, force_reload: bool = False
    ) -> None:
        """
        Reads glossary CSV or XLSX and upserts all terms into ChromaDB.
        On subsequent runs (chroma_db/ exists), skips re-embedding unless
        force_reload=True — so the expensive embedding only happens once.
        """
        collection = self._get_collection()

        if not force_reload and collection.count() > 0:
            logger.info(
                f"Glossary already loaded ({collection.count()} entries). "
                "Skipping embed step."
            )
            logger.info("Pass force_reload=True to re-embed from scratch.")
            self._load_terms_from_file(glossary_source_path)
            return

        if force_reload:
            logger.info("Clearing existing ChromaDB collection...")
            self.clear()

        logger.info(f"Loading glossary from {glossary_source_path}...")
        self._load_terms_from_file(glossary_source_path)

        if not self._terms:
            raise ValueError("Glossary is empty. Check your file.")

        # Validate IDs before proceeding
        self._validate_ids()

        model = self._get_model()

        # We embed BOTH directions as separate documents so that a query in
        # either language hits the right glossary entry.
        ids, embeddings, documents, metadatas = [], [], [], []

        total_terms = len(self._terms)
        for term in self._terms:
            term_id = term["id"]
            src = term["source"]
            tgt = term["target"]

            # Store canonical pair (first term is always canonical_source, second is canonical_target)
            # This allows us to normalize the direction in retrieval
            canonical_src, canonical_tgt = sorted([src, tgt])

            # Forward entry (source → target)
            fwd_text = f"passage: {src}"
            fwd_embedding = model.encode(fwd_text, normalize_embeddings=True).tolist()
            ids.append(f"fwd_{term_id}")
            embeddings.append(fwd_embedding)
            documents.append(src)
            metadatas.append(
                {
                    "source": src,
                    "target": tgt,
                    "direction": "fwd",
                    "canonical_source": canonical_src,
                    "canonical_target": canonical_tgt,
                }
            )

            # Reverse entry (target → source) — same pair, but queryable from target lang
            rev_text = f"passage: {tgt}"
            rev_embedding = model.encode(rev_text, normalize_embeddings=True).tolist()
            ids.append(f"rev_{term_id}")
            embeddings.append(rev_embedding)
            documents.append(tgt)
            metadatas.append(
                {
                    "source": tgt,
                    "target": src,
                    "direction": "rev",
                    "canonical_source": canonical_src,
                    "canonical_target": canonical_tgt,
                }
            )

            logger.debug(f"Embedding term {term_id}: {canonical_src} = {canonical_tgt}")
            if settings.log.level != "DEBUG":
                # Progress logging - every 100 terms (200 embeddings)
                if len(ids) % 200 == 0 or len(ids) == total_terms * 2:
                    progress = len(ids) // 2
                    progress_pct = (progress / total_terms) * 100
                    logger.info(
                        f"Embedding terms: {progress}/{total_terms} ({progress_pct:.1f}%)"
                    )

        # Batch upsert (handles duplicates gracefully)
        BATCH = 500
        for start in range(0, len(ids), BATCH):
            collection.upsert(
                ids=ids[start : start + BATCH],
                embeddings=embeddings[start : start + BATCH],
                documents=documents[start : start + BATCH],
                metadatas=metadatas[start : start + BATCH],
            )
            logger.info(
                f"  Upserted {min(start + BATCH, len(ids))}/{len(ids)} entries..."
            )

        logger.info(
            f"Glossary embedded and stored. Total entries in DB: {collection.count()}"
        )

    def _validate_ids(self) -> None:
        """Validate that all term IDs are unique integers."""
        ids = []
        for i, term in enumerate(self._terms):
            term_id = term.get("id")
            if term_id is None:
                raise ValueError(f"Term at index {i} is missing an 'id' field.")
            if not isinstance(term_id, int):
                raise ValueError(
                    f"Term at index {i} has invalid 'id' value '{term_id}'. "
                    "ID must be an integer."
                )
            ids.append(term_id)

        # Check for duplicates
        unique_ids = set(ids)
        if len(ids) != len(unique_ids):
            duplicates = [id for id in ids if ids.count(id) > 1]
            raise ValueError(
                f"Duplicate ID(s) found in glossary: {set(duplicates)}. "
                "All IDs must be unique."
            )

        logger.debug(f"Validated {len(unique_ids)} unique IDs.")

    def _load_terms_from_csv(self, csv_path: str) -> None:
        """Read CSV into self._terms list."""
        self._terms = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            # Skip header row
            next(reader, None)
            for row in reader:
                if len(row) < 3:
                    logger.warning(f"Row has less than 3 columns: {' | '.join(row)}")
                    continue
                try:
                    term_id = int(row[0].strip())
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Invalid ID value '{row[0].strip()}' in CSV. "
                        "ID must be a valid integer."
                    )
                source, target = row[1].strip(), row[2].strip()
                if source and target:
                    self._terms.append(
                        {"id": term_id, "source": source, "target": target}
                    )
        logger.info(f"Read {len(self._terms)} terms from CSV.")

    def _load_terms_from_xlsx(self, xlsx_path: str) -> None:
        """Read XLSX into self._terms list."""
        self._terms = []

        if not os.path.exists(xlsx_path):
            raise FileNotFoundError(f"XLSX file not found: {xlsx_path}")

        try:
            workbook = load_workbook(xlsx_path, read_only=True, data_only=True)
            sheet = workbook.active

            if sheet is None:
                workbook.close()
                raise ValueError("XLSX file is empty or has no active sheet.")

            # Skip header row by enumerating and skipping index 0
            for i, row in enumerate(sheet.iter_rows(values_only=True)):
                if i == 0:  # Skip header row
                    continue
                if not row or len(row) < 3:
                    continue
                try:
                    term_id = int(str(row[0]).strip())
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Invalid ID value '{row[0]}' in XLSX at row {i + 1}. "
                        "ID must be a valid integer."
                    )
                source, target = str(row[1]).strip(), str(row[2]).strip()
                if source and target:
                    self._terms.append(
                        {"id": term_id, "source": source, "target": target}
                    )

        finally:
            if "workbook" in locals():
                workbook.close()

        logger.info(f"Read {len(self._terms)} terms from XLSX.")

    def _load_terms_from_file(self, file_path: str) -> None:
        """Auto-detect file format and load accordingly."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".csv":
            self._load_terms_from_csv(file_path)
        elif ext == ".xlsx":
            self._load_terms_from_xlsx(file_path)
        else:
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                "Only .csv and .xlsx files are supported."
            )

    # ── RAG retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_text: str,
        source_lang: str,
        target_lang: str,
        top_k: int = 20,
    ) -> list[dict]:
        """
        Given a text chunk (query), returns the top_k most relevant glossary
        terms as a list of {"source": ..., "target": ...} dicts.

        The returned dicts always have source=source_lang term, target=target_lang term
        regardless of which direction the DB entry was stored in.
        """
        model = self._get_model()
        collection = self._get_collection()

        if collection.count() == 0:
            return []

        # e5 requires "query: " prefix for the search query
        query_embedding = model.encode(
            f"query: {query_text}", normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(
                top_k * 2, collection.count()
            ),  # over-fetch, then deduplicate
            include=["metadatas", "distances"],
        )

        # Check if we got any results
        if not results or not results["metadatas"] or not results["metadatas"][0]:
            return []

        seen_pairs = set()
        terms = []

        for meta in results["metadatas"][0]:
            # Use canonical pair for deduplication (sorted ensures fwd and rev are grouped)
            canonical_pair = (meta["canonical_source"], meta["canonical_target"])
            if canonical_pair in seen_pairs:
                continue
            seen_pairs.add(canonical_pair)

            # Normalize direction: always return {source: src_lang, target: tgt_lang}
            # Compare the requested languages with the canonical pair
            canonical_src, canonical_tgt = canonical_pair

            # Safely convert to strings for comparison
            src_lang_str = str(source_lang).lower()
            tgt_lang_str = str(target_lang).lower()
            canonical_src_str = str(canonical_src).lower()
            canonical_tgt_str = str(canonical_tgt).lower()

            if canonical_src_str == src_lang_str and canonical_tgt_str == tgt_lang_str:
                # Entry already in correct direction
                terms.append({"source": meta["source"], "target": meta["target"]})
            elif (
                canonical_src_str == tgt_lang_str and canonical_tgt_str == src_lang_str
            ):
                # Entry is reversed - swap it
                terms.append({"source": meta["target"], "target": meta["source"]})
            else:
                # Shouldn't happen if canonical pairs are consistent, but return as-is
                terms.append({"source": meta["source"], "target": meta["target"]})

            if len(terms) >= top_k:
                break

        # Log retrieval results
        logger.info(f"Retrieved {len(terms)} glossary terms for chunk")
        if terms:
            pairs_log = "\n".join(f"  - {t['source']} → {t['target']}" for t in terms)
            logger.debug(f"Glossary pairs:\n{pairs_log}")

        return terms

    # ── Utility ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._get_collection().count()

    def clear(self) -> None:
        """Wipe the ChromaDB collection (use before force_reload)."""
        client = chromadb.PersistentClient(
            path=self.chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        client.delete_collection(settings.glossary.collection_name)
        self.collection = None
        logger.info("Collection cleared.")
