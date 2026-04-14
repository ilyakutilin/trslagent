"""
Glossary Manager
================
Loads a glossary CSV, embeds all terms using multilingual-e5-large,
stores them in ChromaDB, and provides RAG-style retrieval per chunk.

Glossary CSV format (no header required, but header is fine):
  source_term, target_term
  e.g.:
  contract termination, Vertragsauflösung
  force majeure, höhere Gewalt

The glossary is bidirectional: querying in either language will find the entry.
"""

# TODO: Implement logging instead of print()

import csv
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import get_settings

# ── Configuration ─────────────────────────────────────────────────────────────

settings = get_settings()

# Use centralized settings (can be overridden via environment variables)
EMBEDDING_MODEL = settings.glossary.embedding_model
CHROMA_DIR = settings.glossary.chroma_dir
COLLECTION_NAME = settings.glossary.collection_name


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
    def __init__(self, chroma_dir: str = CHROMA_DIR):
        self.chroma_dir = chroma_dir
        self.model: Optional[SentenceTransformer] = None
        self.collection = None
        self._terms: list[dict] = []  # in-memory copy for quick lookup by id

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        """Load the embedding model once, reuse thereafter."""
        if self.model is None:
            print(
                f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads ~2GB)..."
            )
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            print("Model loaded.")
        return self.model

    # ── ChromaDB setup ────────────────────────────────────────────────────────

    def _get_collection(self):
        if self.collection is None:
            client = chromadb.PersistentClient(
                path=self.chroma_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},  # cosine similarity for text
            )
        return self.collection

    # ── Glossary loading & embedding ──────────────────────────────────────────

    def load_glossary(self, csv_path: str, force_reload: bool = False) -> None:
        """
        Reads glossary CSV and upserts all terms into ChromaDB.
        On subsequent runs (chroma_db/ exists), skips re-embedding unless
        force_reload=True — so the expensive embedding only happens once.
        """
        collection = self._get_collection()

        if not force_reload and collection.count() > 0:
            print(
                f"Glossary already loaded ({collection.count()} entries). Skipping embed step."
            )
            print("Pass force_reload=True to re-embed from scratch.")
            self._load_terms_from_csv(csv_path)
            return

        if force_reload:
            print("Clearing existing ChromaDB collection...")
            self.clear()

        print(f"Loading glossary from {csv_path}...")
        self._load_terms_from_csv(csv_path)

        if not self._terms:
            raise ValueError("Glossary is empty. Check your CSV file.")

        model = self._get_model()

        # We embed BOTH directions as separate documents so that a query in
        # either language hits the right glossary entry.
        ids, embeddings, documents, metadatas = [], [], [], []

        for i, term in enumerate(self._terms):
            src = term["source"]
            tgt = term["target"]

            # Store canonical pair (first term is always canonical_source, second is canonical_target)
            # This allows us to normalize the direction in retrieval
            canonical_src, canonical_tgt = sorted([src, tgt])

            # Forward entry (source → target)
            fwd_text = f"passage: {src}"
            fwd_embedding = model.encode(fwd_text, normalize_embeddings=True).tolist()
            ids.append(f"fwd_{i}")
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
            ids.append(f"rev_{i}")
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

        # Batch upsert (handles duplicates gracefully)
        BATCH = 500
        for start in range(0, len(ids), BATCH):
            collection.upsert(
                ids=ids[start : start + BATCH],
                embeddings=embeddings[start : start + BATCH],
                documents=documents[start : start + BATCH],
                metadatas=metadatas[start : start + BATCH],
            )
            print(f"  Upserted {min(start + BATCH, len(ids))}/{len(ids)} entries...")

        print(
            f"Glossary embedded and stored. Total entries in DB: {collection.count()}"
        )

    def _load_terms_from_csv(self, csv_path: str) -> None:
        """Read CSV into self._terms list."""
        self._terms = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                source, target = row[0].strip(), row[1].strip()
                # Skip header rows
                if source.lower() in ("source", "source_term", "term", "from"):
                    continue
                if source and target:
                    self._terms.append({"source": source, "target": target})
        print(f"Read {len(self._terms)} terms from CSV.")

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
        client.delete_collection(COLLECTION_NAME)
        self.collection = None
        print("Collection cleared.")
