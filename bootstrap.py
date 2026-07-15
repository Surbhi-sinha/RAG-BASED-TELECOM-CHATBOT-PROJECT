"""
Ensures the Chroma vector store exists before the app serves requests.

On a fresh deploy (e.g. Streamlit Community Cloud) the `chroma_store/` directory
is not present because it's gitignored. This module rebuilds it from the
committed source files in `data/` by running each ingestion script once.

It's idempotent: a collection that already has vectors is skipped, so restarts
are fast and only a missing/empty collection triggers a rebuild.
"""
import chromadb

import ingest_faq
import ingest_tickets
import ingest_pdf

CHROMA_DIR = "chroma_store"

# (collection name, ingestion entry point)
_STEPS = [
    ("faq", ingest_faq.main),
    ("tickets", ingest_tickets.main),
    ("guides", ingest_pdf.main),
]


def _count(collection: str) -> int:
    """Number of vectors in a collection, or 0 if it doesn't exist yet."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        return client.get_collection(collection).count()
    except Exception:
        return 0


def ensure_ingested() -> None:
    """Build any collection that is missing or empty. Safe to call on every start."""
    for name, run_ingest in _STEPS:
        if _count(name) == 0:
            print(f"[bootstrap] Collection '{name}' is empty — ingesting...")
            run_ingest()
        else:
            print(f"[bootstrap] Collection '{name}' already populated — skipping.")


if __name__ == "__main__":
    ensure_ingested()
