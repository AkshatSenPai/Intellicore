"""
InteliCore — Build ChromaDB Vector Store
=========================================
Reads transcripts from ./transcripts/, chunks them, embeds them,
and stores them in a persistent ChromaDB collection.

Improvements over original:
  - Text chunking with overlap for better retrieval precision
  - Metadata (source filename, chunk index) stored alongside embeddings
  - Deduplication: skips files already in the database
  - Batch embedding for efficiency
  - Progress bar and summary statistics
  - Error handling per file
"""

import os
import sys
import time
import hashlib
from typing import List, Tuple

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from config import Config


# ──────────────────────────────────────────────
#  Text chunking
# ──────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks.
    Tries to break on sentence boundaries when possible.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary (., !, ?) within the last 20% of the chunk
        if end < len(text):
            search_start = start + int(chunk_size * 0.8)
            best_break = -1
            for delim in [".\n", ". ", "!\n", "! ", "?\n", "? "]:
                idx = text.rfind(delim, search_start, end)
                if idx > best_break:
                    best_break = idx + len(delim)
            if best_break > start:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < len(text) else len(text)

    return chunks


def file_hash(filepath: str) -> str:
    """Quick hash of file contents for dedup tracking."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ──────────────────────────────────────────────
#  Main build pipeline
# ──────────────────────────────────────────────

def build_database(cfg: Config, force_rebuild: bool = False):
    """Build or update the ChromaDB vector store from transcripts."""

    cfg.validate()

    # --- Init ChromaDB ---
    client = PersistentClient(path=os.path.abspath(cfg.chroma_db_path))

    if force_rebuild:
        try:
            client.delete_collection(cfg.collection_name)
            print("🗑️  Deleted existing collection (force rebuild).")
        except Exception:
            pass

    collection = client.get_or_create_collection(cfg.collection_name)

    # --- Check what's already indexed ---
    existing_meta = collection.get(include=["metadatas"])
    existing_sources = set()
    if existing_meta and existing_meta["metadatas"]:
        existing_sources = {m.get("source") for m in existing_meta["metadatas"] if m}

    # --- Load embedding model ---
    print(f"Loading embedding model: {cfg.embedding_model}...")
    embedder = SentenceTransformer(cfg.embedding_model)

    # --- Process transcripts ---
    txt_files = sorted(
        f for f in os.listdir(cfg.transcripts_dir) if f.endswith(".txt")
    )

    if not txt_files:
        print(f"⚠️  No .txt files found in {cfg.transcripts_dir}/")
        return

    all_ids, all_docs, all_embeds, all_metas = [], [], [], []
    skipped, failed = 0, 0

    for filename in txt_files:
        filepath = os.path.join(cfg.transcripts_dir, filename)

        # Skip already-indexed files (unless force rebuild)
        if filename in existing_sources and not force_rebuild:
            print(f"  ⏭️  {filename} (already indexed)")
            skipped += 1
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text:
                print(f"  ⚠️  {filename} (empty, skipping)")
                skipped += 1
                continue

            # Chunk the text
            chunks = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
            print(f"  📄 {filename} → {len(chunks)} chunk(s) ({len(text):,} chars)")

            for j, chunk in enumerate(chunks):
                chunk_id = f"{filename}::chunk_{j}"
                all_ids.append(chunk_id)
                all_docs.append(chunk)
                all_metas.append({
                    "source": filename,
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk),
                })

        except Exception as e:
            print(f"  ❌ {filename}: {e}")
            failed += 1

    # --- Batch embed and insert ---
    if all_docs:
        print(f"\nEmbedding {len(all_docs)} chunks...")
        t0 = time.time()
        all_embeds = embedder.encode(all_docs, show_progress_bar=True, batch_size=32).tolist()
        elapsed = time.time() - t0

        # Insert in batches (ChromaDB has a max batch size of ~5000)
        BATCH_SIZE = 5000
        for batch_start in range(0, len(all_docs), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(all_docs))
            collection.add(
                ids=all_ids[batch_start:batch_end],
                documents=all_docs[batch_start:batch_end],
                embeddings=all_embeds[batch_start:batch_end],
                metadatas=all_metas[batch_start:batch_end],
            )
            print(f"   Inserted batch {batch_start // BATCH_SIZE + 1}/{-(-len(all_docs) // BATCH_SIZE)} ({batch_end - batch_start} chunks)")

        print(f"\n{'='*50}")
        print(f"✅ Indexed {len(all_docs)} chunks from {len(txt_files) - skipped - failed} file(s)")
        print(f"   ⏭️  Skipped: {skipped} | ❌ Failed: {failed}")
        print(f"   ⏱️  Embedding time: {elapsed:.1f}s")
        print(f"   📂 Database: {os.path.abspath(cfg.chroma_db_path)}")
        print(f"   📊 Total docs in collection: {collection.count()}")
        print(f"{'='*50}")
    else:
        print("\n⚠️  No new documents to index.")
        print(f"   📊 Existing docs in collection: {collection.count()}")


# ──────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build InteliCore vector database from transcripts.")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the entire database.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override chunk size (chars).")
    parser.add_argument("--transcripts", type=str, default=None, help="Override transcripts directory.")
    args = parser.parse_args()

    cfg = Config()
    if args.chunk_size:
        cfg.chunk_size = args.chunk_size
    if args.transcripts:
        cfg.transcripts_dir = args.transcripts

    build_database(cfg, force_rebuild=args.force)
