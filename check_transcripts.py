"""
InteliCore — Transcript & Database Diagnostics
================================================
Checks transcripts directory and ChromaDB health.
"""

import os
import sys
from chromadb import PersistentClient
from config import Config


def check_transcripts(cfg: Config):
    """Check transcript files and database status."""

    print("=" * 50)
    print("  🔍 InteliCore Diagnostics")
    print("=" * 50)

    # --- Check transcripts directory ---
    print(f"\n📁 Transcripts directory: {os.path.abspath(cfg.transcripts_dir)}")

    if not os.path.isdir(cfg.transcripts_dir):
        print("   ❌ Directory does not exist!")
        print(f"   Create it: mkdir {cfg.transcripts_dir}")
        return

    all_files = os.listdir(cfg.transcripts_dir)
    txt_files = sorted(f for f in all_files if f.endswith(".txt"))
    other_files = [f for f in all_files if not f.endswith(".txt") and not f.startswith(".")]

    print(f"   Total files:    {len(all_files)}")
    print(f"   .txt files:     {len(txt_files)}")

    if other_files:
        print(f"   ⚠️  Non-.txt files (ignored): {', '.join(other_files[:5])}")

    if txt_files:
        print(f"\n   {'Filename':<40} {'Size':>10}")
        print(f"   {'─'*40} {'─'*10}")
        total_chars = 0
        for f in txt_files:
            path = os.path.join(cfg.transcripts_dir, f)
            size = os.path.getsize(path)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                char_count = len(fh.read().strip())
            total_chars += char_count
            status = "✅" if char_count > 0 else "⚠️ empty"
            print(f"   {f:<40} {char_count:>8,} chars  {status}")
        print(f"   {'─'*40} {'─'*10}")
        print(f"   {'Total':<40} {total_chars:>8,} chars")
    else:
        print("\n   ⚠️  No .txt transcript files found!")
        print(f"   Add transcript .txt files to: {os.path.abspath(cfg.transcripts_dir)}/")

    # --- Check ChromaDB ---
    print(f"\n📊 ChromaDB: {os.path.abspath(cfg.chroma_db_path)}")

    if not os.path.isdir(cfg.chroma_db_path):
        print("   ⚠️  Database directory does not exist.")
        print("   Run: python build_chroma.py")
        return

    try:
        client = PersistentClient(path=os.path.abspath(cfg.chroma_db_path))
        collection = client.get_or_create_collection(cfg.collection_name)
        count = collection.count()
        print(f"   Collection: {cfg.collection_name}")
        print(f"   Indexed chunks: {count}")

        if count > 0:
            # Show source breakdown
            meta = collection.get(include=["metadatas"])
            if meta and meta["metadatas"]:
                source_counts = {}
                for m in meta["metadatas"]:
                    src = m.get("source", "unknown") if m else "unknown"
                    source_counts[src] = source_counts.get(src, 0) + 1
                print(f"\n   {'Source':<40} {'Chunks':>8}")
                print(f"   {'─'*40} {'─'*8}")
                for src, cnt in sorted(source_counts.items()):
                    print(f"   {src:<40} {cnt:>8}")
        elif txt_files:
            print("   ⚠️  Database is empty but transcripts exist.")
            print("   Run: python build_chroma.py")

    except Exception as e:
        print(f"   ❌ Error reading database: {e}")

    print()


if __name__ == "__main__":
    check_transcripts(Config())
