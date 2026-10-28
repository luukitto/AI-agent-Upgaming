"""Index the corpus built by build_corpus.py into the configured Qdrant (local or cloud).

Usage:
  python -m scripts.ingest_corpus                 # everything
  python -m scripts.ingest_corpus --limit 2000    # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from app.config import ROOT_DIR
from app.container import get_container
from app.models import ParsedDocument

CORPUS = ROOT_DIR / "data" / "corpus"
SYNTHETIC = ROOT_DIR / "data" / "synthetic"


def read_jsonl(path: Path, limit: int | None):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)


def hotpot_docs(limit):
    for r in read_jsonl(CORPUS / "hotpot_paragraphs.jsonl", limit):
        yield ParsedDocument(title=r["title"], text=r["text"], source=r["title"], doc_type="wiki",
                             metadata={"dataset": "hotpot_qa"})


def postmortem_docs():
    path = CORPUS / "postmortems.jsonl"
    if not path.exists():
        return
    for r in read_jsonl(path, None):
        yield ParsedDocument(title=r["title"], text=r["text"], source=r["url"], doc_type="postmortem",
                             metadata={"dataset": "danluu/post-mortems", "url": r["url"], "category": r["category"]})


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    c = get_container()
    t = time.time()
    for path in sorted(SYNTHETIC.iterdir()):
        res = c.ingestion.ingest_file(path.read_bytes(), path.name, corpus="novapay")
        print(f"  {path.name}: {res}")
    print("postmortems:", c.ingestion.bulk_ingest(postmortem_docs(), corpus="postmortems"))
    print("hotpot:", c.ingestion.bulk_ingest(hotpot_docs(args.limit), corpus="hotpot"))
    print(f"done in {time.time() - t:.0f}s: {c.store.count_documents()} documents, {c.store.count_chunks()} chunks")


if __name__ == "__main__":
    main()
