"""Copy the local on-disk Qdrant index to Qdrant Cloud, vectors included (no re-embedding).

Embedding the corpus takes ~30-60 min on CPU; copying the finished points takes a few minutes.

Usage (QDRANT_URL / QDRANT_API_KEY set in .env or the environment):
  python -m scripts.copy_to_cloud
"""
from __future__ import annotations

import time

from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.store.vector_store import VectorStore


def copy_collection(src: QdrantClient, dst: QdrantClient, name: str, page: int = 500) -> int:
    offset, n = None, 0
    while True:
        pts, offset = src.scroll(name, limit=page, offset=offset, with_payload=True, with_vectors=True)
        if pts:
            dst.upsert(name, points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in pts], wait=True)
            n += len(pts)
            print(f"  {name}: {n}", end="\r", flush=True)
        if offset is None:
            break
    print()
    return n


def main():
    s = get_settings()
    if not s.qdrant_url:
        raise SystemExit("Set QDRANT_URL (and QDRANT_API_KEY) to the Qdrant Cloud cluster first.")
    src = QdrantClient(path=s.qdrant_path)
    dst = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None, timeout=120)
    dim = src.get_collection(s.chunks_collection).config.params.vectors.size
    VectorStore(dst, s.chunks_collection, s.documents_collection, dim)  # creates collections + payload indexes
    t = time.time()
    for name in (s.documents_collection, s.chunks_collection):
        local, copied = src.count(name, exact=True).count, copy_collection(src, dst, name)
        print(f"{name}: {copied}/{local} copied, cloud now has {dst.count(name, exact=True).count}")
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
