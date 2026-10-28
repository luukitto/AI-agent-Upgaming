"""Qdrant persistence layer.

Two collections:
* `chunks`    - one point per chunk: dense vector + payload (text, doc_id, title, metadata)
* `documents` - one point per document: registry (hashes, version, timestamps); dummy 1-d vector

Local development uses Qdrant's embedded on-disk mode; production uses Qdrant Cloud (same API).
"""
from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Iterator

import numpy as np
from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.models import Chunk, DocumentRecord, SearchHit

NAMESPACE = uuid.UUID("7c1f2b9e-1b7a-4c55-9d1e-3f2a6b8c9d01")


def point_id(key: str) -> str:
    """Qdrant needs UUID/int ids; derive a stable UUID from our string id."""
    return str(uuid.uuid5(NAMESPACE, key))


class VectorStore:
    def __init__(self, client: QdrantClient, chunks: str, documents: str, dim: int):
        self.client = client
        self.chunks = chunks
        self.documents = documents
        self.dim = dim
        self._ensure_collections()

    # ---------- setup ----------
    def _ensure_collections(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.chunks not in existing:
            self.client.create_collection(
                self.chunks,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE,
                                                   on_disk=True),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
            )
            self.client.create_payload_index(self.chunks, "doc_id", models.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.chunks, "corpus", models.PayloadSchemaType.KEYWORD)
        if self.documents not in existing:
            self.client.create_collection(
                self.documents,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT),
            )
            self.client.create_payload_index(self.documents, "content_hash",
                                             models.PayloadSchemaType.KEYWORD)

    # ---------- chunks ----------
    def upsert_chunks(self, chunks: list[Chunk], vectors: np.ndarray, batch: int = 512) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            self.client.upsert(
                self.chunks,
                points=[
                    models.PointStruct(
                        id=point_id(c.chunk_id),
                        vector=vectors[i + j].tolist(),
                        payload={
                            "chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text,
                            "position": c.position, "title": c.title, "source": c.source,
                            "corpus": c.metadata.get("corpus", "user"), "metadata": c.metadata,
                        },
                    )
                    for j, c in enumerate(part)
                ],
                wait=True,
            )

    def delete_doc_chunks(self, doc_id: str) -> None:
        self.client.delete(
            self.chunks,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])),
            wait=True,
        )

    def dense_search(self, vector: np.ndarray, limit: int, doc_ids: list[str] | None = None,
                     with_vectors: bool = False) -> list[tuple[SearchHit, np.ndarray | None]]:
        flt = None
        if doc_ids:
            flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))])
        res = self.client.query_points(
            self.chunks, query=vector.tolist(), limit=limit, query_filter=flt,
            with_payload=True, with_vectors=with_vectors,
            search_params=models.SearchParams(hnsw_ef=128),
        ).points
        out = []
        for p in res:
            vec = np.asarray(p.vector, dtype=np.float32) if with_vectors and p.vector is not None else None
            out.append((self._hit(p.payload, p.score), vec))
        return out

    def get_chunks(self, chunk_ids: list[str], with_vectors: bool = False) -> dict[str, tuple[SearchHit, np.ndarray | None]]:
        if not chunk_ids:
            return {}
        pts = self.client.retrieve(self.chunks, ids=[point_id(c) for c in chunk_ids],
                                   with_payload=True, with_vectors=with_vectors)
        out = {}
        for p in pts:
            vec = np.asarray(p.vector, dtype=np.float32) if with_vectors and p.vector is not None else None
            out[p.payload["chunk_id"]] = (self._hit(p.payload, 0.0), vec)
        return out

    def doc_chunks(self, doc_id: str) -> list[SearchHit]:
        pts, _ = self.client.scroll(
            self.chunks, limit=1000, with_payload=True,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]),
        )
        hits = [self._hit(p.payload, 0.0) for p in pts]
        return sorted(hits, key=lambda h: h.metadata.get("position", 0))

    def iter_all_chunks(self, page: int = 2000) -> Iterator[dict]:
        offset = None
        while True:
            pts, offset = self.client.scroll(self.chunks, limit=page, offset=offset,
                                             with_payload=["chunk_id", "doc_id", "text", "title"])
            for p in pts:
                yield p.payload
            if offset is None:
                break

    def count_chunks(self) -> int:
        return self.client.count(self.chunks, exact=True).count

    @staticmethod
    def _hit(payload: dict, score: float) -> SearchHit:
        meta = dict(payload.get("metadata") or {})
        meta["position"] = payload.get("position", 0)
        return SearchHit(chunk_id=payload["chunk_id"], doc_id=payload["doc_id"], title=payload.get("title", ""),
                         source=payload.get("source", ""), text=payload["text"], score=score, metadata=meta)

    # ---------- documents registry ----------
    def upsert_document(self, rec: DocumentRecord) -> None:
        self.client.upsert(self.documents, points=[
            models.PointStruct(id=point_id(rec.doc_id), vector=[1.0], payload=rec.model_dump())], wait=True)

    def upsert_documents(self, recs: list[DocumentRecord], batch: int = 1000) -> None:
        for i in range(0, len(recs), batch):
            self.client.upsert(self.documents, points=[
                models.PointStruct(id=point_id(r.doc_id), vector=[1.0], payload=r.model_dump())
                for r in recs[i:i + batch]], wait=True)

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        pts = self.client.retrieve(self.documents, ids=[point_id(doc_id)], with_payload=True)
        return DocumentRecord(**pts[0].payload) if pts else None

    def delete_document(self, doc_id: str) -> None:
        self.delete_doc_chunks(doc_id)
        self.client.delete(self.documents, points_selector=models.PointIdsList(points=[point_id(doc_id)]), wait=True)

    def list_documents(self, limit: int = 50, offset: str | None = None,
                       corpus: str | None = None) -> tuple[list[DocumentRecord], str | None]:
        flt = None
        if corpus:
            flt = models.Filter(must=[models.FieldCondition(key="metadata.corpus", match=models.MatchValue(value=corpus))])
        pts, next_off = self.client.scroll(self.documents, limit=limit, offset=offset,
                                           with_payload=True, scroll_filter=flt)
        return [DocumentRecord(**p.payload) for p in pts], (str(next_off) if next_off else None)

    def iter_all_documents(self, page: int = 2000) -> Iterator[DocumentRecord]:
        offset = None
        while True:
            pts, offset = self.client.scroll(self.documents, limit=page, offset=offset, with_payload=True)
            for p in pts:
                yield DocumentRecord(**p.payload)
            if offset is None:
                break

    def count_documents(self) -> int:
        return self.client.count(self.documents, exact=True).count


def make_client() -> QdrantClient:
    s = get_settings()
    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None, timeout=60)
    return QdrantClient(path=s.qdrant_path)


@lru_cache
def get_store() -> VectorStore:
    s = get_settings()
    return VectorStore(make_client(), s.chunks_collection, s.documents_collection, s.embed_dim)
