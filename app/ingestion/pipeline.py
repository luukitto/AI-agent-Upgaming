"""Ingestion service: parse -> dedup -> chunk -> embed -> store (Qdrant) + keyword index (BM25).

Document identity: doc_id = "<corpus>:<source>" so re-uploading the same file name is an
*update* (new version, old chunks replaced), while identical content under a different name is
reported as an exact duplicate and not indexed twice.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Iterable

from app.embeddings import Embedder
from app.ingestion.chunking import Chunker
from app.ingestion.dedup import DuplicateIndex, content_hash, simhash, to_signed
from app.ingestion.parsers import parse_file
from app.models import Chunk, DocumentRecord, ParsedDocument
from app.retrieval.hybrid import HybridRetriever
from app.store.vector_store import VectorStore

log = logging.getLogger(__name__)


def make_doc_id(corpus: str, source: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", source).strip("_")[:150]
    return f"{corpus}:{slug}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class IngestionService:
    def __init__(self, store: VectorStore, embedder: Embedder, retriever: HybridRetriever, chunker: Chunker):
        self.store = store
        self.embedder = embedder
        self.retriever = retriever
        self.chunker = chunker
        self.dups = DuplicateIndex()

    def load_dup_index(self) -> None:
        for rec in self.store.iter_all_documents():
            self.dups.add(rec.doc_id, rec.content_hash, rec.simhash)
        log.info("Duplicate index loaded: %d docs", len(self.dups.doc_ids))

    # ---------- single document ----------
    def ingest_file(self, data: bytes, name: str, corpus: str = "user") -> dict:
        return self.ingest_parsed(parse_file(data, name), corpus=corpus)

    def ingest_parsed(self, doc: ParsedDocument, corpus: str = "user") -> dict:
        if not doc.text.strip():
            return {"status": "rejected", "reason": "empty document", "source": doc.source}
        doc_id = make_doc_id(corpus, doc.source)
        chash = content_hash(doc.text)
        existing = self.store.get_document(doc_id)

        dup_of = self.dups.find_exact(chash)
        if dup_of is not None:
            return {"status": "duplicate", "doc_id": dup_of, "source": doc.source,
                    "reason": "identical content already indexed"}

        sim = simhash(doc.text)
        near = self.dups.find_near(sim, exclude=doc_id)
        if existing is not None:
            self._remove_indexes(doc_id)
        rec = self._index([(doc_id, doc, chash, sim)], corpus, previous={doc_id: existing})[0]
        result = {"status": "updated" if existing else "created", "doc_id": doc_id,
                  "version": rec.version, "num_chunks": rec.num_chunks, "title": rec.title}
        if near:
            result["near_duplicate_of"] = {"doc_id": near[0], "hamming_distance": near[1]}
        self.retriever.persist()
        return result

    def delete(self, doc_id: str) -> bool:
        if self.store.get_document(doc_id) is None:
            return False
        self._remove_indexes(doc_id)
        self.store.delete_document(doc_id)
        self.retriever.persist()
        return True

    def _remove_indexes(self, doc_id: str) -> None:
        self.store.delete_doc_chunks(doc_id)
        self.retriever.remove_doc(doc_id)
        self.dups.remove(doc_id)

    # ---------- bulk (dataset loading) ----------
    def bulk_ingest(self, docs: Iterable[ParsedDocument], corpus: str, batch_docs: int = 500) -> dict:
        stats = {"created": 0, "duplicates": 0, "chunks": 0}
        batch: list[tuple[str, ParsedDocument, str, int]] = []
        seen_ids: set[str] = set()

        def flush():
            if batch:
                recs = self._index(batch, corpus, previous={})
                stats["created"] += len(recs)
                stats["chunks"] += sum(r.num_chunks for r in recs)
                batch.clear()
                log.info("bulk: %s", stats)

        for doc in docs:
            doc_id = make_doc_id(corpus, doc.source)
            chash = content_hash(doc.text)
            if doc_id in seen_ids or self.dups.find_exact(chash):
                stats["duplicates"] += 1
                continue
            seen_ids.add(doc_id)
            # near-dup check is skipped in bulk mode for speed (exact dedup only)
            batch.append((doc_id, doc, chash, simhash(doc.text)))
            if len(batch) >= batch_docs:
                flush()
        flush()
        self.retriever.persist()
        return stats

    # ---------- core ----------
    def _index(self, items: list[tuple[str, ParsedDocument, str, int]], corpus: str,
               previous: dict[str, DocumentRecord | None]) -> list[DocumentRecord]:
        all_chunks: list[Chunk] = []
        records: list[DocumentRecord] = []
        ts = now()
        for doc_id, doc, chash, sim in items:
            pieces = self.chunker.split(doc.text)
            meta = {**doc.metadata, "corpus": corpus, "doc_type": doc.doc_type}
            for i, text in enumerate(pieces):
                all_chunks.append(Chunk(chunk_id=f"{doc_id}#{i}", doc_id=doc_id, text=text, position=i,
                                        title=doc.title, source=doc.source,
                                        metadata={"corpus": corpus, "doc_type": doc.doc_type,
                                                  "dates": doc.metadata.get("dates", [])}))
            prev = previous.get(doc_id)
            records.append(DocumentRecord(
                doc_id=doc_id, title=doc.title, source=doc.source, doc_type=doc.doc_type,
                content_hash=chash, simhash=to_signed(sim), num_chunks=len(pieces),
                version=(prev.version + 1) if prev else 1,
                created_at=prev.created_at if prev else ts, updated_at=ts, metadata=meta,
            ))
        # title is prepended for embedding: it disambiguates short chunks ("He was born in...")
        vectors = self.embedder.embed_documents([f"{c.title}\n{c.text}" for c in all_chunks])
        self.store.upsert_chunks(all_chunks, vectors)
        self.store.upsert_documents(records)
        self.retriever.index_chunks([(c.chunk_id, c.doc_id, f"{c.title} {c.text}") for c in all_chunks])
        for r in records:
            self.dups.add(r.doc_id, r.content_hash, r.simhash)
        return records
