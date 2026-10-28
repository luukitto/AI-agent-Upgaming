"""Hybrid retriever: dense (Qdrant HNSW) + keyword (own BM25) -> RRF fusion -> optional MMR re-rank."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np

from app.config import get_settings
from app.embeddings import Embedder
from app.models import SearchHit
from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import mmr, reciprocal_rank_fusion
from app.store.vector_store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"        # "dense" | "bm25" | "hybrid"
    top_k: int = 8
    candidate_k: int = 50
    rrf_k: int = 60
    use_mmr: bool = True
    mmr_lambda: float = 0.7
    dense_weight: float = 1.0
    bm25_weight: float = 1.0


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, bm25: BM25Index | None = None):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25 or BM25Index()
        self.lock = RLock()
        self.cache_path: str | None = None  # set by load_or_build_bm25; None = in-memory only (tests, eval)

    # ---------- keyword index lifecycle ----------
    def load_or_build_bm25(self, cache_path: str) -> None:
        """Load the BM25 pickle if it matches the store; otherwise rebuild from Qdrant (source of truth)."""
        self.cache_path = cache_path
        n_store = self.store.count_chunks()
        if Path(cache_path).exists():
            try:
                idx = BM25Index.load(cache_path)
                if len(idx) == n_store:
                    self.bm25 = idx
                    log.info("BM25 loaded from cache (%d chunks)", n_store)
                    return
                log.info("BM25 cache stale (%d vs %d), rebuilding", len(idx), n_store)
            except Exception:  # corrupt cache -> rebuild
                log.exception("BM25 cache unreadable, rebuilding")
        t = time.time()
        idx = BM25Index()
        for p in self.store.iter_all_chunks():
            idx.add(p["chunk_id"], p["doc_id"], f'{p.get("title", "")} {p["text"]}')
        idx.compact()  # everything into the compact frozen segment
        self.bm25 = idx
        idx.save(cache_path)
        log.info("BM25 rebuilt: %d chunks in %.1fs", len(idx), time.time() - t)

    # ---------- search ----------
    def search(self, query: str, cfg: RetrievalConfig | None = None,
               doc_ids: list[str] | None = None) -> list[SearchHit]:
        cfg = cfg or default_config()
        qvec = self.embedder.embed_query(query)
        rankings: dict[str, list[str]] = {}
        hits: dict[str, SearchHit] = {}
        vecs: dict[str, np.ndarray] = {}

        if cfg.mode in ("dense", "hybrid"):
            dense = self.store.dense_search(qvec, cfg.candidate_k, doc_ids=doc_ids, with_vectors=cfg.use_mmr)
            rankings["dense"] = [h.chunk_id for h, _ in dense]
            for h, v in dense:
                hits[h.chunk_id] = h
                if v is not None:
                    vecs[h.chunk_id] = v
        if cfg.mode in ("bm25", "hybrid"):
            with self.lock:
                kw = self.bm25.search(query, cfg.candidate_k, set(doc_ids) if doc_ids else None)
            rankings["bm25"] = [cid for cid, _ in kw]
            missing = [cid for cid, _ in kw if cid not in hits or (cfg.use_mmr and cid not in vecs)]
            for cid, (h, v) in self.store.get_chunks(missing, with_vectors=cfg.use_mmr).items():
                hits[cid] = h
                if v is not None:
                    vecs[cid] = v

        fused = reciprocal_rank_fusion(rankings, k=cfg.rrf_k,
                                       weights={"dense": cfg.dense_weight, "bm25": cfg.bm25_weight})
        fused = [f for f in fused if f[0] in hits][: max(cfg.candidate_k, cfg.top_k)]

        if cfg.use_mmr and fused and all(f[0] in vecs for f in fused):
            ids = [f[0] for f in fused]
            order = mmr(qvec, np.stack([vecs[i] for i in ids]), np.array([f[1] for f in fused]),
                        cfg.top_k, cfg.mmr_lambda)
            fused = [fused[i] for i in order]
        else:
            fused = fused[: cfg.top_k]

        out = []
        for cid, score, ranks in fused:
            h = hits[cid].model_copy()
            h.score = round(score, 6)
            h.debug = {"ranks": ranks}
            out.append(h)
        return out

    # ---------- index maintenance (called by ingestion) ----------
    def index_chunks(self, items: list[tuple[str, str, str]]) -> None:
        with self.lock:
            for chunk_id, doc_id, text in items:
                self.bm25.add(chunk_id, doc_id, text)

    def remove_doc(self, doc_id: str) -> None:
        with self.lock:
            self.bm25.remove_doc(doc_id)

    def persist(self) -> None:
        if self.cache_path:
            with self.lock:
                self.bm25.save(self.cache_path)


def default_config() -> RetrievalConfig:
    s = get_settings()
    return RetrievalConfig(top_k=s.top_k, candidate_k=s.candidate_k, rrf_k=s.rrf_k,
                           use_mmr=s.use_mmr, mmr_lambda=s.mmr_lambda)
