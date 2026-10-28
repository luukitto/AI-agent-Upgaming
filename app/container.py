"""Composition root: builds and wires the singletons used by the API, scripts and eval."""
from __future__ import annotations

import logging
from functools import cached_property

from app.config import get_settings
from app.embeddings import get_embedder
from app.ingestion.chunking import get_chunker
from app.ingestion.pipeline import IngestionService
from app.retrieval.hybrid import HybridRetriever
from app.store.vector_store import get_store

log = logging.getLogger(__name__)


class Container:
    def __init__(self, chunk_strategy: str | None = None):
        self.settings = get_settings()
        self.store = get_store()
        self.embedder = get_embedder()
        self.retriever = HybridRetriever(self.store, self.embedder)
        self.retriever.load_or_build_bm25(self.settings.bm25_cache_path)
        s = self.settings
        self.ingestion = IngestionService(self.store, self.embedder, self.retriever,
                                          get_chunker(chunk_strategy or s.chunk_strategy, s.chunk_size, s.chunk_overlap))
        self.ingestion.load_dup_index()

    @cached_property
    def investigator(self):
        # lazy: the app can still serve search/documents if the LLM key is missing
        from app.agent.investigator import Investigator
        from app.llm.client import get_llm
        return Investigator(get_llm(), self.retriever, self.store)


_container: Container | None = None


def get_container() -> Container:
    global _container
    if _container is None:
        _container = Container()
    return _container
