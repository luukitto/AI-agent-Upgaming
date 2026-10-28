"""Embedding providers (EMBED_PROVIDER): "local" (default) or "gemini".

LocalEmbedder: BAAI/bge-small-en-v1.5 (384-d) via fastembed/ONNX on CPU. No quota and no cost; it was chosen
because the Gemini free tier caps embedding requests per day (each text counts), far below the ~65k corpus chunks.
Queries get the BGE retrieval instruction prefix (fastembed `query_embed`), documents don't: asymmetric like Gemini.

Embedder (Gemini, gemini-embedding-001):

Uses the native `batchEmbedContents` endpoint (up to 100 texts per call) because it supports
task types: documents are embedded as RETRIEVAL_DOCUMENT and queries as RETRIEVAL_QUERY, which
measurably improves retrieval vs. one symmetric embedding. Vectors are truncated (Matryoshka) to
`embed_dim` and L2-normalised, as Google recommends for dimensions below 3072.

Handles: retries with backoff on 429/5xx/network errors, a client-side request throttle for
free-tier quotas, and an LRU cache for query embeddings.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from functools import lru_cache
from threading import Lock

import httpx
import numpy as np
from tenacity import (retry, retry_if_exception, stop_after_attempt, wait_random_exponential,
                      before_sleep_log)

from app.config import get_settings

log = logging.getLogger(__name__)

MAX_BATCH = 100  # API limit per batchEmbedContents call


class EmbeddingError(RuntimeError):
    """Raised when embedding fails permanently (after retries) or is misconfigured."""


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (408, 429, 500, 502, 503, 504)


class Embedder:
    def __init__(self, api_key: str, model: str, dim: int, base_url: str, batch_size: int = MAX_BATCH,
                 max_rpm: int = 0, max_retries: int = 6, timeout: float = 60.0, cache_size: int = 2048):
        if not api_key:
            raise EmbeddingError("No API key for embeddings. Set LLM_API_KEY (or EMBED_API_KEY) in .env.")
        self.model = model
        self.dim = dim
        self.batch_size = min(batch_size, MAX_BATCH)
        self.max_retries = max_retries
        # key goes in a header, not the URL, so it never shows up in logged request URLs
        self.http = httpx.Client(base_url=base_url, timeout=timeout, headers={"x-goog-api-key": api_key})
        self._min_interval = 60.0 / max_rpm if max_rpm > 0 else 0.0
        self._next_allowed = 0.0
        self._throttle_lock = Lock()
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._lock = Lock()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts = [self._embed_batch(texts[i:i + self.batch_size], "RETRIEVAL_DOCUMENT")
                 for i in range(0, len(texts), self.batch_size)]
        return np.vstack(parts)

    def embed_query(self, text: str) -> np.ndarray:
        """Queries use the RETRIEVAL_QUERY task type; results are LRU-cached."""
        with self._lock:
            if text in self._query_cache:
                self._query_cache.move_to_end(text)
                return self._query_cache[text]
        vec = self._embed_batch([text], "RETRIEVAL_QUERY")[0]
        with self._lock:
            self._query_cache[text] = vec
            if len(self._query_cache) > self._cache_size:
                self._query_cache.popitem(last=False)
        return vec

    def _throttle(self, n_texts: int) -> None:
        """Free tier counts every text in a batch as one request, so the budget is texts per minute."""
        if not self._min_interval:
            return
        with self._throttle_lock:
            wait = self._next_allowed - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._next_allowed = time.monotonic() + n_texts * self._min_interval

    def _embed_batch(self, texts: list[str], task_type: str) -> np.ndarray:
        body = {"requests": [{"model": f"models/{self.model}", "content": {"parts": [{"text": t}]},
                              "taskType": task_type, "outputDimensionality": self.dim} for t in texts]}

        @retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(self.max_retries + 1),
               wait=wait_random_exponential(multiplier=2, max=60), reraise=True,
               before_sleep=before_sleep_log(log, logging.WARNING))
        def _do() -> dict:
            self._throttle(len(texts))
            resp = self.http.post(f"models/{self.model}:batchEmbedContents", json=body)
            resp.raise_for_status()
            return resp.json()

        try:
            data = _do()
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code in (401, 403):
                raise EmbeddingError("Embedding authentication failed - check the API key") from e
            raise EmbeddingError(f"Embedding request failed ({code}): {e.response.text[:300]}") from e
        except Exception as e:
            raise EmbeddingError(f"Embedding call failed after retries: {e}") from e

        vecs = np.asarray([emb["values"] for emb in data["embeddings"]], dtype=np.float32)
        if vecs.shape != (len(texts), self.dim):
            raise EmbeddingError(f"Unexpected embedding shape {vecs.shape}, expected ({len(texts)}, {self.dim})")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-12)


class LocalEmbedder:
    def __init__(self, model: str, cache_dir: str | None = None, batch_size: int = 32):
        from fastembed import TextEmbedding

        self.model = model
        self.batch_size = batch_size
        self._m = TextEmbedding(model, cache_dir=cache_dir)
        self.dim = len(next(iter(self._m.embed(["probe"]))))
        self._lock = Lock()  # ONNX session is thread-safe, but serialising keeps CPU use predictable
        self.embed_query = lru_cache(maxsize=2048)(self._embed_query)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # batches are padded to their longest text: sorting by length is ~2.5x faster on CPU
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        with self._lock:
            vecs = list(self._m.embed([texts[i] for i in order], batch_size=self.batch_size))
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        out[order] = np.asarray(vecs, dtype=np.float32)
        return out

    def _embed_query(self, text: str) -> np.ndarray:
        with self._lock:
            return np.asarray(next(iter(self._m.query_embed(text))), dtype=np.float32)


@lru_cache
def get_embedder() -> Embedder | LocalEmbedder:
    s = get_settings()
    if s.embed_provider == "local":
        return LocalEmbedder(s.local_embed_model, s.embed_cache_dir)
    if s.embed_provider == "gemini":
        return Embedder(s.embed_api_key or s.llm_api_key, s.embed_model, s.embed_dim, s.embed_base_url,
                        s.embed_batch_size, s.embed_max_rpm)
    raise EmbeddingError(f"Unknown EMBED_PROVIDER {s.embed_provider!r} (use 'local' or 'gemini')")
