"""Own implementation of Okapi BM25 over an incremental inverted index (no search library).

score(D, Q) = sum_{t in Q} IDF(t) * tf(t,D) * (k1 + 1) / (tf(t,D) + k1 * (1 - b + b * |D| / avgdl))
IDF(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

Supports add/remove of chunks so the keyword index stays in sync with document CRUD.
"""
from __future__ import annotations

import heapq
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
STOPWORDS = frozenset("""
a an and are as at be but by for from has have he her his i in is it its of on or she that the
their them they this to was were which who whom will with what when where why how did does do
""".split())


def light_stem(tok: str) -> str:
    """Very small suffix stripper; enough to match payments/payment, failed/failing/fail."""
    for suf in ("ing", "ed", "es", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> list[str]:
    return [light_stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)  # term -> {internal id: tf}
        self.doc_len: dict[int, int] = {}
        self.ext_ids: dict[int, str] = {}       # internal id -> chunk_id
        self.int_ids: dict[str, int] = {}       # chunk_id -> internal id
        self.chunk_doc: dict[int, str] = {}     # internal id -> doc_id
        self.chunk_terms: dict[int, list[str]] = {}
        self.total_len = 0
        self._next = 0

    def __len__(self) -> int:
        return len(self.doc_len)

    @property
    def avgdl(self) -> float:
        return self.total_len / len(self.doc_len) if self.doc_len else 0.0

    def add(self, chunk_id: str, doc_id: str, text: str) -> None:
        if chunk_id in self.int_ids:
            self.remove_chunk(chunk_id)
        iid = self._next
        self._next += 1
        tf = Counter(tokenize(text))
        for term, c in tf.items():
            self.postings[term][iid] = c
        length = sum(tf.values())
        self.doc_len[iid] = length
        self.total_len += length
        self.ext_ids[iid] = chunk_id
        self.int_ids[chunk_id] = iid
        self.chunk_doc[iid] = doc_id
        self.chunk_terms[iid] = list(tf)

    def remove_chunk(self, chunk_id: str) -> None:
        iid = self.int_ids.pop(chunk_id, None)
        if iid is None:
            return
        for term in self.chunk_terms.pop(iid, []):
            plist = self.postings.get(term)
            if plist is not None:
                plist.pop(iid, None)
                if not plist:
                    del self.postings[term]
        self.total_len -= self.doc_len.pop(iid, 0)
        self.ext_ids.pop(iid, None)
        self.chunk_doc.pop(iid, None)

    def remove_doc(self, doc_id: str) -> None:
        for iid in [i for i, d in self.chunk_doc.items() if d == doc_id]:
            self.remove_chunk(self.ext_ids[iid])

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        n = len(self.doc_len)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 50, doc_ids: set[str] | None = None) -> list[tuple[str, float]]:
        terms = set(tokenize(query))
        if not terms or not self.doc_len:
            return []
        scores: dict[int, float] = defaultdict(float)
        avgdl = self.avgdl
        k1, b = self.k1, self.b
        # process rare (high-IDF) terms first; skip extremely common terms on huge postings lists
        for term in sorted(terms, key=lambda t: len(self.postings.get(t, ()))):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf(term)
            if idf < 0.05 and len(plist) > 20000:
                continue
            for iid, tf in plist.items():
                norm = tf + k1 * (1 - b + b * self.doc_len[iid] / avgdl)
                scores[iid] += idf * tf * (k1 + 1) / norm
        items = scores.items()
        if doc_ids is not None:
            items = [(i, s) for i, s in items if self.chunk_doc[i] in doc_ids]
        top = heapq.nlargest(k, items, key=lambda x: x[1])
        return [(self.ext_ids[i], s) for i, s in top]

    # ---------- persistence ----------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = self.__dict__.copy()
        state["postings"] = dict(self.postings)
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            state = pickle.load(f)
        idx = cls()
        idx.__dict__.update(state)
        idx.postings = defaultdict(dict, state["postings"])
        return idx
