"""Own implementation of Okapi BM25 over a compact, incremental inverted index (no search library).

score(D, Q) = sum_{t in Q} IDF(t) * tf(t,D) * (k1 + 1) / (tf(t,D) + k1 * (1 - b + b * |D| / avgdl))
IDF(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

Storage is split in two segments, the same idea Lucene uses:
* frozen: CSR arrays - for term t, postings live in docs[offsets[t]:offsets[t+1]] (int32 chunk ids) and
  tfs[...] (uint16). ~6 bytes per posting, vs ~100 bytes for a Python dict entry: the 2.7M postings of the
  full corpus take ~40 MB instead of ~300 MB, which is what lets the app run on a 512 MB free instance.
* delta: plain dicts for chunks added since the last compaction (uploads, bulk loading in progress).
Deleting a chunk only marks it dead (tombstone); compact() merges delta into frozen and drops dead chunks.
Until then N, df and avgdl still count dead chunks (as in Lucene) - a small, temporary IDF skew.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
STOPWORDS = frozenset("""
a an and are as at be but by for from has have he her his i in is it its of on or she that the
their them they this to was were which who whom will with what when where why how did does do
""".split())

FORMAT_VERSION = 2          # bump when the pickled layout changes; old caches are then rebuilt
COMPACT_DELTA_CHUNKS = 5000  # merge the delta segment once it holds this many chunks
COMPACT_DEAD_RATIO = 0.1     # ... or once this share of chunks is dead
MAX_TF = np.iinfo(np.uint16).max


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
        self.format_version = FORMAT_VERSION
        # per internal id (position = id); dead ids keep their slot until compaction
        self.ext_ids: list[str] = []            # internal id -> chunk_id
        self.chunk_doc: list[str] = []          # internal id -> doc_id
        self.doc_len: list[int] = []
        self.total_len = 0                      # over all slots, dead included (see module doc)
        self.int_ids: dict[str, int] = {}       # chunk_id -> internal id (live chunks only)
        self.dead: set[int] = set()
        # frozen segment (CSR)
        self.vocab: dict[str, int] = {}         # term -> row in offsets
        self.offsets = np.zeros(1, dtype=np.int64)
        self.post_docs = np.zeros(0, dtype=np.int32)
        self.post_tfs = np.zeros(0, dtype=np.uint16)
        # delta segment
        self.delta: dict[str, dict[int, int]] = defaultdict(dict)
        self.delta_chunks = 0
        self._dl_cache: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.int_ids)

    @property
    def num_terms(self) -> int:
        return len(self.vocab) + sum(1 for t in self.delta if t not in self.vocab)

    @property
    def avgdl(self) -> float:
        return self.total_len / len(self.doc_len) if self.doc_len else 0.0

    # ---------- updates ----------
    def add(self, chunk_id: str, doc_id: str, text: str) -> None:
        if chunk_id in self.int_ids:
            self.remove_chunk(chunk_id)
        iid = len(self.ext_ids)
        tf = Counter(tokenize(text))
        for term, c in tf.items():
            self.delta[term][iid] = min(c, MAX_TF)
        length = sum(tf.values())
        self.ext_ids.append(chunk_id)
        self.chunk_doc.append(doc_id)
        self.doc_len.append(length)
        self.total_len += length
        self.int_ids[chunk_id] = iid
        self.delta_chunks += 1
        if self.delta_chunks >= COMPACT_DELTA_CHUNKS:
            self.compact()

    def remove_chunk(self, chunk_id: str) -> None:
        iid = self.int_ids.pop(chunk_id, None)
        if iid is None:
            return
        self.dead.add(iid)
        if len(self.dead) > COMPACT_DEAD_RATIO * len(self.ext_ids):
            self.compact()

    def remove_doc(self, doc_id: str) -> None:
        for chunk_id in [c for c, i in self.int_ids.items() if self.chunk_doc[i] == doc_id]:
            self.remove_chunk(chunk_id)

    def compact(self) -> None:
        """Merge the delta segment into the frozen CSR arrays and drop dead chunks (ids are renumbered)."""
        n = len(self.ext_ids)
        alive = np.ones(n, dtype=bool)
        if self.dead:
            alive[np.fromiter(self.dead, dtype=np.int64, count=len(self.dead))] = False
        remap = np.cumsum(alive, dtype=np.int64) - 1    # old id -> new id (valid where alive)

        # flatten both segments into parallel (term, doc, tf) arrays
        vocab = dict(self.vocab)
        terms = [np.repeat(np.arange(len(self.vocab), dtype=np.int32), np.diff(self.offsets))]
        docs, tfs = [self.post_docs], [self.post_tfs]
        for term, plist in self.delta.items():
            tid = vocab.setdefault(term, len(vocab))
            terms.append(np.full(len(plist), tid, dtype=np.int32))
            docs.append(np.fromiter(plist.keys(), dtype=np.int32, count=len(plist)))
            tfs.append(np.fromiter(plist.values(), dtype=np.uint16, count=len(plist)))
        t, d, f = np.concatenate(terms), np.concatenate(docs), np.concatenate(tfs)
        keep = alive[d]
        t, d, f = t[keep], remap[d[keep]].astype(np.int32), f[keep]

        # drop terms that lost all their postings, then sort postings by (term, doc)
        counts = np.bincount(t, minlength=len(vocab))
        used = counts > 0
        tid_map = np.cumsum(used) - 1
        self.vocab = {term: int(tid_map[tid]) for term, tid in vocab.items() if used[tid]}
        t = tid_map[t]
        order = np.lexsort((d, t))
        self.post_docs, self.post_tfs = d[order], f[order]
        self.offsets = np.concatenate([[0], np.cumsum(counts[used])]).astype(np.int64)

        keep_ids = np.flatnonzero(alive)
        self.ext_ids = [self.ext_ids[i] for i in keep_ids]
        self.chunk_doc = [self.chunk_doc[i] for i in keep_ids]
        self.doc_len = [self.doc_len[i] for i in keep_ids]
        self.total_len = sum(self.doc_len)
        self.int_ids = {c: i for i, c in enumerate(self.ext_ids)}
        self.dead.clear()
        self.delta = defaultdict(dict)
        self.delta_chunks = 0
        self._dl_cache = None

    # ---------- search ----------
    def _postings(self, term: str) -> tuple[np.ndarray, np.ndarray]:
        """All postings of a term across both segments, dead chunks included."""
        tid = self.vocab.get(term)
        if tid is not None:
            s, e = self.offsets[tid], self.offsets[tid + 1]
            docs, tfs = self.post_docs[s:e], self.post_tfs[s:e]
        else:
            docs, tfs = self.post_docs[:0], self.post_tfs[:0]
        plist = self.delta.get(term)
        if plist:
            docs = np.concatenate([docs, np.fromiter(plist.keys(), dtype=np.int32, count=len(plist))])
            tfs = np.concatenate([tfs, np.fromiter(plist.values(), dtype=np.uint16, count=len(plist))])
        return docs, tfs

    def _doc_lens(self) -> np.ndarray:
        if self._dl_cache is None or len(self._dl_cache) != len(self.doc_len):
            self._dl_cache = np.asarray(self.doc_len, dtype=np.float64)
        return self._dl_cache

    def idf(self, term: str) -> float:
        df = len(self._postings(term)[0])
        n = len(self.doc_len)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 50, doc_ids: set[str] | None = None) -> list[tuple[str, float]]:
        terms = set(tokenize(query))
        if not terms or not self.int_ids:
            return []
        n = len(self.doc_len)
        dl, avgdl = self._doc_lens(), self.avgdl
        k1, b = self.k1, self.b
        scores = np.zeros(n, dtype=np.float64)    # dense accumulator: 65k chunks = 0.5 MB
        for term in terms:
            docs, tfs = self._postings(term)
            df = len(docs)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            if idf < 0.05 and df > 20000:   # skip near-stopwords with huge postings lists
                continue
            tf = tfs.astype(np.float64)
            # a term's postings hold each chunk once, so fancy-index += is safe here
            scores[docs] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[docs] / avgdl))
        if self.dead:
            scores[np.fromiter(self.dead, dtype=np.int64, count=len(self.dead))] = 0.0
        cand = np.flatnonzero(scores > 0)
        if doc_ids is not None:
            cand = np.array([i for i in cand if self.chunk_doc[i] in doc_ids], dtype=np.int64)
        if len(cand) > k:
            cand = cand[np.argpartition(-scores[cand], k - 1)[:k]]
        cand = cand[np.argsort(-scores[cand], kind="stable")]
        return [(self.ext_ids[i], float(scores[i])) for i in cand]

    # ---------- persistence ----------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = self.__dict__.copy()
        state["delta"] = dict(self.delta)
        state["_dl_cache"] = None
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            state = pickle.load(f)
        if state.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"BM25 cache format {state.get('format_version')} != {FORMAT_VERSION}")
        idx = cls()
        idx.__dict__.update(state)
        idx.delta = defaultdict(dict, state["delta"])
        return idx
