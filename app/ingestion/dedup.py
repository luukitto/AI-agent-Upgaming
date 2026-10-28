"""Duplicate detection.

* Exact duplicates: SHA-256 of normalised text.
* Near duplicates:  64-bit SimHash over word 3-shingles (Charikar 2002). Two documents whose
  SimHashes differ in <= `max_distance` bits are considered near-duplicates
  (e.g. the same report re-exported with a changed footer).
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")
MASK64 = (1 << 64) - 1


def normalize_for_hash(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode()).hexdigest()


def simhash(text: str, shingle: int = 3) -> int:
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0
    grams = [" ".join(tokens[i:i + shingle]) for i in range(max(1, len(tokens) - shingle + 1))]
    weights = [0] * 64
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if (h >> bit) & 1 else -1
    value = 0
    for bit in range(64):
        if weights[bit] > 0:
            value |= 1 << bit
    return value


def hamming(a: int, b: int) -> int:
    return bin((a ^ b) & MASK64).count("1")


def to_signed(v: int) -> int:
    """Qdrant payload ints are signed 64-bit."""
    return v - (1 << 64) if v >= 1 << 63 else v


def to_unsigned(v: int) -> int:
    return v + (1 << 64) if v < 0 else v


class DuplicateIndex:
    """In-memory index of document fingerprints for O(1) exact and vectorised near-dup lookup."""

    def __init__(self, max_distance: int = 3):
        self.max_distance = max_distance
        self.by_hash: dict[str, str] = {}
        self.doc_ids: list[str] = []
        self.sims: list[int] = []

    def add(self, doc_id: str, chash: str, sim: int) -> None:
        self.by_hash[chash] = doc_id
        self.doc_ids.append(doc_id)
        self.sims.append(to_unsigned(sim))

    def remove(self, doc_id: str) -> None:
        self.by_hash = {h: d for h, d in self.by_hash.items() if d != doc_id}
        keep = [i for i, d in enumerate(self.doc_ids) if d != doc_id]
        self.doc_ids = [self.doc_ids[i] for i in keep]
        self.sims = [self.sims[i] for i in keep]

    def find_exact(self, chash: str) -> str | None:
        return self.by_hash.get(chash)

    def find_near(self, sim: int, exclude: str | None = None) -> tuple[str, int] | None:
        if not self.sims:
            return None
        arr = np.array(self.sims, dtype=np.uint64)
        xor = np.bitwise_xor(arr, np.uint64(to_unsigned(sim)))
        # popcount via byte view
        dist = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        order = np.argsort(dist)
        for i in order[:5]:
            if dist[i] > self.max_distance:
                break
            if self.doc_ids[i] != exclude:
                return self.doc_ids[i], int(dist[i])
        return None
