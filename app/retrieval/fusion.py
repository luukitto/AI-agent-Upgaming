"""Own rank-fusion and re-ranking algorithms.

* reciprocal_rank_fusion: RRF (Cormack et al. 2009). score(d) = sum_r w_r / (k + rank_r(d)).
  Rank-based, so it needs no score normalisation between BM25 (unbounded) and cosine ([-1, 1]).
* mmr: Maximal Marginal Relevance (Carbonell & Goldstein 1998). Greedily picks items that are
  relevant to the query but dissimilar to already-selected items -> fewer near-identical chunks,
  more distinct sources, which matters for multi-document questions.
"""
from __future__ import annotations

import numpy as np


def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int = 60,
                           weights: dict[str, float] | None = None) -> list[tuple[str, float, dict[str, int]]]:
    """rankings: retriever name -> ordered list of ids. Returns (id, score, {retriever: rank})."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for rank, item in enumerate(ids, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
            ranks.setdefault(item, {})[name] = rank
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(i, s, ranks[i]) for i, s in ordered]


def mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, relevance: np.ndarray, k: int,
        lam: float = 0.7) -> list[int]:
    """Return indices of selected candidates.

    relevance: prior relevance per candidate (e.g. fused score), min-max normalised here so it is
    comparable with cosine similarity.
    """
    n = len(cand_vecs)
    if n == 0:
        return []
    k = min(k, n)
    rel = relevance.astype(np.float32)
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
    vecs = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9)
    sim = vecs @ vecs.T
    selected: list[int] = [int(np.argmax(rel))]
    remaining = set(range(n)) - set(selected)
    while len(selected) < k and remaining:
        rem = np.array(sorted(remaining))
        redundancy = sim[np.ix_(rem, selected)].max(axis=1)
        mmr_score = lam * rel[rem] - (1 - lam) * redundancy
        best = int(rem[int(np.argmax(mmr_score))])
        selected.append(best)
        remaining.remove(best)
    return selected
