"""Reciprocal Rank Fusion.

    rrf(d) = sum over retrievers r of  weight_r / (k + rank_r(d))      (ranks are 1-based)

Uses ranks only, never raw scores: dense cosine (~0.6) and sparse dot product
(~0.2) live on different, query-dependent scales, and normalising them is
fragile. A document ranked well by several retrievers wins; one found by a
single retriever still gets credit. `k` dampens the advantage of the very top
ranks: with k=60, rank 1 vs rank 2 differ by ~1.6%, so agreement between
retrievers matters more than being first in one of them.
"""


def rrf(
    rankings: dict[str, list[str]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one list of (id, score), best first.

    Ties are broken by the best single rank, then by id, so the output is
    deterministic (important for reproducible evaluation).
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for name, ids in rankings.items():
        weight = weights.get(name, 1.0)
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    return sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))
