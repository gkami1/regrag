"""Retrieval metrics against section-level gold labels.

A retrieved hit is relevant to a gold reference if it is in the same document and
scope and covers the gold section or one of its subsections (or, for unnumbered
content, the gold page). Because the match is on sections, not chunk ids, the
same golden set evaluates any chunking.

recall@k  share of a question's gold references covered by the top k hits
          (a comparison question with 2 gold sections finding 1 scores 0.5)
mrr       1 / rank of the first relevant hit (0 if none): how high the first
          useful result is
"""

from collections.abc import Sequence
from typing import Protocol

from regrag.evaluation.golden import GoldRef


class HitLike(Protocol):
    doc_id: str
    scope: str
    sections: list[str]
    page_start: int
    page_end: int


def matches(hit: HitLike, ref: GoldRef) -> bool:
    if hit.doc_id != ref.doc_id or hit.scope != ref.scope:
        return False
    if ref.section is None:
        return ref.page is not None and hit.page_start <= ref.page <= hit.page_end
    return any(s == ref.section or s.startswith(ref.section + ".") for s in hit.sections)


def recall_at_k(ranked: Sequence[HitLike], gold: list[GoldRef], k: int) -> float:
    if not gold:
        raise ValueError("recall is undefined for questions without gold (unanswerable)")
    top = ranked[:k]
    found = sum(1 for ref in gold if any(matches(h, ref) for h in top))
    return found / len(gold)


def reciprocal_rank(ranked: Sequence[HitLike], gold: list[GoldRef], k: int) -> float:
    for rank, hit in enumerate(ranked[:k], start=1):
        if any(matches(hit, ref) for ref in gold):
            return 1.0 / rank
    return 0.0


def first_relevant_rank(ranked: Sequence[HitLike], gold: list[GoldRef]) -> int | None:
    for rank, hit in enumerate(ranked, start=1):
        if any(matches(hit, ref) for ref in gold):
            return rank
    return None
