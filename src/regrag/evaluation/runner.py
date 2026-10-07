"""Run the golden set through the retriever and measure every stage.

One retrieval per question yields all stages at once (the result carries the
per-retriever ranks):

    dense     pool sorted by dense rank           (dense search alone)
    sparse    pool sorted by sparse rank          (sparse search alone)
    fused     pool in RRF order                   (hybrid, no reranker)
    reranked  rerank candidates by rerank score   (the production output)

Each answerable question also gets a diagnosis naming the stage that lost it:

    ok                  relevant hit in the final top-5
    not_retrieved       no retriever returned anything relevant
    below_cutoff        retrieved, but fused rank beyond the rerank candidates
    lost_in_rerank      among the candidates, but the reranker pushed it below top-5
    missed_without_rerank  (only with rerank disabled) not in the fused top-5
"""

import logging
import statistics
from collections import defaultdict
from typing import Literal

from regrag.evaluation.golden import GoldItem
from regrag.evaluation.metrics import first_relevant_rank, recall_at_k, reciprocal_rank
from regrag.ingestion.models import Model
from regrag.retrieval.models import Hit, RetrievalResult
from regrag.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

STAGES = ("dense", "sparse", "fused", "reranked")
KS = (5, 10, 20)
FINAL_K = 5

Diagnosis = Literal[
    "ok", "not_retrieved", "below_cutoff", "lost_in_rerank", "missed_without_rerank", "n/a"
]


class StageScore(Model):
    recall: dict[int, float]  # k -> recall@k
    rr: float  # reciprocal rank within top-20
    first_rank: int | None


class QuestionResult(Model):
    id: str
    type: str
    lang: str
    split: str
    question: str
    stages: dict[str, StageScore] = {}  # empty for unanswerable
    diagnosis: Diagnosis = "n/a"
    top1_rerank: float | None  # best rerank score: the "is anything relevant?" signal
    top_hits: list[str]  # "scope / sections" of the final top-5, for reading failures


class StageSummary(Model):
    recall: dict[int, float]
    mrr: float


class ThresholdAnalysis(Model):
    answerable_top1: list[float]
    unanswerable_top1: list[float]
    best_threshold: float | None
    best_balanced_accuracy: float | None


def stage_rankings(result: RetrievalResult) -> dict[str, list[Hit]]:
    pool = result.pool
    reranked = result.candidates
    if any(h.rerank_score is not None for h in reranked):
        reranked = sorted(reranked, key=lambda h: -(h.rerank_score or 0.0))
    return {
        "dense": sorted((h for h in pool if h.dense_rank), key=lambda h: h.dense_rank or 0),
        "sparse": sorted((h for h in pool if h.sparse_rank), key=lambda h: h.sparse_rank or 0),
        "fused": pool,
        "reranked": reranked,
    }


def _diagnose(rankings: dict[str, list[Hit]], item: GoldItem, n_candidates: int, reranked: bool):
    fused_rank = first_relevant_rank(rankings["fused"], item.gold)
    final_rank = first_relevant_rank(rankings["reranked"], item.gold)
    if final_rank is not None and final_rank <= FINAL_K:
        return "ok"
    if fused_rank is None:
        return "not_retrieved"
    if fused_rank > n_candidates:
        return "below_cutoff"
    return "lost_in_rerank" if reranked else "missed_without_rerank"


def _hit_label(h: Hit) -> str:
    secs = f"{h.sections[0]}..{h.sections[-1]}" if h.sections else f"p.{h.page_start}"
    return f"{h.scope} / {secs}"


def evaluate_question(retriever: HybridRetriever, item: GoldItem, rerank: bool) -> QuestionResult:
    result = retriever.search(item.question, rerank=rerank)
    rankings = stage_rankings(result)
    top1 = result.hits[0].rerank_score if result.hits else None
    out = QuestionResult(
        id=item.id,
        type=item.type,
        lang=item.lang,
        split=item.split,
        question=item.question,
        top1_rerank=top1,
        top_hits=[_hit_label(h) for h in result.hits[:FINAL_K]],
    )
    if item.gold:
        for stage, ranked in rankings.items():
            out.stages[stage] = StageScore(
                recall={k: recall_at_k(ranked, item.gold, k) for k in KS},
                rr=reciprocal_rank(ranked, item.gold, max(KS)),
                first_rank=first_relevant_rank(ranked, item.gold),
            )
        out.diagnosis = _diagnose(rankings, item, retriever.cfg.rerank_candidates, rerank)
    return out


def summarize(results: list[QuestionResult]) -> dict[str, StageSummary]:
    answerable = [r for r in results if r.stages]
    summary = {}
    for stage in STAGES:
        if not answerable:
            break
        summary[stage] = StageSummary(
            recall={k: statistics.fmean(r.stages[stage].recall[k] for r in answerable) for k in KS},
            mrr=statistics.fmean(r.stages[stage].rr for r in answerable),
        )
    return summary


def summarize_by(results: list[QuestionResult], key: str, stage: str) -> dict[str, StageSummary]:
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        if r.stages:
            groups[getattr(r, key)].append(r)
    return {name: summarize(rs)[stage] for name, rs in sorted(groups.items())}


def threshold_analysis(results: list[QuestionResult]) -> ThresholdAnalysis:
    """Can the top rerank score tell answerable from unanswerable questions?

    Sweeps every observed score as a threshold ("answer if top1 >= t") and picks
    the one with the best balanced accuracy (mean of the two per-class accuracies,
    so the smaller unanswerable class counts as much as the larger one).
    """
    pos = sorted(r.top1_rerank for r in results if r.stages and r.top1_rerank is not None)
    neg = sorted(r.top1_rerank for r in results if not r.stages and r.top1_rerank is not None)
    best_t = best_acc = None
    if pos and neg:
        for t in sorted(set(pos + neg)):
            tpr = sum(s >= t for s in pos) / len(pos)
            tnr = sum(s < t for s in neg) / len(neg)
            acc = (tpr + tnr) / 2
            if best_acc is None or acc > best_acc:
                best_t, best_acc = t, acc
    return ThresholdAnalysis(
        answerable_top1=pos,
        unanswerable_top1=neg,
        best_threshold=best_t,
        best_balanced_accuracy=best_acc,
    )
