"""End-to-end evaluation of generated answers on the golden set.

Per question, cheap mechanical checks first, the judge only where judgment is needed:

    answerable question
      answer.error            -> outcome "error"
      covered == False        -> "false_refusal"  (score 0, no judge needed)
      covered == True         -> judge -> "answered" (score 1 / 0.5 / 0)
      + context_recall        gold sections present in the LLM's context
      + citation_precision    share of verified citations whose chunk covers a gold section
    unanswerable question
      covered == False        -> "correct_refusal" (score 1)
      covered == True         -> "hallucinated"   (score 0)

The primary metric is `answer_score`: the mean score over answerable questions.
"""

import hashlib
import statistics
from collections import Counter
from typing import Literal

from regrag.evaluation.golden import GoldItem, gold_text
from regrag.evaluation.judge import JudgeError, JudgeInput, JudgeVerdict, LLMJudge
from regrag.evaluation.metrics import matches, recall_at_k
from regrag.generation.models import Answer
from regrag.generation.pipeline import RAGPipeline
from regrag.ingestion.models import Model, ParsedBlock

Outcome = Literal[
    "answered", "false_refusal", "correct_refusal", "hallucinated", "error", "judge_error"
]


class GenQuestionResult(Model):
    id: str
    type: str
    lang: str
    split: str
    question: str
    answerable: bool
    outcome: Outcome
    score: float | None  # None when it could not be scored (judge error)
    covered: bool
    grounded: bool
    answer: str
    answer_hash: str
    citations: list[dict]
    citations_verified: int
    citation_precision: float | None  # None: no verified citation, or unanswerable
    context_recall: float | None
    judge: JudgeVerdict | None = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    timings_ms: dict[str, float] = {}


def evidence_for(item: GoldItem, blocks_by_doc: dict[str, list[ParsedBlock]]) -> str:
    return "\n\n".join(
        f"[{ref.label()}]\n{gold_text(ref, blocks_by_doc[ref.doc_id])}" for ref in item.gold
    )


def judge_input(item: GoldItem, answer: Answer, evidence: str) -> JudgeInput:
    return JudgeInput(
        question=item.question,
        reference_answer=item.answer or "",
        evidence=evidence,
        system_answer=answer.text,
        quotes=[c.quote for c in answer.citations if c.verified],
    )


def score_answer(
    item: GoldItem,
    answer: Answer,
    judge: LLMJudge | None,
    blocks_by_doc: dict[str, list[ParsedBlock]],
) -> GenQuestionResult:
    answerable = bool(item.gold)
    by_label = {c.label: c for c in answer.context}
    verified = [c for c in answer.citations if c.verified]

    context_recall = citation_precision = None
    if answerable:
        context_recall = recall_at_k(answer.context, item.gold, len(answer.context) or 1)
        if verified:
            hits = sum(
                1
                for c in verified
                if c.chunk_label in by_label
                and any(matches(by_label[c.chunk_label], ref) for ref in item.gold)
            )
            citation_precision = hits / len(verified)

    verdict: JudgeVerdict | None = None
    error = answer.error
    if answer.error:
        outcome, score = "error", 0.0
    elif not answerable:
        outcome, score = ("hallucinated", 0.0) if answer.covered else ("correct_refusal", 1.0)
    elif not answer.covered:
        outcome, score = "false_refusal", 0.0
    elif judge is None:
        outcome, score = "answered", None
    else:
        try:
            verdict = judge.judge(judge_input(item, answer, evidence_for(item, blocks_by_doc)))
            outcome, score = "answered", verdict.score
        except JudgeError as e:
            outcome, score, error = "judge_error", None, str(e)

    return GenQuestionResult(
        id=item.id,
        type=item.type,
        lang=item.lang,
        split=item.split,
        question=item.question,
        answerable=answerable,
        outcome=outcome,
        score=score,
        covered=answer.covered,
        grounded=answer.grounded,
        answer=answer.text,
        answer_hash=hashlib.sha256(answer.text.encode()).hexdigest()[:16],
        citations=[c.model_dump() for c in answer.citations],
        citations_verified=len(verified),
        citation_precision=citation_precision,
        context_recall=context_recall,
        judge=verdict,
        error=error,
        prompt_tokens=answer.usage.prompt_tokens,
        completion_tokens=answer.usage.completion_tokens,
        timings_ms=answer.timings_ms,
    )


def evaluate(
    items: list[GoldItem],
    pipeline: RAGPipeline,
    judge: LLMJudge | None,
    blocks_by_doc: dict[str, list[ParsedBlock]],
    progress=None,
) -> tuple[list[GenQuestionResult], dict[str, Answer]]:
    """Returns the scored results and the raw answers (kept for re-judging and labelling)."""
    results, answers = [], {}
    for n, item in enumerate(items, start=1):
        if progress:
            progress(n, len(items), item.id)
        answers[item.id] = pipeline.ask(item.question)
        results.append(score_answer(item, answers[item.id], judge, blocks_by_doc))
    return results, answers


# --- summary ---------------------------------------------------------------------------


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 3) if values else None


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q * len(values)))], 1)


class GenSummary(Model):
    n: int
    answerable: int
    answer_score: float | None  # PRIMARY: mean score over answerable questions
    verdicts: dict[str, int]
    overstated_rate: float | None  # among judged answers
    missing_condition_rate: float | None
    contradicts_rate: float | None
    false_refusal_rate: float | None  # answerable questions refused
    refusal_accuracy: float | None  # unanswerable questions correctly refused
    grounded_rate: float | None  # covered answers with >= 1 verified quote
    citation_verified_rate: float | None  # all citations that were found verbatim
    citation_precision: float | None
    context_recall: float | None
    errors: int
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    llm_ms_p50: float | None
    completion_tokens_mean: float | None


def summarize(results: list[GenQuestionResult]) -> GenSummary:
    ans = [r for r in results if r.answerable]
    unans = [r for r in results if not r.answerable]
    judged = [r for r in ans if r.judge is not None]
    covered = [r for r in results if r.covered and not r.error]
    all_cits = [c for r in results for c in r.citations]
    total = [r.timings_ms.get("total_ms", 0) + r.timings_ms.get("llm_ms", 0) for r in results]

    def rate(rows, pred):
        return round(sum(1 for r in rows if pred(r)) / len(rows), 3) if rows else None

    return GenSummary(
        n=len(results),
        answerable=len(ans),
        answer_score=_mean(r.score for r in ans),
        verdicts=dict(Counter(r.judge.verdict for r in judged)),
        overstated_rate=rate(judged, lambda r: r.judge.overstated),
        missing_condition_rate=rate(judged, lambda r: r.judge.missing_condition),
        contradicts_rate=rate(judged, lambda r: r.judge.contradicts_reference),
        false_refusal_rate=rate(ans, lambda r: r.outcome == "false_refusal"),
        refusal_accuracy=rate(unans, lambda r: r.outcome == "correct_refusal"),
        grounded_rate=rate(covered, lambda r: r.grounded),
        citation_verified_rate=(
            round(sum(c["verified"] for c in all_cits) / len(all_cits), 3) if all_cits else None
        ),
        citation_precision=_mean(r.citation_precision for r in ans),
        context_recall=_mean(r.context_recall for r in ans),
        errors=sum(1 for r in results if r.outcome in ("error", "judge_error")),
        latency_ms_p50=_pct(total, 0.5),
        latency_ms_p95=_pct(total, 0.95),
        llm_ms_p50=_pct([r.timings_ms.get("llm_ms", 0) for r in results], 0.5),
        completion_tokens_mean=_mean(r.completion_tokens for r in results),
    )


def judge_consistency(
    results: list[GenQuestionResult],
    items: dict[str, GoldItem],
    judge: LLMJudge,
    blocks_by_doc: dict[str, list[ParsedBlock]],
    answers: dict[str, Answer],
    sample: int,
) -> float | None:
    """Re-judge a sample WITHOUT the cache: how often does the verdict stay the same?"""
    judged = [r for r in results if r.judge is not None][:sample]
    if not judged:
        return None
    same = 0
    for r in judged:
        item = items[r.id]
        again = judge.judge(
            judge_input(item, answers[r.id], evidence_for(item, blocks_by_doc)), use_cache=False
        )
        same += again.verdict == r.judge.verdict
    return round(same / len(judged), 3)
