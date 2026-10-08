"""Human labels for calibrating the LLM judge.

A label belongs to one exact answer: (question id, answer hash). If the system
later produces a different answer, the old label is not silently compared to it.
Labels are a curated asset, versioned in git like the golden set.
"""

import hashlib
import json
from pathlib import Path

from regrag.evaluation.generation_eval import GenQuestionResult
from regrag.evaluation.judge import Verdict
from regrag.ingestion.models import Model

LABELS_PATH = Path("evals/judge_labels.jsonl")


class HumanLabel(Model):
    id: str
    answer_hash: str
    verdict: Verdict
    overstated: bool = False
    missing_condition: bool = False
    notes: str = ""


def load_labels(path: Path = LABELS_PATH) -> list[HumanLabel]:
    if not path.exists():
        return []
    return [
        HumanLabel.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_label(label: HumanLabel, path: Path = LABELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(label.model_dump(), ensure_ascii=False) + "\n")


def sample_for_labelling(results: list[GenQuestionResult], n: int) -> list[GenQuestionResult]:
    """Deterministic sample of judged answers (stable across runs: ordered by hashed id)."""
    judged = [r for r in results if r.judge is not None]
    return sorted(judged, key=lambda r: hashlib.sha256(r.id.encode()).hexdigest())[:n]


class Agreement(Model):
    n: int
    verdict: float | None  # exact verdict agreement
    overstated: float | None
    missing_condition: float | None
    disagreements: list[str]  # "q017: human=partial judge=correct"


def agreement(results: list[GenQuestionResult], labels: list[HumanLabel]) -> Agreement:
    by_key = {(lb.id, lb.answer_hash): lb for lb in labels}
    pairs = [
        (r, by_key[(r.id, r.answer_hash)])
        for r in results
        if r.judge is not None and (r.id, r.answer_hash) in by_key
    ]
    if not pairs:
        return Agreement(
            n=0, verdict=None, overstated=None, missing_condition=None, disagreements=[]
        )

    def share(pred) -> float:
        return round(sum(1 for r, lb in pairs if pred(r, lb)) / len(pairs), 3)

    return Agreement(
        n=len(pairs),
        verdict=share(lambda r, lb: r.judge.verdict == lb.verdict),
        overstated=share(lambda r, lb: r.judge.overstated == lb.overstated),
        missing_condition=share(lambda r, lb: r.judge.missing_condition == lb.missing_condition),
        disagreements=[
            f"{r.id}: human={lb.verdict} judge={r.judge.verdict} ({r.judge.explanation[:80]})"
            for r, lb in pairs
            if r.judge.verdict != lb.verdict
        ],
    )
