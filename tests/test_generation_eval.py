"""Tests for the LLM judge (request, cache, failures) and generation scoring.

Fake clients only: no API calls, no cost, and the exact request can be inspected.
"""

import json
from types import SimpleNamespace

import pytest

from regrag.evaluation import GoldItem, GoldRef
from regrag.evaluation.generation_eval import score_answer, summarize
from regrag.evaluation.judge import JudgeConfig, JudgeError, JudgeInput, LLMJudge
from regrag.evaluation.labels import HumanLabel, agreement, sample_for_labelling
from regrag.generation.models import Answer, Citation, ContextChunk
from regrag.ingestion.models import ParsedBlock

DOC = "doc"


class FakeChat:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        msg = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


VERDICT = {
    "verdict": "partial",
    "overstated": False,
    "missing_condition": True,
    "contradicts_reference": False,
    "explanation": "drops the double-deck exception",
}
INPUT = JudgeInput(
    question="q",
    reference_answer="3 doors",
    evidence="71-100: 3",
    system_answer="3 doors",
    quotes=["71 - 100 3"],
)


# --- judge -------------------------------------------------------------------------------


def test_judge_request_uses_json_mode_and_thinking():
    client = FakeChat(json.dumps(VERDICT))
    verdict = LLMJudge(client, JudgeConfig()).judge(INPUT)
    assert verdict.verdict == "partial" and verdict.score == 0.5 and verdict.missing_condition
    req = client.requests[0]
    assert req["model"] == "deepseek-v4-pro"
    assert req["response_format"] == {"type": "json_object"}
    assert req["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "json" in req["messages"][0]["content"].lower()  # JSON mode requires the word


def test_cache_avoids_repeat_calls_in_memory_and_on_disk(tmp_path):
    cache = tmp_path / "judge_cache.jsonl"
    client = FakeChat(json.dumps(VERDICT))
    judge = LLMJudge(client, JudgeConfig(), cache)
    judge.judge(INPUT)
    judge.judge(INPUT)
    assert judge.calls == 1

    reloaded = LLMJudge(FakeChat("never called"), JudgeConfig(), cache)
    assert reloaded.judge(INPUT).verdict == "partial" and reloaded.calls == 0
    # Anything that can change the verdict changes the key:
    changed = INPUT.model_copy(update={"system_answer": "2 doors"})
    assert reloaded.cache_key(changed) != reloaded.cache_key(INPUT)


def test_bypassing_the_cache_calls_again():
    client = FakeChat(json.dumps(VERDICT))
    judge = LLMJudge(client, JudgeConfig())
    judge.judge(INPUT)
    judge.judge(INPUT, use_cache=False)
    assert judge.calls == 2


def test_unparseable_judge_output_raises_and_is_not_cached(tmp_path):
    cache = tmp_path / "c.jsonl"
    with pytest.raises(JudgeError):
        LLMJudge(FakeChat("Sure! The answer is correct."), JudgeConfig(), cache).judge(INPUT)
    assert not cache.exists()


# --- scoring -----------------------------------------------------------------------------


def ctx(label: str, scope: str, sections: list[str]) -> ContextChunk:
    return ContextChunk(
        label=label,
        chunk_id=f"{DOC}:{label}",
        doc_id=DOC,
        scope=scope,
        sections=sections,
        page_start=47,
        page_end=47,
        header="h",
        text="t",
        index=int(label[1:]),
        origin="retrieved",
    )


def answer(covered: bool, citations=(), context=None, error=None) -> Answer:
    return Answer(
        question="q",
        covered=covered,
        text="three doors",
        citations=list(citations),
        context=context or [ctx("C1", "Annex 3", ["7.6.1"])],
        model="m",
        prompt_version="1",
        error=error,
    )


def cite(label: str, verified: bool = True) -> Citation:
    return Citation(chunk_label=label, quote="x", verified=verified)


GOLD = GoldItem(
    id="q1",
    question="How many doors?",
    lang="en",
    type="numeric",
    gold=[GoldRef(doc_id=DOC, scope="Annex 3", section="7.6.1")],
    answer="Three.",
    source="t",
    status="verified",
)
UNANSWERABLE = GoldItem(
    id="q2",
    question="Max noise?",
    lang="en",
    type="unanswerable",
    gold=[],
    answer=None,
    source="t",
    status="verified",
)
BLOCKS = {
    DOC: [
        ParsedBlock(
            text="7.6.1. three doors",
            page_number=47,
            bbox=(0, 0, 1, 1),
            scope="Annex 3",
            section_number="7.6.1",
        )
    ]
}


def test_answered_question_is_judged_and_citation_precision_computed():
    judge = LLMJudge(FakeChat(json.dumps({**VERDICT, "verdict": "correct"})), JudgeConfig())
    two_ctx = [ctx("C1", "Annex 3", ["7.6.1"]), ctx("C2", "Annex 8", ["3.6"])]
    r = score_answer(
        GOLD, answer(True, [cite("C1"), cite("C2"), cite("C1", False)], two_ctx), judge, BLOCKS
    )
    assert r.outcome == "answered" and r.score == 1.0
    assert r.citation_precision == 0.5  # of 2 VERIFIED citations, 1 points at a gold section
    assert r.context_recall == 1.0
    sent = judge.client.requests[0]["messages"][1]["content"]
    assert "7.6.1. three doors" in sent  # the judge sees the gold EVIDENCE text


def test_refusals_and_hallucinations_need_no_judge():
    judge = LLMJudge(FakeChat("unused"), JudgeConfig())
    assert score_answer(GOLD, answer(False), judge, BLOCKS).outcome == "false_refusal"
    assert score_answer(UNANSWERABLE, answer(False), judge, BLOCKS).score == 1.0
    halluc = score_answer(UNANSWERABLE, answer(True), judge, BLOCKS)
    assert halluc.outcome == "hallucinated" and halluc.score == 0.0
    assert judge.calls == 0


def test_generation_error_scores_zero():
    r = score_answer(GOLD, answer(False, error="JSONDecodeError"), None, BLOCKS)
    assert r.outcome == "error" and r.score == 0.0


def test_summary_rates():
    judge = LLMJudge(FakeChat(json.dumps(VERDICT)), JudgeConfig())  # always "partial"
    results = [
        score_answer(GOLD, answer(True, [cite("C1")]), judge, BLOCKS),  # 0.5
        score_answer(GOLD.model_copy(update={"id": "q3"}), answer(False), judge, BLOCKS),  # 0
        score_answer(UNANSWERABLE, answer(False), judge, BLOCKS),  # correct refusal
    ]
    s = summarize(results)
    assert s.answer_score == 0.25  # mean of 0.5 and 0 over answerable questions
    assert s.false_refusal_rate == 0.5 and s.refusal_accuracy == 1.0
    assert s.missing_condition_rate == 1.0 and s.verdicts == {"partial": 1}


# --- calibration ---------------------------------------------------------------------------


def test_agreement_matches_labels_to_the_exact_answer():
    judge = LLMJudge(FakeChat(json.dumps(VERDICT)), JudgeConfig())
    r = score_answer(GOLD, answer(True, [cite("C1")]), judge, BLOCKS)
    same = HumanLabel(id="q1", answer_hash=r.answer_hash, verdict="partial", missing_condition=True)
    other_answer = HumanLabel(id="q1", answer_hash="different", verdict="incorrect")
    a = agreement([r], [same, other_answer])
    assert a.n == 1 and a.verdict == 1.0 and a.missing_condition == 1.0
    assert sample_for_labelling([r], 5) == [r]
