"""Tests for the golden set model, gold matching, metrics and stage diagnosis.

A bug in a metric silently lies about every future experiment, so the formulas
are checked against hand-computed values.
"""

import pytest
from pydantic import ValidationError

from regrag.evaluation import (
    GoldItem,
    GoldRef,
    check_against_corpus,
    gold_text,
    load_golden,
    longest_shared_run,
    save_golden,
)
from regrag.evaluation.metrics import first_relevant_rank, matches, recall_at_k, reciprocal_rank
from regrag.evaluation.runner import (
    QuestionResult,
    StageScore,
    _diagnose,
    stage_rankings,
    threshold_analysis,
)
from regrag.ingestion.models import ParsedBlock
from regrag.ingestion.storage import StorageError
from regrag.retrieval.models import Hit, RetrievalResult

DOC = "unece-r107-rev9"


def ref(section: str | None = "7.6.1", scope: str = "Annex 3", page: int | None = None) -> GoldRef:
    return GoldRef(doc_id=DOC, scope=scope, section=section, page=page)


def hit(sections: list[str], scope: str = "Annex 3", pages=(47, 48), **ranks) -> Hit:
    return Hit(
        chunk_id=f"c-{'-'.join(sections) or pages[0]}",
        doc_id=DOC,
        scope=scope,
        section_path=[scope],
        sections=sections,
        page_start=pages[0],
        page_end=pages[1],
        header="h",
        text="t",
        **ranks,
    )


def item(**overrides) -> GoldItem:
    data = dict(
        id="q001",
        question="How many doors?",
        lang="en",
        type="numeric",
        gold=[ref()],
        answer="Two.",
        source="test",
    )
    data.update(overrides)
    return GoldItem(**data)


# --- schema ----------------------------------------------------------------------


def test_unanswerable_must_have_no_gold_and_no_answer():
    item(type="unanswerable", gold=[], answer=None)  # valid
    with pytest.raises(ValidationError):
        item(type="unanswerable", gold=[ref()], answer=None)
    with pytest.raises(ValidationError):
        item(type="numeric", gold=[], answer="x")


def test_unnumbered_gold_needs_a_page():
    with pytest.raises(ValidationError):
        GoldRef(doc_id=DOC, scope="Annex 4", section=None)
    assert ref(section=None, scope="Annex 4", page=82).label() == "Annex 4 / p.82"


def test_split_is_deterministic_and_roughly_70_30():
    splits = [item(id=f"q{i:03d}").split for i in range(1000)]
    assert splits == [item(id=f"q{i:03d}").split for i in range(1000)]
    assert 0.25 < splits.count("test") / 1000 < 0.35


def test_save_load_round_trip_and_duplicate_ids(tmp_path):
    path = tmp_path / "golden.jsonl"
    items = [item(id="q001", question="Ширина прохода?", lang="ru"), item(id="q002")]
    save_golden(path, items)
    assert load_golden(path) == items
    assert "Ширина" in path.read_text(encoding="utf-8")  # readable, not \u escapes
    path.write_text(
        "\n".join([path.read_text(encoding="utf-8").splitlines()[0]] * 2), encoding="utf-8"
    )
    with pytest.raises(StorageError, match="duplicate id"):
        load_golden(path)


# --- gold text and corpus checks ---------------------------------------------------


def block(text: str, section: str | None, scope: str = "Annex 3", page: int = 47) -> ParsedBlock:
    return ParsedBlock(
        text=text, page_number=page, bbox=(0, 0, 1, 1), scope=scope, section_number=section
    )


BLOCKS = [
    block("7.6.1. Number of exits", "7.6.1"),
    block("7.6.1.1. The minimum number of doors shall be two.", "7.6.1.1"),
    block("Number of passengers | doors", None),  # table row: belongs to 7.6.1.1
    block("7.6.2. Positioning of exits", "7.6.2"),
    block("7.6.10. Retractable steps", "7.6.10"),  # must NOT match gold 7.6.1
    block("Figure 6 gangway gauge 450 mm", None, scope="Annex 4", page=82),
]


def test_gold_text_includes_subsections_and_unnumbered_rows_only():
    text = gold_text(ref("7.6.1"), BLOCKS)
    assert "two" in text and "Number of passengers" in text
    assert "Positioning" not in text and "Retractable" not in text  # 7.6.10 is not 7.6.1.x


def test_gold_text_by_page_for_unnumbered_scope():
    assert "450 mm" in gold_text(ref(None, "Annex 4", page=82), BLOCKS)


def test_corpus_check_reports_gold_pointing_at_nothing():
    items = [
        item(gold=[ref("9.9.9")]),
        item(id="q002", gold=[GoldRef(doc_id="nope", scope="X", section="1")]),
    ]
    problems = check_against_corpus(items, {DOC: BLOCKS})
    assert len(problems) == 2


def test_longest_shared_run():
    assert longest_shared_run("how many doors shall be fitted", "the doors shall be two") == 3
    assert longest_shared_run("Сколько дверей", "Минимальное количество дверей") == 1


# --- metrics ---------------------------------------------------------------------------


def test_matching_by_section_prefix_scope_and_page():
    assert matches(hit(["7.6.1.1", "7.6.1.2"]), ref("7.6.1"))
    assert not matches(hit(["7.6.10"]), ref("7.6.1"))  # prefix must end at a dot
    assert not matches(hit(["7.6.1"], scope="Annex 8"), ref("7.6.1"))
    assert matches(hit([], scope="Annex 4", pages=(81, 83)), ref(None, "Annex 4", page=82))


def test_recall_and_reciprocal_rank():
    ranked = [hit(["1.1"]), hit(["7.6.1.1"]), hit(["7.6.2"])]
    gold = [ref("7.6.1"), ref("7.6.2")]
    assert recall_at_k(ranked, gold, 1) == 0.0
    assert recall_at_k(ranked, gold, 2) == 0.5  # one of two gold sections
    assert recall_at_k(ranked, gold, 3) == 1.0
    assert reciprocal_rank(ranked, gold, 20) == 0.5  # first relevant at rank 2
    assert first_relevant_rank(ranked, [ref("9.9")]) is None


# --- stages and diagnosis -------------------------------------------------------------


def test_stage_rankings_and_diagnosis():
    a = hit(["1.1"], dense_rank=1, sparse_rank=2, rrf_score=0.03, rerank_score=5.0)
    b = hit(["7.6.1.1"], dense_rank=2, sparse_rank=None, rrf_score=0.02, rerank_score=-1.0)
    result = RetrievalResult(query="q", hits=[a], candidates=[a, b], pool=[a, b], timings_ms={})
    stages = stage_rankings(result)
    assert [h.sections for h in stages["sparse"]] == [["1.1"]]  # b not found by sparse
    assert [h.sections for h in stages["reranked"]] == [["1.1"], ["7.6.1.1"]]

    gold_b = item(gold=[ref("7.6.1")])
    assert _diagnose(stages, gold_b, n_candidates=20, reranked=True) == "ok"  # rank 2 <= 5
    # In the pool but not among the rerank candidates: lost at the cut-off.
    cut = RetrievalResult(query="q", hits=[a], candidates=[a], pool=[a, b], timings_ms={})
    assert _diagnose(stage_rankings(cut), gold_b, n_candidates=1, reranked=True) == "below_cutoff"
    assert _diagnose(stages, item(gold=[ref("9.9")]), 20, True) == "not_retrieved"


def test_threshold_picks_the_separating_score():
    def qr(top1: float, answerable: bool) -> QuestionResult:
        stages = (
            {"reranked": StageScore(recall={5: 1.0}, rr=1.0, first_rank=1)} if answerable else {}
        )
        return QuestionResult(
            id="x",
            type="t",
            lang="en",
            split="dev",
            question="q",
            stages=stages,
            top1_rerank=top1,
            top_hits=[],
        )

    results = [qr(4.0, True), qr(2.5, True), qr(-1.0, False), qr(0.5, False)]
    th = threshold_analysis(results)
    assert th.best_threshold == 2.5 and th.best_balanced_accuracy == 1.0
