"""Tests for RRF fusion and the hybrid retrieval pipeline.

The pipeline runs on in-memory Qdrant with FakeEmbedder (sparse = keyword match)
and FakeReranker (score = shared words), so expected rankings can be derived by hand.
"""

import pytest
from fakes import FakeEmbedder, FakeReranker, make_chunk

from regrag.indexing import ChunkIndex, IndexConfig
from regrag.retrieval import HybridRetriever, RetrievalConfig, SearchFilter, rrf

# --- RRF ---------------------------------------------------------------------


def test_rrf_scores_match_the_formula():
    fused = dict(rrf({"dense": ["a", "b"], "sparse": ["b", "c"]}, k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)  # found by both
    assert fused["c"] == pytest.approx(1 / 62)


def test_agreement_beats_a_single_first_place():
    # "b" is 2nd in dense and 1st in sparse; "a" is 1st in dense only.
    order = [doc for doc, _ in rrf({"dense": ["a", "b"], "sparse": ["b"]})]
    assert order[0] == "b"


def test_weights_scale_a_retrievers_contribution():
    fused = dict(rrf({"dense": ["a"], "sparse": ["b"]}, k=60, weights={"sparse": 2.0}))
    assert fused["b"] == pytest.approx(2 * fused["a"])


def test_ties_are_broken_deterministically():
    # Same score for "x" and "y": the better single rank wins, then the id.
    assert [d for d, _ in rrf({"dense": ["x"], "sparse": ["y"]})] == ["x", "y"]
    assert [d for d, _ in rrf({"dense": ["y"], "sparse": ["x"]})] == ["x", "y"]


# --- pipeline ------------------------------------------------------------------

TEXTS = [
    "gangway width shall be at least 450 mm",
    "service doors minimum number for class II",
    "seat width shall be at least 400 mm",
    "trolleybus insulation monitoring system",
    "emergency exits and escape hatches",
]
CONFIG = RetrievalConfig(index=IndexConfig(collection="test"), rerank_candidates=5, top_k=3)


@pytest.fixture
def indexed(client):
    embedder = FakeEmbedder()
    ChunkIndex(client, embedder, CONFIG.index).index_document(
        "doc", [make_chunk(i, t) for i, t in enumerate(TEXTS)]
    )
    ChunkIndex(client, embedder, CONFIG.index).index_document(
        "other",
        [make_chunk(0, "gangway width in another regulation", doc_id="other", scope="Annex 8")],
    )
    return client, embedder


def test_reranked_hits_carry_provenance_of_every_stage(indexed):
    client, embedder = indexed
    result = HybridRetriever(client, embedder, FakeReranker(), CONFIG).search("gangway width")
    best = result.hits[0]
    assert best.text.startswith("gangway width")
    assert best.sparse_rank is not None and best.dense_rank is not None
    assert best.rrf_score > 0 and best.rerank_score == 2.0
    assert len(result.hits) == 3
    assert {"search_ms", "rerank_ms", "total_ms"} <= set(result.timings_ms)


def test_reranker_reorders_fused_candidates(indexed):
    client, embedder = indexed
    result = HybridRetriever(client, embedder, FakeReranker(), CONFIG).search("seat width")
    scores = [h.rerank_score for h in result.hits]
    assert scores == sorted(scores, reverse=True)
    assert result.hits[0].text.startswith("seat width")


def test_no_rerank_keeps_fused_order_and_skips_the_model(indexed):
    client, embedder = indexed
    reranker = FakeReranker()
    result = HybridRetriever(client, embedder, reranker, CONFIG).search("seat width", rerank=False)
    assert reranker.calls == 0
    assert all(h.rerank_score is None for h in result.hits)
    rrf_scores = [h.rrf_score for h in result.hits]
    assert rrf_scores == sorted(rrf_scores, reverse=True)


def test_filters_apply_inside_both_searches(indexed):
    client, embedder = indexed
    retriever = HybridRetriever(client, embedder, FakeReranker(), CONFIG)
    only_other = retriever.search("gangway width", SearchFilter(doc_ids=["other"]))
    assert {h.doc_id for h in only_other.candidates} == {"other"}
    only_annex8 = retriever.search("gangway width", SearchFilter(scopes=["Annex 8"]))
    assert {h.scope for h in only_annex8.candidates} == {"Annex 8"}


def test_candidates_are_limited_and_unique(indexed):
    client, embedder = indexed
    config = CONFIG.model_copy(update={"rerank_candidates": 2})
    result = HybridRetriever(client, embedder, FakeReranker(), config).search("width")
    ids = [h.chunk_id for h in result.candidates]
    assert len(ids) == 2 == len(set(ids))
