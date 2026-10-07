"""Tests for Qdrant indexing: idempotency, incremental updates, stale-point cleanup.

Runs against Qdrant's in-process local mode (QdrantClient(":memory:")) with a
fake embedder: no server, no model download, same client API as production.
"""

import pytest
from fakes import FakeEmbedder, make_chunk
from qdrant_client import models

from regrag.indexing import ChunkIndex, IndexConfig, point_id


def make_index(client, embedder=None) -> ChunkIndex:
    return ChunkIndex(client, embedder or FakeEmbedder(), IndexConfig(collection="test"))


def test_first_run_embeds_and_stores_everything(client):
    index = make_index(client)
    report = index.index_document("doc", [make_chunk(i) for i in range(5)])
    assert (report.embedded, report.unchanged, report.deleted) == (5, 0, 0)
    assert index.count("doc") == 5


def test_rerun_is_idempotent_and_embeds_nothing(client):
    chunks = [make_chunk(i) for i in range(5)]
    make_index(client).index_document("doc", chunks)

    embedder = FakeEmbedder()
    report = make_index(client, embedder).index_document("doc", chunks)
    assert (report.embedded, report.unchanged) == (0, 5)
    assert embedder.calls == []  # the model was never asked for anything
    assert make_index(client).count("doc") == 5  # no duplicates


def test_only_changed_chunks_are_reembedded(client):
    chunks = [make_chunk(i) for i in range(5)]
    make_index(client).index_document("doc", chunks)

    chunks[2] = make_chunk(2, text="paragraph 2 rewritten by a parser fix")
    embedder = FakeEmbedder()
    report = make_index(client, embedder).index_document("doc", chunks)
    assert report.embedded == 1
    assert embedder.calls == [chunks[2].embed_text]


def test_removed_chunks_are_deleted(client):
    make_index(client).index_document("doc", [make_chunk(i) for i in range(5)])
    report = make_index(client).index_document("doc", [make_chunk(i) for i in range(3)])
    assert report.deleted == 2
    assert make_index(client).count("doc") == 3


def test_other_documents_are_untouched(client):
    index = make_index(client)
    index.index_document("doc", [make_chunk(i) for i in range(3)])
    index.index_document("other", [make_chunk(i, doc_id="other") for i in range(2)])
    index.index_document("doc", [make_chunk(0)])  # shrink "doc" only
    assert (index.count("doc"), index.count("other")) == (1, 2)


def test_model_change_reembeds_everything(client):
    chunks = [make_chunk(i) for i in range(3)]
    make_index(client).index_document("doc", chunks)
    report = make_index(client, FakeEmbedder(model_id="fake@2")).index_document("doc", chunks)
    assert report.embedded == 3


def test_dimension_mismatch_is_refused(client):
    make_index(client).index_document("doc", [make_chunk(0)])
    with pytest.raises(ValueError, match="8-d vectors"):
        make_index(client, FakeEmbedder(dim=16)).ensure_collection()


def test_point_ids_are_deterministic_and_payload_is_complete(client):
    index = make_index(client)
    chunk = make_chunk(7)
    index.index_document("doc", [chunk])
    assert point_id(chunk.id) == point_id("doc:c00007")
    [point] = client.retrieve("test", ids=[point_id(chunk.id)], with_payload=True)
    payload = point.payload
    assert payload["chunk_id"] == chunk.id
    assert payload["text"] == chunk.text
    assert payload["sections"] == ["7.6.7"]
    assert payload["embedding_model"] == "fake@1"


def test_dense_and_sparse_search_find_the_chunk(client):
    embedder = FakeEmbedder()
    index = make_index(client, embedder)
    chunks = [make_chunk(i, text=t) for i, t in enumerate(["seat width", "door count", "ramp"])]
    index.index_document("doc", chunks)

    q = embedder.embed([chunks[1].embed_text])[0]
    dense_hit = client.query_points("test", query=q.dense, using="dense", limit=1).points[0]
    sparse_q = models.SparseVector(indices=q.sparse.indices, values=q.sparse.values)
    sparse_hit = client.query_points("test", query=sparse_q, using="sparse", limit=1).points[0]
    assert dense_hit.payload["chunk_id"] == sparse_hit.payload["chunk_id"] == chunks[1].id
