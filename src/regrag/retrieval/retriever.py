"""Hybrid retrieval: dense + sparse search -> RRF fusion -> cross-encoder rerank.

    query --embed--> dense vector + sparse weights
          --Qdrant (one batched request)--> dense top-30 | sparse top-30
          --rrf()--> fused candidates -> top-20
          --reranker--> top-5 hits

Fusion is done here, client-side, rather than in Qdrant: every hit keeps its
dense rank, sparse rank, RRF score and rerank score, so evaluation can tell
which stage lost a relevant chunk. Moving to Qdrant's server-side fusion later
only changes `_search`.
"""

import logging
import time
from typing import Any

from qdrant_client import QdrantClient, models

from regrag.embedding.models import Embedder
from regrag.retrieval.fusion import rrf
from regrag.retrieval.models import Hit, RetrievalConfig, RetrievalResult, SearchFilter
from regrag.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

_HIT_FIELDS = (
    "chunk_id",
    "doc_id",
    "scope",
    "section_path",
    "sections",
    "page_start",
    "page_end",
    "header",
    "text",
)


def _qdrant_filter(flt: SearchFilter | None) -> models.Filter | None:
    if flt is None:
        return None
    conditions: list[Any] = []
    if flt.doc_ids:
        conditions.append(
            models.FieldCondition(key="doc_id", match=models.MatchAny(any=flt.doc_ids))
        )
    if flt.scopes:
        conditions.append(models.FieldCondition(key="scope", match=models.MatchAny(any=flt.scopes)))
    return models.Filter(must=conditions) if conditions else None


class HybridRetriever:
    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.reranker = reranker
        self.cfg = config or RetrievalConfig()

    def _search(self, query: str, flt: SearchFilter | None) -> tuple[list, list]:
        """Dense and sparse search in ONE round trip; returns two ranked point lists."""
        emb = self.embedder.embed([query])[0]
        index = self.cfg.index
        qfilter = _qdrant_filter(flt)
        requests = [
            models.QueryRequest(
                query=emb.dense,
                using=index.dense_name,
                limit=self.cfg.dense_k,
                filter=qfilter,
                with_payload=list(_HIT_FIELDS),
            )
        ]
        # A query made only of special/stop tokens has no sparse weights; an empty
        # sparse vector is not a valid query, so skip that retriever.
        if emb.sparse.indices:
            requests.append(
                models.QueryRequest(
                    query=models.SparseVector(indices=emb.sparse.indices, values=emb.sparse.values),
                    using=index.sparse_name,
                    limit=self.cfg.sparse_k,
                    filter=qfilter,
                    with_payload=list(_HIT_FIELDS),
                )
            )
        responses = self.client.query_batch_points(index.collection, requests=requests)
        dense = responses[0].points
        sparse = responses[1].points if len(responses) > 1 else []
        return dense, sparse

    def search(
        self,
        query: str,
        flt: SearchFilter | None = None,
        top_k: int | None = None,
        rerank: bool = True,
    ) -> RetrievalResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        dense, sparse = self._search(query, flt)
        timings["search_ms"] = (time.perf_counter() - t0) * 1000  # embed + both searches

        # Build one Hit per chunk, recording what each retriever thought of it.
        hits: dict[str, Hit] = {}
        for name, points in (("dense", dense), ("sparse", sparse)):
            for rank, point in enumerate(points, start=1):
                payload = point.payload or {}
                cid = payload["chunk_id"]
                hit = hits.get(cid) or Hit(**{f: payload[f] for f in _HIT_FIELDS})
                setattr(hit, f"{name}_rank", rank)
                setattr(hit, f"{name}_score", point.score)
                hits[cid] = hit

        fused = rrf(
            {
                "dense": [p.payload["chunk_id"] for p in dense],
                "sparse": [p.payload["chunk_id"] for p in sparse],
            },
            k=self.cfg.rrf_k,
            weights={"dense": self.cfg.dense_weight, "sparse": self.cfg.sparse_weight},
        )
        pool = []
        for cid, score in fused:
            hits[cid].rrf_score = score
            pool.append(hits[cid])
        candidates = pool[: self.cfg.rerank_candidates]

        ranked = candidates
        if rerank and self.reranker is not None and candidates:
            t1 = time.perf_counter()
            scores = self.reranker.score(query, [h.passage for h in candidates])
            timings["rerank_ms"] = (time.perf_counter() - t1) * 1000
            for hit, score in zip(candidates, scores, strict=True):
                hit.rerank_score = score
            # Stable sort: equal rerank scores keep their fused order.
            ranked = sorted(candidates, key=lambda h: -(h.rerank_score or 0.0))

        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        k = top_k or self.cfg.top_k
        logger.info("search %r: %d candidates, %s", query[:60], len(candidates), timings)
        return RetrievalResult(
            query=query,
            hits=[h.model_copy() for h in ranked[:k]],
            candidates=[h.model_copy() for h in candidates],
            pool=[h.model_copy() for h in pool],
            timings_ms={name: round(v, 1) for name, v in timings.items()},
        )
