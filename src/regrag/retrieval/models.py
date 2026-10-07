"""Retrieval configuration and result types."""

from pydantic import ConfigDict, Field

from regrag.indexing.qdrant_index import IndexConfig
from regrag.ingestion.models import Model


class RetrievalConfig(Model):
    """Every knob of the retrieval pipeline. These are hyperparameters: the
    evaluation stage decides their values, not intuition."""

    model_config = ConfigDict(frozen=True)

    index: IndexConfig = Field(default_factory=IndexConfig)
    dense_k: int = Field(30, gt=0)  # candidates from dense search
    sparse_k: int = Field(30, gt=0)  # candidates from sparse search
    rrf_k: int = Field(60, gt=0)  # RRF smoothing constant (60 = original paper)
    dense_weight: float = Field(1.0, ge=0)
    sparse_weight: float = Field(1.0, ge=0)
    rerank_candidates: int = Field(20, gt=0)  # fused candidates sent to the cross-encoder
    top_k: int = Field(5, gt=0)  # hits returned


class SearchFilter(Model):
    """Hard filters applied inside both searches (not after them)."""

    doc_ids: list[str] | None = None
    scopes: list[str] | None = None


class Hit(Model):
    chunk_id: str
    doc_id: str
    scope: str
    section_path: list[str]
    sections: list[str]
    page_start: int
    page_end: int
    header: str
    text: str
    # Provenance of every stage, so a miss can be traced to the stage that lost it.
    dense_rank: int | None = None  # 1-based; None = not in dense top-k
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None  # cross-encoder logit; None if not reranked

    @property
    def passage(self) -> str:
        """What the reranker (and later the LLM) reads: header for context, then text."""
        return f"{self.header}\n\n{self.text}"


class RetrievalResult(Model):
    query: str
    hits: list[Hit]  # final top_k, best first
    candidates: list[Hit]  # fused top-N sent to the reranker, in fused order
    # Every chunk returned by any retriever, in fused order, with its per-retriever
    # ranks. Lets evaluation measure each stage (dense-only, sparse-only, fused) at
    # depths beyond the rerank cut-off.
    pool: list[Hit] = []
    timings_ms: dict[str, float]
