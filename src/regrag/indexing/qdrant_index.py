"""Index chunks into Qdrant: dense + sparse named vectors, idempotent and incremental.

Collection layout (one point per chunk):

    vectors  "dense"   1024-d, cosine        meaning ("exits" ~ "doors" ~ "evacuation")
    sparse   "sparse"  bge-m3 lexical weights exact terms ("M2", "Class II", "7.6.1")
    payload  chunk metadata + text, so a search result is self-sufficient

Production properties:

- Idempotent: point id = uuid5(namespace, chunk_id). Re-running upserts the same
  ids, overwriting instead of duplicating.
- Incremental: each point stores sha256(embed_text) and the embedding model id.
  Only chunks whose text or model changed are re-embedded, the expensive part.
- No visibility gap: new/changed points are upserted *first*, and only then are
  the document's points that no longer correspond to a chunk deleted. Deleting
  first would leave a window in which the document cannot be found.
"""

import hashlib
import logging
import uuid
from collections.abc import Iterator

from pydantic import ConfigDict, Field
from qdrant_client import QdrantClient, models

from regrag.chunking.models import Chunk
from regrag.chunking.storage import CHUNKER_VERSION
from regrag.embedding.models import Embedder
from regrag.ingestion.models import Model

logger = logging.getLogger(__name__)

# Fixed namespace: the same chunk id must map to the same point id forever.
POINT_NAMESPACE = uuid.UUID("6f1d2a8e-4c3b-5e7f-9a0b-1c2d3e4f5a6b")


class IndexConfig(Model):
    model_config = ConfigDict(frozen=True)

    collection: str = "regrag_chunks"
    dense_name: str = "dense"
    sparse_name: str = "sparse"
    upsert_batch: int = Field(32, gt=0)  # chunks embedded + sent per round trip


class IndexReport(Model):
    doc_id: str
    chunks: int
    embedded: int  # new or changed, re-embedded
    unchanged: int
    deleted: int  # stale points removed


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )


def _batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class ChunkIndex:
    def __init__(self, client: QdrantClient, embedder: Embedder, config: IndexConfig | None = None):
        self.client = client
        self.embedder = embedder
        self.cfg = config or IndexConfig()

    def ensure_collection(self) -> None:
        """Create the collection if missing; refuse to use one with an incompatible shape."""
        name = self.cfg.collection
        if self.client.collection_exists(name):
            vectors = self.client.get_collection(name).config.params.vectors
            size = vectors[self.cfg.dense_name].size  # type: ignore[index]
            if size != self.embedder.dense_dim:
                raise ValueError(
                    f"Collection {name!r} has {size}-d vectors, embedder produces "
                    f"{self.embedder.dense_dim}-d. Use another collection or --recreate."
                )
            return
        self.client.create_collection(
            name,
            vectors_config={
                self.cfg.dense_name: models.VectorParams(
                    size=self.embedder.dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={self.cfg.sparse_name: models.SparseVectorParams()},
        )
        # Keyword indexes make filtered search ("only Annex 8", "only R107") fast.
        for field in ("doc_id", "scope"):
            self.client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
        logger.info("Created collection %s", name)

    def _existing(self, doc_id: str) -> dict[str, tuple[str, str]]:
        """point id -> (text_sha256, embedding_model) for every point of a document."""
        found: dict[str, tuple[str, str]] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.cfg.collection,
                scroll_filter=_doc_filter(doc_id),
                with_payload=["text_sha256", "embedding_model"],
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for p in points:
                payload = p.payload or {}
                found[str(p.id)] = (
                    payload.get("text_sha256", ""),
                    payload.get("embedding_model", ""),
                )
            if offset is None:
                return found

    def _payload(self, chunk: Chunk, sha: str) -> dict:
        return {
            "chunk_id": chunk.id,
            "doc_id": chunk.doc_id,
            "index": chunk.index,
            "scope": chunk.scope,
            "section_path": chunk.section_path,
            "sections": chunk.sections,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "header": chunk.header,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "block_ids": chunk.block_ids,
            "text_sha256": sha,
            "embedding_model": self.embedder.model_id,
            "chunker_version": CHUNKER_VERSION,
        }

    def index_document(self, doc_id: str, chunks: list[Chunk]) -> IndexReport:
        self.ensure_collection()
        existing = self._existing(doc_id)
        model = self.embedder.model_id

        todo = []
        for chunk in chunks:
            sha = text_sha256(chunk.embed_text)
            if existing.get(point_id(chunk.id)) != (sha, model):
                todo.append((chunk, sha))

        for batch in _batches(todo, self.cfg.upsert_batch):
            embeddings = self.embedder.embed([c.embed_text for c, _ in batch])
            points = [
                models.PointStruct(
                    id=point_id(chunk.id),
                    vector={
                        self.cfg.dense_name: emb.dense,
                        self.cfg.sparse_name: models.SparseVector(
                            indices=emb.sparse.indices, values=emb.sparse.values
                        ),
                    },
                    payload=self._payload(chunk, sha),
                )
                for (chunk, sha), emb in zip(batch, embeddings, strict=True)
            ]
            self.client.upsert(self.cfg.collection, points=points, wait=True)
            logger.info("%s: upserted %d points", doc_id, len(points))

        # Only after everything new is in place: drop points of removed chunks.
        stale = sorted(set(existing) - {point_id(c.id) for c in chunks})
        if stale:
            self.client.delete(
                self.cfg.collection, points_selector=models.PointIdsList(points=stale), wait=True
            )

        return IndexReport(
            doc_id=doc_id,
            chunks=len(chunks),
            embedded=len(todo),
            unchanged=len(chunks) - len(todo),
            deleted=len(stale),
        )

    def count(self, doc_id: str | None = None) -> int:
        flt = _doc_filter(doc_id) if doc_id else None
        return self.client.count(self.cfg.collection, count_filter=flt, exact=True).count
