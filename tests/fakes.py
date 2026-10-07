"""Test doubles shared by the indexing and retrieval tests.

They implement the same protocols as the real models (Embedder, Reranker), so
the pipeline runs end to end with no model download and no GPU.
"""

import hashlib
import zlib

from regrag.chunking.models import Chunk
from regrag.embedding.models import Embedding, SparseVector


def _word_id(word: str) -> int:
    # zlib.crc32 is stable across processes; built-in hash() of str is randomised
    # per process (PYTHONHASHSEED) and would make tests flaky.
    return zlib.crc32(word.encode()) % 10_000


class FakeEmbedder:
    """Deterministic vectors derived from the text; records what it was asked to embed.

    Sparse = one index per distinct word, so sparse search behaves like keyword
    matching. Dense = bytes of a hash: arbitrary but stable.
    """

    def __init__(self, model_id: str = "fake@1", dim: int = 8) -> None:
        self.model_id = model_id
        self.dense_dim = dim
        self.calls: list[str] = []

    def embed(self, texts: list[str]) -> list[Embedding]:
        self.calls.extend(texts)
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            dense = [b / 255 + 0.01 for b in digest[: self.dense_dim]]
            words = sorted({_word_id(w) for w in text.lower().split()})
            out.append(Embedding(dense, SparseVector(words, [1.0] * len(words))))
        return out


class FakeReranker:
    """Scores a passage by how many query words it contains."""

    model_id = "fake-reranker@1"

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls += 1
        q = set(query.lower().split())
        return [float(len(q & set(p.lower().split()))) for p in passages]


def make_chunk(i: int, text: str = "", doc_id: str = "doc", scope: str = "Annex 3") -> Chunk:
    body = text or f"paragraph {i} about exits and doors"
    return Chunk(
        id=f"{doc_id}:c{i:05d}",
        doc_id=doc_id,
        index=i,
        embed_text=f"{doc_id} | {scope} > 7.6 Exits\n\n{body}",
        header=f"{doc_id} | {scope} > 7.6 Exits",
        text=body,
        token_count=10,
        scope=scope,
        section_path=[scope, "7.6 Exits"],
        sections=[f"7.6.{i}"],
        page_start=47,
        page_end=47,
        block_ids=[f"{doc_id}:{i:05d}"],
    )
