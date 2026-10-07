"""Embedding interfaces and data types.

The rest of the pipeline depends only on the `Embedder` protocol, never on
torch/transformers directly: indexing and retrieval can be tested with a fake
embedder, and the model can later be swapped (ONNX, a hosted API) without
touching them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import ConfigDict, Field

from regrag.ingestion.models import Model


@dataclass(frozen=True)
class SparseVector:
    """Lexical weights: token ids present in the text and their learned importance."""

    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class Embedding:
    dense: list[float]  # L2-normalised, so dot product == cosine similarity
    sparse: SparseVector


class Embedder(Protocol):
    @property
    def model_id(self) -> str:
        """Identifies the vector space: vectors from different ids are not comparable."""
        ...

    @property
    def dense_dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[Embedding]: ...


class EmbedderConfig(Model):
    model_config = ConfigDict(frozen=True)

    model_name: str = "BAAI/bge-m3"
    # Pinned commit of the model repository on Hugging Face. Without a pin, an
    # upstream update would silently change every embedding.
    revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    # Chunks are guaranteed <= 512 tokens by the chunker (same tokenizer), so
    # nothing is ever truncated here.
    max_length: int = Field(512, gt=0)
    batch_size: int = Field(4, gt=0)
    device: Literal["auto", "cuda", "cpu"] = "auto"
    # Half precision halves memory, but is only faster on GPUs with tensor cores.
    # Measured on a GTX 1650 Ti (Turing, no tensor cores): fp16 2.0 chunks/s vs
    # fp32 8.0 chunks/s. Enable it on RTX-class or datacenter GPUs.
    fp16: bool = False
    dense_dim: int = 1024  # checked against the loaded model

    @property
    def model_id(self) -> str:
        """Known without loading the model, so callers can decide whether they need it."""
        return f"{self.model_name}@{self.revision[:8]}"


class LazyEmbedder:
    """Defers building an expensive embedder until something actually needs embedding.

    `model_id` and `dense_dim` come from the config, so an indexing run that finds
    nothing changed never loads the 2 GB model.
    """

    def __init__(self, config: EmbedderConfig, factory: Callable[[], Embedder]) -> None:
        self._config = config
        self._factory = factory
        self._embedder: Embedder | None = None

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def dense_dim(self) -> int:
        return self._config.dense_dim

    def embed(self, texts: list[str]) -> list[Embedding]:
        if self._embedder is None:
            self._embedder = self._factory()
            if self._embedder.dense_dim != self.dense_dim:
                raise ValueError(
                    f"{self.model_id} produces {self._embedder.dense_dim}-d vectors, "
                    f"config says {self.dense_dim}"
                )
        return self._embedder.embed(texts)
