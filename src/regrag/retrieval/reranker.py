"""Cross-encoder reranking with bge-reranker-v2-m3.

A bi-encoder (bge-m3) embeds query and chunk separately: fast, precomputable,
but it never sees them together. A cross-encoder reads "[query] [chunk]" as ONE
input, so attention runs across both and it can judge whether the text actually
answers the question. Far more precise, but nothing can be precomputed: one full
forward pass per (query, candidate) pair. Hence: cheap broad retrieval first,
expensive precise reranking on a few candidates.

The output is a relevance logit (higher = more relevant). Its absolute value is
also a signal: if even the best candidate scores low, the corpus probably does
not answer the question (threshold to be calibrated in evaluation).
"""

import logging
from typing import Literal, Protocol

from pydantic import ConfigDict, Field

from regrag.ingestion.models import Model

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    @property
    def model_id(self) -> str: ...

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class RerankerConfig(Model):
    model_config = ConfigDict(frozen=True)

    model_name: str = "BAAI/bge-reranker-v2-m3"
    revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"  # pinned, like the embedder
    max_length: int = Field(512, gt=0)  # query + passage; only the passage is truncated
    batch_size: int = Field(8, gt=0)
    device: Literal["auto", "cuda", "cpu"] = "auto"
    fp16: bool = False  # see EmbedderConfig.fp16: slower on GPUs without tensor cores

    @property
    def model_id(self) -> str:
        return f"{self.model_name}@{self.revision[:8]}"


class BGEReranker:
    def __init__(self, config: RerankerConfig | None = None) -> None:
        import torch  # heavy imports only when a real reranker is built
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.cfg = config or RerankerConfig()
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        dtype = torch.float16 if self.cfg.fp16 and self.device.type == "cuda" else torch.float32

        name, rev = self.cfg.model_name, self.cfg.revision
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=rev)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, revision=rev, dtype=dtype
        )
        self.model.to(self.device).eval()
        logger.info("Loaded %s@%s on %s (%s)", name, rev[:8], self.device, dtype)

    @property
    def model_id(self) -> str:
        return self.cfg.model_id

    def score(self, query: str, passages: list[str]) -> list[float]:
        scores: list[float] = []
        size = self.cfg.batch_size
        with self._torch.inference_mode():
            for start in range(0, len(passages), size):
                batch = passages[start : start + size]
                enc = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation="only_second",  # never cut the question, only the passage
                    max_length=self.cfg.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.squeeze(-1)
                scores.extend(logits.float().cpu().tolist())
        return scores
