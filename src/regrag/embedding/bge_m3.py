"""BGE-M3 dense + sparse embeddings on plain transformers/torch.

One forward pass through the XLM-RoBERTa encoder gives a hidden state per token:

    dense  = L2-normalise(hidden state of the first token, [CLS])
    sparse = relu(sparse_linear(hidden state of each token))   # one weight per token
             -> per distinct token id keep the max weight; drop special tokens

`sparse_linear` is a single Linear(1024 -> 1) layer shipped with the model as
`sparse_linear.pt`. This mirrors the reference implementation (FlagEmbedding's
BGEM3FlagModel) without its training-oriented dependencies; equivalence was
checked against it.
"""

import logging

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from regrag.embedding.models import EmbedderConfig, Embedding, SparseVector

logger = logging.getLogger(__name__)


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


class BGEM3Embedder:
    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.cfg = config or EmbedderConfig()
        self.device = _resolve_device(self.cfg.device)
        # fp16 only makes sense on GPU; on CPU it is slower and less accurate.
        dtype = torch.float16 if self.cfg.fp16 and self.device.type == "cuda" else torch.float32

        name, rev = self.cfg.model_name, self.cfg.revision
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=rev)
        self.model = AutoModel.from_pretrained(name, revision=rev, dtype=dtype)
        self.model.to(self.device).eval()

        hidden = self.model.config.hidden_size
        self.sparse_linear = torch.nn.Linear(hidden, 1)
        state = torch.load(
            hf_hub_download(name, "sparse_linear.pt", revision=rev),
            map_location="cpu",
            weights_only=True,  # never unpickle arbitrary objects from a download
        )
        self.sparse_linear.load_state_dict(state)
        self.sparse_linear.to(self.device, dtype=dtype).eval()

        self._special_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        }
        logger.info("Loaded %s@%s on %s (%s)", name, rev[:8], self.device, dtype)

    @property
    def model_id(self) -> str:
        return self.cfg.model_id

    @property
    def dense_dim(self) -> int:
        return self.model.config.hidden_size

    @torch.inference_mode()  # no autograd bookkeeping: less memory, faster
    def _embed_batch(self, texts: list[str]) -> list[Embedding]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.model(**enc).last_hidden_state  # [batch, seq, hidden]

        dense = torch.nn.functional.normalize(hidden[:, 0].float(), dim=-1)
        weights = torch.relu(self.sparse_linear(hidden)).squeeze(-1).float()  # [batch, seq]

        results = []
        for row, ids, w in zip(dense.cpu(), enc["input_ids"].cpu(), weights.cpu(), strict=True):
            lexical: dict[int, float] = {}
            for token_id, weight in zip(ids.tolist(), w.tolist(), strict=True):
                if token_id in self._special_ids or weight <= 0:
                    continue
                if weight > lexical.get(token_id, 0.0):
                    lexical[token_id] = weight
            results.append(
                Embedding(
                    dense=row.tolist(),
                    sparse=SparseVector(indices=list(lexical), values=list(lexical.values())),
                )
            )
        return results

    def embed(self, texts: list[str]) -> list[Embedding]:
        # Sort by length so each batch pads to similar lengths (less wasted
        # compute), then restore the caller's order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        out: list[Embedding | None] = [None] * len(texts)
        size = self.cfg.batch_size
        for start in range(0, len(order), size):
            idx = order[start : start + size]
            for i, emb in zip(idx, self._embed_batch([texts[i] for i in idx]), strict=True):
                out[i] = emb
        return out  # type: ignore[return-value]
