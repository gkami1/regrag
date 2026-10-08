"""Question in, cited answer out: retrieval -> context assembly -> LLM."""

import time

from regrag.generation.answerer import Answerer
from regrag.generation.context import ChunkStore, build_context
from regrag.generation.models import Answer, GenerationConfig
from regrag.retrieval.models import SearchFilter
from regrag.retrieval.retriever import HybridRetriever


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        store: ChunkStore,
        answerer: Answerer,
        config: GenerationConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.store = store
        self.answerer = answerer
        self.cfg = config or GenerationConfig()

    def ask(self, question: str, flt: SearchFilter | None = None) -> Answer:
        retrieval = self.retriever.search(question, flt, top_k=self.cfg.context_k)
        t0 = time.perf_counter()
        context = build_context(retrieval.hits, self.store, self.cfg)
        context_ms = (time.perf_counter() - t0) * 1000

        answer = self.answerer.answer(question, context)
        answer.timings_ms = {
            **retrieval.timings_ms,
            "context_ms": round(context_ms, 1),
            **answer.timings_ms,
        }
        return answer
