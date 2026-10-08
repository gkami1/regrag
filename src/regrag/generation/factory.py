"""Build the production pipeline from environment settings, in one place.

Used by every entry point (ask CLI, generation evaluation, later the API), so
they cannot drift apart: an evaluation that builds its pipeline differently from
the product measures something else.

Settings come from the environment, typically loaded from `.env`:
    VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY, QDRANT_URL, QDRANT_API_KEY
"""

import os
from pathlib import Path

from regrag.generation.answerer import VLLMAnswerer, VLLMConfig
from regrag.generation.context import ChunkStore
from regrag.generation.models import GenerationConfig
from regrag.generation.pipeline import RAGPipeline
from regrag.ingestion.storage import DEFAULT_PROCESSED_DIR


def build_vllm_pipeline(
    *,
    model: str | None = None,
    base_url: str | None = None,
    enable_thinking: bool = False,
    qdrant_url: str | None = None,
    processed_root: Path = DEFAULT_PROCESSED_DIR,
    config: GenerationConfig | None = None,
) -> RAGPipeline:
    # Heavy imports here, so importing this module stays cheap.
    from openai import OpenAI
    from qdrant_client import QdrantClient

    from regrag.embedding import EmbedderConfig
    from regrag.embedding.bge_m3 import BGEM3Embedder
    from regrag.retrieval import BGEReranker, HybridRetriever

    model = model or os.environ.get("VLLM_MODEL")
    if not model:
        raise ValueError("set VLLM_MODEL (the model name the vLLM server serves)")
    llm = OpenAI(
        base_url=base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    )
    retriever = HybridRetriever(
        QdrantClient(
            url=qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY") or None,
        ),
        # Query-time device split: one short query on CPU, reranker on GPU.
        BGEM3Embedder(EmbedderConfig(device="cpu")),
        BGEReranker(),
    )
    return RAGPipeline(
        retriever,
        ChunkStore.from_processed(processed_root),
        VLLMAnswerer(llm, VLLMConfig(model=model, enable_thinking=enable_thinking)),
        config or GenerationConfig(),
    )
