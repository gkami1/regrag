#!/usr/bin/env python3
"""Ask a question: hybrid retrieval -> context with references -> self-hosted LLM (vLLM).

The vLLM server is reached through its OpenAI-compatible API. On a rented GPU,
keep the port closed and tunnel it:  ssh -L 8000:localhost:8000 user@gpu-host

    python scripts/ask.py "How many service doors does a 80-passenger Class I bus need?"
    python scripts/ask.py "Минимальная ширина сиденья?" --show-context

Settings (flags override environment):
    VLLM_BASE_URL   default http://localhost:8000/v1
    VLLM_MODEL      the served model name, e.g. Qwen/Qwen3.8-27B-FP8
    VLLM_API_KEY    the --api-key the server was started with
"""

import argparse
import logging
import os
import sys

from qdrant_client import QdrantClient

from regrag.embedding import EmbedderConfig
from regrag.generation import ChunkStore, GenerationConfig, RAGPipeline, VLLMAnswerer, VLLMConfig
from regrag.ingestion.storage import DEFAULT_PROCESSED_DIR
from regrag.retrieval import BGEReranker, HybridRetriever, SearchFilter

sys.stdout.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument(
        "--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    )
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL"))
    parser.add_argument("--thinking", action="store_true", help="enable the model's thinking mode")
    parser.add_argument("--doc-id", action="append")
    parser.add_argument("--scope", action="append")
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument(
        "--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333")
    )
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("set --model or VLLM_MODEL to the model name the vLLM server serves")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    from openai import OpenAI  # imported here: only this entry point needs the client library

    from regrag.embedding.bge_m3 import BGEM3Embedder

    llm = OpenAI(base_url=args.base_url, api_key=os.environ.get("VLLM_API_KEY", "EMPTY"))
    retriever = HybridRetriever(
        QdrantClient(url=args.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY") or None),
        BGEM3Embedder(EmbedderConfig(device="cpu")),
        BGEReranker(),
    )
    pipeline = RAGPipeline(
        retriever,
        ChunkStore.from_processed(DEFAULT_PROCESSED_DIR),
        VLLMAnswerer(llm, VLLMConfig(model=args.model, enable_thinking=args.thinking)),
        GenerationConfig(),
    )
    answer = pipeline.ask(args.question, SearchFilter(doc_ids=args.doc_id, scopes=args.scope))

    status = "COVERED" if answer.covered else "NOT COVERED"
    if answer.error:
        status = f"ERROR ({answer.error}, finish_reason={answer.finish_reason})"
    elif not answer.grounded:
        status += "  ⚠ no verified citation"
    print(f"\nQ: {answer.question}\n[{status}]\n\n{answer.text}\n")
    for c in answer.citations:
        mark = "✓" if c.verified else "✗ NOT FOUND IN SOURCE"
        where = f"{c.scope}, {c.sections[0] if c.sections else '-'}, p.{c.page_start}"
        print(f'  [{c.chunk_label}] {mark}  {where}\n       "{c.quote}"')
    if args.show_context:
        print("\ncontext:")
        for c in answer.context:
            extra = f"  via {c.via!r}" if c.via else f"  rerank {c.rerank_score:.2f}"
            print(f"  {c.label:4} {c.origin:9} {c.source()}{extra}")
    u = answer.usage
    print(f"\n{answer.model} · prompt {u.prompt_tokens} + completion {u.completion_tokens} tokens")
    print(f"timings: {answer.timings_ms}")
    return 0 if not answer.error else 1


if __name__ == "__main__":
    sys.exit(main())
