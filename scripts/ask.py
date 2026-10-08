#!/usr/bin/env python3
"""Ask a question: hybrid retrieval -> context with references -> self-hosted LLM (vLLM).

The vLLM server is reached through its OpenAI-compatible API. On a rented GPU,
keep the port closed and tunnel it:  ssh -L 8000:localhost:8000 user@gpu-host

    python scripts/ask.py "How many service doors does a 80-passenger Class I bus need?"
    python scripts/ask.py "Минимальная ширина сиденья?" --show-context

Settings (flags override environment; .env is loaded automatically):
    VLLM_BASE_URL   default http://localhost:8000/v1
    VLLM_MODEL      the served model name, e.g. Qwen/Qwen3.8-27B-FP8
    VLLM_API_KEY    the --api-key the server was started with
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from regrag.generation.factory import build_vllm_pipeline
from regrag.retrieval import SearchFilter

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()  # .env -> os.environ (never overrides variables already set in the shell)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--base-url", help="default: $VLLM_BASE_URL")
    parser.add_argument("--model", help="default: $VLLM_MODEL")
    parser.add_argument("--thinking", action="store_true", help="enable the model's thinking mode")
    parser.add_argument("--doc-id", action="append")
    parser.add_argument("--scope", action="append")
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--qdrant-url", help="default: $QDRANT_URL")
    args = parser.parse_args(argv)
    if not (args.model or os.environ.get("VLLM_MODEL")):
        parser.error("set --model or VLLM_MODEL to the model name the vLLM server serves")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    pipeline = build_vllm_pipeline(
        model=args.model,
        base_url=args.base_url,
        enable_thinking=args.thinking,
        qdrant_url=args.qdrant_url,
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
