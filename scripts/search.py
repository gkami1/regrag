#!/usr/bin/env python3
"""Search the index: hybrid (dense + sparse, RRF) retrieval with cross-encoder reranking.

Examples:
    python scripts/search.py "minimum gangway width"
    python scripts/search.py "wheelchair space dimensions" --scope "Annex 8"
    python scripts/search.py "Class III gangway" --no-rerank --candidates

Query-time device split: the embedder encodes ONE short query (~150 ms on CPU),
the reranker scores ~20 long pairs (the heavy part), and both models do not fit
together in 4 GB of VRAM. So: embedder on CPU, reranker on GPU when available.
"""

import argparse
import logging
import os
import sys

from qdrant_client import QdrantClient

from regrag.embedding import EmbedderConfig
from regrag.retrieval import (
    BGEReranker,
    Hit,
    HybridRetriever,
    RetrievalConfig,
    SearchFilter,
)

sys.stdout.reconfigure(encoding="utf-8")


def _row(rank: int, h: Hit) -> str:
    sections = f"{h.sections[0]}..{h.sections[-1]}" if h.sections else "-"
    rerank = f"{h.rerank_score:6.2f}" if h.rerank_score is not None else "     -"
    d = f"d{h.dense_rank}" if h.dense_rank else "d-"
    s = f"s{h.sparse_rank}" if h.sparse_rank else "s-"
    return (
        f"{rank:2}. rerank {rerank}  rrf {h.rrf_score:.4f}  {d:>3} {s:>3}  "
        f"p.{h.page_start:<3} {h.scope[:22]:22} {sections}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--doc-id", action="append", help="restrict to document(s)")
    parser.add_argument("--scope", action="append", help='restrict to scope(s), e.g. "Annex 8"')
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true", help="return fused order only")
    parser.add_argument("--candidates", action="store_true", help="also list fused candidates")
    parser.add_argument(
        "--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333")
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    from regrag.embedding.bge_m3 import BGEM3Embedder  # torch import only for a real run

    client = QdrantClient(url=args.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY") or None)
    embedder = BGEM3Embedder(EmbedderConfig(device="cpu"))
    reranker = None if args.no_rerank else BGEReranker()
    retriever = HybridRetriever(client, embedder, reranker, RetrievalConfig())

    flt = SearchFilter(doc_ids=args.doc_id, scopes=args.scope)
    result = retriever.search(args.query, flt, top_k=args.top_k, rerank=not args.no_rerank)

    print(f"\nQ: {result.query}\n   timings: {result.timings_ms}\n")
    for i, hit in enumerate(result.hits, start=1):
        print(_row(i, hit))
        print(f"      {hit.header}")
        print(f"      {hit.text[:200]!r}\n")
    if args.candidates:
        print("--- fused candidates (before rerank) ---")
        for i, hit in enumerate(result.candidates, start=1):
            print(_row(i, hit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
