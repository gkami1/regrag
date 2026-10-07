#!/usr/bin/env python3
"""Evaluate retrieval on the golden set: recall@k and MRR per stage, diagnoses, thresholds.

Only `verified` items count unless --include-drafts is given (then results are
labelled preliminary). Tune on the dev split; look at test rarely.

    python scripts/eval_retrieval.py                      # dev split, reranked
    python scripts/eval_retrieval.py --split test
    python scripts/eval_retrieval.py --no-rerank          # fused order as final output
    python scripts/eval_retrieval.py --include-drafts     # before review is finished

Full per-question results go to data/eval_runs/<timestamp>.json.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient

from regrag.embedding import EmbedderConfig
from regrag.evaluation import load_golden
from regrag.evaluation.runner import (
    KS,
    STAGES,
    evaluate_question,
    summarize,
    summarize_by,
    threshold_analysis,
)
from regrag.retrieval import BGEReranker, HybridRetriever, RetrievalConfig

sys.stdout.reconfigure(encoding="utf-8")


def _table(rows: dict, title: str) -> None:
    head = "  ".join(f"R@{k:<4}" for k in KS)
    print(f"\n{title:22} {head}  MRR    n")
    for name, (s, n) in rows.items():
        cells = "  ".join(f"{s.recall[k]:.3f}" for k in KS)
        print(f"{name:22} {cells}  {s.mrr:.3f}  {n}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--include-drafts", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("data/eval_runs"))
    parser.add_argument(
        "--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333")
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    allowed = {"verified", "draft"} if args.include_drafts else {"verified"}
    items = [
        i
        for i in load_golden(args.golden)
        if i.status in allowed and (args.split == "all" or i.split == args.split)
    ]
    if not items:
        print("No items to evaluate (review drafts first, or use --include-drafts).")
        return 1

    from regrag.embedding.bge_m3 import BGEM3Embedder  # torch only for a real run

    config = RetrievalConfig()
    retriever = HybridRetriever(
        QdrantClient(url=args.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY") or None),
        BGEM3Embedder(EmbedderConfig(device="cpu")),
        None if args.no_rerank else BGEReranker(),
        config,
    )
    rerank = not args.no_rerank

    results = []
    for n, item in enumerate(items, start=1):
        print(f"\r{n}/{len(items)} {item.id}", end="", flush=True)
        results.append(evaluate_question(retriever, item, rerank))
    print()

    golden_sha = hashlib.sha256(args.golden.read_bytes()).hexdigest()[:12]
    label = "PRELIMINARY (includes drafts)" if args.include_drafts else "verified only"
    answerable = [r for r in results if r.stages]
    print(f"\n=== split={args.split}  rerank={rerank}  golden={golden_sha}  {label}")
    print(f"{len(results)} questions, {len(answerable)} answerable")

    overall = summarize(results)
    _table({s: (overall[s], len(answerable)) for s in STAGES}, "stage")
    final = "reranked" if rerank else "fused"
    for key in ("type", "lang"):
        groups = summarize_by(results, key, final)
        counts = {g: sum(1 for r in answerable if getattr(r, key) == g) for g in groups}
        _table({g: (s, counts[g]) for g, s in groups.items()}, f"{final} by {key}")

    print(
        "\ndiagnosis:",
        {
            d: sum(r.diagnosis == d for r in answerable)
            for d in sorted({r.diagnosis for r in answerable})
        },
    )
    for r in answerable:
        if r.diagnosis != "ok":
            print(f"  {r.diagnosis:15} {r.id} [{r.type}] {r.question[:70]}")

    th = threshold_analysis(results) if rerank else None
    if th and th.unanswerable_top1:
        print(f"\ntop-1 rerank score  answerable: {[round(s, 2) for s in th.answerable_top1]}")
        print(f"                  unanswerable: {[round(s, 2) for s in th.unanswerable_top1]}")
        acc = th.best_balanced_accuracy
        print(f"best threshold {th.best_threshold:.2f}  balanced accuracy {acc:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out / f"{stamp}_{args.split}{'_norerank' if not rerank else ''}.json"
    out.write_text(
        json.dumps(
            {
                "created_at": stamp,
                "golden_sha256": golden_sha,
                "split": args.split,
                "include_drafts": args.include_drafts,
                "rerank": rerank,
                "config": config.model_dump(mode="json"),
                "summary": {s: v.model_dump() for s, v in overall.items()},
                "threshold": th.model_dump() if th else None,
                "questions": [r.model_dump() for r in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
