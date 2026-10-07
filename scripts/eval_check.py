#!/usr/bin/env python3
"""Validate evals/golden.jsonl: schema, gold sections exist, leakage, mix of types.

Run after every edit to the golden set (and in CI). Exit code 1 on any problem.

    python scripts/eval_check.py
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from regrag.evaluation import check_against_corpus, gold_text, load_golden, longest_shared_run
from regrag.ingestion.storage import DEFAULT_PROCESSED_DIR, StorageError, load_document

sys.stdout.reconfigure(encoding="utf-8")
LEAK_WARN = 4  # consecutive words shared with the gold text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    parser.add_argument("--root", type=Path, default=DEFAULT_PROCESSED_DIR)
    args = parser.parse_args(argv)

    try:
        items = load_golden(args.golden)
    except StorageError as e:
        print(f"INVALID: {e}")
        return 1
    doc_ids = {ref.doc_id for item in items for ref in item.gold}
    blocks = {d: load_document(args.root / d).blocks for d in doc_ids}
    problems = check_against_corpus(items, blocks)

    print(f"{len(items)} items in {args.golden}")
    for field in ("status", "type", "lang", "source", "split"):
        print(f"  {field:7} {dict(Counter(getattr(i, field) for i in items))}")

    leaks = []
    for item in items:
        text = "\n".join(gold_text(ref, blocks[ref.doc_id]) for ref in item.gold)
        run = longest_shared_run(item.question, text) if text else 0
        if run >= LEAK_WARN:
            leaks.append(f"{item.id}: shares {run} consecutive words with its gold text")

    for p in problems:
        print(f"PROBLEM  {p}")
    for w in leaks:
        print(f"LEAKAGE  {w}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
