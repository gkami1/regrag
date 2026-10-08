#!/usr/bin/env python3
"""Evaluate generated answers end to end: refusals, citations, context, judged correctness.

Needs the vLLM server reachable (SSH tunnel) and DEEPSEEK_API_KEY for the judge.
Settings are read from .env. Judge verdicts are cached in data/eval_runs/judge_cache.jsonl,
so re-running never re-pays for unchanged answers.

    python scripts/eval_generation.py --limit 3            # smoke test first
    python scripts/eval_generation.py                      # dev split
    python scripts/eval_generation.py --consistency 5      # + re-judge 5 answers uncached
    python scripts/eval_generation.py --no-judge           # pipeline + mechanical checks only
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from regrag.evaluation import load_golden
from regrag.evaluation.generation_eval import evaluate, judge_consistency, summarize
from regrag.evaluation.judge import JUDGE_PROMPT_VERSION, JudgeConfig, LLMJudge
from regrag.evaluation.labels import agreement, load_labels
from regrag.generation.factory import build_vllm_pipeline
from regrag.generation.prompt import PROMPT_VERSION
from regrag.ingestion.storage import DEFAULT_PROCESSED_DIR, load_document

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()
RUNS = Path("data/eval_runs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--consistency", type=int, default=0, help="re-judge N answers uncached")
    parser.add_argument("--judge-model", default=JudgeConfig().model)
    parser.add_argument("--thinking", action="store_true", help="answerer thinking mode")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    items = [
        i
        for i in load_golden(args.golden)
        if i.status == "verified" and (args.split == "all" or i.split == args.split)
    ][: args.limit]
    doc_ids = {r.doc_id for i in items for r in i.gold} or {"unece-r107-rev9"}
    blocks = {d: load_document(DEFAULT_PROCESSED_DIR / d).blocks for d in doc_ids}

    judge = None
    if not args.no_judge:
        from openai import OpenAI

        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            parser.error("DEEPSEEK_API_KEY is not set (add it to .env) or use --no-judge")
        cfg = JudgeConfig(model=args.judge_model)
        judge = LLMJudge(
            OpenAI(base_url=cfg.base_url, api_key=key), cfg, RUNS / "judge_cache.jsonl"
        )

    pipeline = build_vllm_pipeline(enable_thinking=args.thinking)

    def progress(n, total, qid):
        print(f"\r{n}/{total} {qid}   ", end="", flush=True)

    results, answers = evaluate(items, pipeline, judge, blocks, progress)
    print()
    summary = summarize(results)

    consistency = None
    if judge and args.consistency:
        golden_by_id = {i.id: i for i in items}
        consistency = judge_consistency(
            results, golden_by_id, judge, blocks, answers, args.consistency
        )

    golden_sha = hashlib.sha256(args.golden.read_bytes()).hexdigest()[:12]
    print(
        f"\n=== generation eval  split={args.split}  golden={golden_sha}  "
        f"prompt=v{PROMPT_VERSION}  judge={args.judge_model if judge else '-'} "
        f"v{JUDGE_PROMPT_VERSION}"
    )
    for field, value in summary.model_dump().items():
        print(f"  {field:24} {value}")
    if consistency is not None:
        print(f"  {'judge_consistency':24} {consistency}  (same verdict when re-judged)")
    if judge:
        print(f"  {'judge_api_calls':24} {judge.calls}  (the rest came from the cache)")

    agree = agreement(results, load_labels())
    if agree.n:
        print(
            f"\n  judge vs human on {agree.n} labelled answers: verdict {agree.verdict}, "
            f"overstated {agree.overstated}, missing_condition {agree.missing_condition}"
        )
        for d in agree.disagreements:
            print(f"    {d}")

    print("\n  not fully correct:")
    for r in results:
        if r.score is None or r.score < 1:
            flags = ",".join(
                f
                for f in ("overstated", "missing_condition", "contradicts_reference")
                if r.judge and getattr(r.judge, f)
            )
            why = r.judge.explanation[:90] if r.judge else (r.error or "")
            print(f"    {r.id:5} {r.outcome:16} score={r.score} {flags:20} {why}")

    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = RUNS / f"{stamp}_gen_{args.split}.json"
    out.write_text(
        json.dumps(
            {
                "created_at": stamp,
                "golden_sha256": golden_sha,
                "split": args.split,
                "prompt_version": PROMPT_VERSION,
                "answer_model": os.environ.get("VLLM_MODEL"),
                "judge": JudgeConfig(model=args.judge_model).model_dump() if judge else None,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "summary": summary.model_dump(),
                "judge_consistency": consistency,
                "questions": [r.model_dump() for r in results],
                "answers": {k: v.model_dump() for k, v in answers.items()},
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
