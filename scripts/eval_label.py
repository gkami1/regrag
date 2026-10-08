#!/usr/bin/env python3
"""Label answers by hand to calibrate the LLM judge.

Shows a deterministic sample of judged answers from a generation run, WITHOUT the
judge's verdict (so it cannot anchor you), and records your verdict to
evals/judge_labels.jsonl. Then prints how often the judge agrees with you.

    python scripts/eval_label.py                 # latest generation run, 10 answers
    python scripts/eval_label.py --n 15 --run data/eval_runs/<file>.json
"""

import argparse
import json
import sys
from pathlib import Path

from regrag.evaluation.generation_eval import GenQuestionResult
from regrag.evaluation.labels import (
    HumanLabel,
    agreement,
    append_label,
    load_labels,
    sample_for_labelling,
)

sys.stdout.reconfigure(encoding="utf-8")
VERDICTS = {"c": "correct", "p": "partial", "i": "incorrect"}


def ask_bool(prompt: str) -> bool:
    return input(f"{prompt} [y/N] ").strip().lower() == "y"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, help="default: latest *_gen_*.json")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args(argv)

    run_path = args.run or max(Path("data/eval_runs").glob("*_gen_*.json"), default=None)
    if run_path is None:
        print("No generation run found; run scripts/eval_generation.py first.")
        return 1
    run = json.loads(run_path.read_text(encoding="utf-8"))
    results = [GenQuestionResult.model_validate(q) for q in run["questions"]]
    golden = {
        json.loads(line)["id"]: json.loads(line)
        for line in Path("evals/golden.jsonl").read_text(encoding="utf-8").splitlines()
    }
    done = {(lb.id, lb.answer_hash) for lb in load_labels()}

    todo = [r for r in sample_for_labelling(results, args.n) if (r.id, r.answer_hash) not in done]
    print(f"{run_path}: {len(todo)} answers to label ({args.n - len(todo)} already labelled)")
    for n, r in enumerate(todo, start=1):
        g = golden[r.id]
        print("\n" + "=" * 100)
        print(f"[{n}/{len(todo)}] {r.id} [{r.type}]\nQ: {r.question}")
        print(f"\nREFERENCE: {g['answer']}")
        print(f"\nSYSTEM:    {r.answer}")
        for c in r.citations:
            mark = "✓" if c["verified"] else "✗"
            print(f'  {mark} [{c["chunk_label"]}] {c["scope"]} {c["sections"][:1]}: "{c["quote"]}"')
        key = ""
        while key not in VERDICTS and key != "q":
            key = input("\nverdict: [c]orrect [p]artial [i]ncorrect  [q]uit > ").strip().lower()
        if key == "q":
            break
        append_label(
            HumanLabel(
                id=r.id,
                answer_hash=r.answer_hash,
                verdict=VERDICTS[key],
                overstated=ask_bool("overstated (claims more than the regulation says)?"),
                missing_condition=ask_bool("missing a condition / class distinction?"),
                notes=input("notes (optional): ").strip(),
            )
        )

    agree = agreement(results, load_labels())
    if agree.n:
        print(
            f"\njudge vs you on {agree.n} answers: verdict agreement {agree.verdict}, "
            f"overstated {agree.overstated}, missing_condition {agree.missing_condition}"
        )
        for d in agree.disagreements:
            print(f"  {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
