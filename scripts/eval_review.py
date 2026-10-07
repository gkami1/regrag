#!/usr/bin/env python3
"""Review golden-set drafts one by one: accept, edit, or reject.

Shows each draft with the actual text of its gold sections, so you judge the
label against the document, not against the author's claim. Every decision is
saved immediately (atomic rewrite), so you can quit and resume at any time.

    python scripts/eval_review.py               # all drafts
    python scripts/eval_review.py --id h001     # one item (any status)

Keys:  a accept   r reject   e edit a field   s skip   q quit
"""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from regrag.evaluation import GoldItem, gold_text, load_golden, longest_shared_run, save_golden
from regrag.ingestion.storage import DEFAULT_PROCESSED_DIR, load_document

sys.stdout.reconfigure(encoding="utf-8")
EDITABLE = ("question", "answer", "type", "lang", "gold", "notes")


def show(item: GoldItem, blocks: dict, position: str) -> None:
    print("\n" + "=" * 100)
    print(f"{position}  {item.id}  [{item.type} · {item.lang} · {item.split} · {item.source}]")
    print(f"\nQ: {item.question}")
    print(f"A: {item.answer}")
    if item.notes:
        print(f"notes: {item.notes}")
    for ref in item.gold:
        text = gold_text(ref, blocks[ref.doc_id])
        run = longest_shared_run(item.question, text)
        print(f"\n--- gold {ref.label()}  (shares {run} consecutive words with the question)")
        print(text[:1500] + (" […]" if len(text) > 1500 else ""))
    if not item.gold:
        print("\n--- no gold: unanswerable. Check that the document really does not answer it.")


def parse_gold(raw: str, doc_id: str) -> list[dict]:
    """'Annex 3|7.6.1; Annex 4|p.82' -> gold refs. Empty input -> []."""
    refs = []
    for part in filter(None, (p.strip() for p in raw.split(";"))):
        scope, _, where = (s.strip() for s in part.partition("|"))
        if where.startswith("p."):
            refs.append({"doc_id": doc_id, "scope": scope, "section": None, "page": int(where[2:])})
        else:
            refs.append({"doc_id": doc_id, "scope": scope, "section": where})
    return refs


def edit(item: GoldItem) -> GoldItem:
    field = input(f"field {EDITABLE}: ").strip()
    if field not in EDITABLE:
        print("unknown field")
        return item
    data = item.model_dump()
    if field == "gold":
        doc_id = item.gold[0].doc_id if item.gold else "unece-r107-rev9"
        print('format: "Annex 3|7.6.1; Annex 4|p.82"  (empty = no gold)')
        data["gold"] = parse_gold(input("gold: "), doc_id)
    else:
        value = input(f"new {field} (empty = null for answer): ").strip()
        data[field] = value or (None if field == "answer" else data[field])
    try:
        return GoldItem.model_validate(data)
    except ValidationError as e:
        print(f"not saved, invalid: {e}")
        return item


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    parser.add_argument("--root", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--id", help="review one item regardless of status")
    args = parser.parse_args(argv)

    items = load_golden(args.golden)
    doc_ids = {r.doc_id for i in items for r in i.gold} or {"unece-r107-rev9"}
    blocks = {d: load_document(args.root / d).blocks for d in doc_ids}

    queue = [
        i for i, it in enumerate(items) if (it.id == args.id if args.id else it.status == "draft")
    ]
    for n, idx in enumerate(queue, start=1):
        while True:
            show(items[idx], blocks, f"[{n}/{len(queue)}]")
            key = input("\n[a]ccept [r]eject [e]dit [s]kip [q]uit > ").strip().lower()
            if key == "a":
                items[idx] = items[idx].model_copy(update={"status": "verified"})
            elif key == "r":
                reason = input("reason (kept in notes): ").strip()
                notes = f"{items[idx].notes} [rejected: {reason}]".strip()
                items[idx] = items[idx].model_copy(update={"status": "rejected", "notes": notes})
            elif key == "e":
                items[idx] = edit(items[idx])
                save_golden(args.golden, items)
                continue  # show the edited item again
            elif key == "q":
                return 0
            if key in ("a", "r"):
                save_golden(args.golden, items)
            if key in ("a", "r", "s"):
                break
    print("\nNothing left to review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
