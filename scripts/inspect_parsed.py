#!/usr/bin/env python3
"""Inspect parsed PDF: layout decisions and block structure, for manual review."""

import sys
from collections import Counter
from pathlib import Path

from regrag.ingestion import parse_pdf

# Windows consoles default to a legacy code page; regulation text has ≤, °, Ω, ...
sys.stdout.reconfigure(encoding="utf-8")


def main(pdf_path: str, limit: int = 60) -> None:
    parsed = parse_pdf(Path(pdf_path))
    body = parsed.body_blocks()
    report = parsed.layout

    print(f"\n=== {pdf_path} ===")
    print(f"Pages: {parsed.total_pages}, Blocks: {len(parsed.blocks)} (body: {len(body)})")

    print("\n=== Layout ===")
    print(f"Body font size: {report.body_font_size}pt")
    print(f"Header zone: top {report.header_depth}pt, footer zone: bottom {report.footer_depth}pt")
    print("Repeated patterns:")
    for p in report.repeated_patterns:
        print(f"  {p}")
    print(f"Roles: {dict(report.role_counts)}")
    print(f"Footnotes found on pages: {report.footnote_pages}")

    for role in ("header", "footer", "footnote", "toc", "noise"):
        examples = [b for b in parsed.blocks if b.role == role]
        if not examples:
            continue
        distinct = Counter(b.text[:70] for b in examples)
        print(f"\n--- {role} ({len(examples)} blocks, {len(distinct)} distinct) ---")
        for text, count in distinct.most_common(8):
            print(f"  x{count:<3} {text!r}")

    print(f"\nBlock types (body): {dict(Counter(b.block_type for b in body))}")
    fonts = Counter((b.font_size, b.is_bold) for b in body)
    print("\nFont distribution (size, bold) -> count:")
    for (size, bold), count in fonts.most_common(15):
        print(f"  size={size:5.1f}  bold={bold!s:5}  count={count}")

    print(f"\n=== First {limit} body blocks ===")
    for i, b in enumerate(body[:limit]):
        sec = f"[{b.section_number}]" if b.section_number else ""
        print(
            f"[{i:04d}] p.{b.page_number:3d} {b.block_type:8s} "
            f"sz={b.font_size:4.1f} bold={int(b.is_bold)} {sec:10s} | {b.text[:90]}"
        )

    print("\n=== All detected titles ===")
    for i, b in enumerate(body):
        if b.block_type in ("title", "subtitle"):
            sec = f"[{b.section_number}]" if b.section_number else ""
            print(f"[{i:04d}] p.{b.page_number:3d} {b.block_type:8s} {sec:10s} | {b.text[:90]}")


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/raw/R107r9e.pdf"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    main(pdf, limit)
