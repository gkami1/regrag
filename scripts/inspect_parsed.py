#!/usr/bin/env python3
"""Inspect parsed PDF: show block structure for manual review."""

import sys
from collections import Counter
from pathlib import Path

from regrag.ingestion import parse_pdf


def main(pdf_path: str, limit: int = 60) -> None:
    parsed = parse_pdf(Path(pdf_path))

    print(f"\n=== {pdf_path} ===")
    print(f"Pages: {parsed.total_pages}, Blocks: {len(parsed.blocks)}")

    types = Counter(b.block_type for b in parsed.blocks)
    print(f"Block types: {dict(types)}")

    fonts = Counter((round(b.font_size, 1), b.is_bold) for b in parsed.blocks)
    print("\nFont distribution (size, bold) -> count:")
    for (size, bold), count in fonts.most_common(15):
        print(f"  size={size:5.1f}  bold={bold!s:5}  count={count}")

    print(f"\n=== First {limit} blocks ===")
    for i, b in enumerate(parsed.blocks[:limit]):
        sec = f"[{b.section_number}]" if b.section_number else ""
        print(
            f"[{i:04d}] p.{b.page_number:3d} {b.block_type:8s} "
            f"sz={b.font_size:4.1f} bold={int(b.is_bold)} {sec:10s} | {b.text[:90]}"
        )

    print("\n=== All detected titles ===")
    for i, b in enumerate(parsed.blocks):
        if b.block_type in ("title", "subtitle"):
            sec = f"[{b.section_number}]" if b.section_number else ""
            print(f"[{i:04d}] p.{b.page_number:3d} {b.block_type:8s} {sec:10s} | {b.text[:90]}")


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/raw/r107r9e.pdf"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    main(pdf, limit)