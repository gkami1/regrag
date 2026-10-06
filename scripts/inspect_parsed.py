#!/usr/bin/env python3
"""Inspect a parsed PDF for manual review: layout, styles, outline, breadcrumbs.

Usage: python scripts/inspect_parsed.py [PDF] [PAGE]
    PAGE: also print every body block on that page with its breadcrumb.
"""

import sys
from collections import Counter
from pathlib import Path

from regrag.ingestion import ParsedDocument, parse_pdf

# Windows consoles default to a legacy code page; regulation text has ≤, °, Ω, ...
sys.stdout.reconfigure(encoding="utf-8")


def print_layout(parsed: ParsedDocument) -> None:
    report = parsed.layout
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
        for text, count in distinct.most_common(5):
            print(f"  x{count:<3} {text!r}")


def print_structure(parsed: ParsedDocument) -> None:
    body = parsed.body_blocks()
    styles, structure = parsed.styles, parsed.structure

    print("\n=== Styles ===")
    print(f"Body style: {styles.body_style}")
    print(f"Heading levels: {styles.heading_levels}")

    print("\n=== Structure ===")
    print(f"Block types (body): {dict(Counter(b.block_type for b in body))}")
    print(f"Numbered blocks: {structure.numbered_blocks}")
    print(f"Scopes ({len(structure.scopes)}):")
    for s in structure.scopes:
        print(f"  p.{s.first_page:3d} {s.id:32s} {s.title[:70]}")
    print(f"Backward jumps ({len(structure.backward_jumps)}):")
    for j in structure.backward_jumps[:20]:
        print(f"  {j}")

    print("\n=== Outline (numbered headings, indented by depth) ===")
    scope = None
    for b in body:
        if b.scope != scope:
            scope = b.scope
            print(f"[{scope}]")
        if b.block_type == "heading" and b.section_number:
            depth = b.section_number.count(".")
            print(f"  p.{b.page_number:3d} {'  ' * depth}{b.text[:80]}")


def print_page(parsed: ParsedDocument, page: int) -> None:
    print(f"\n=== Body blocks on page {page} with breadcrumbs ===")
    for b in parsed.body_blocks():
        if b.page_number == page:
            print(f"{b.block_type:7s} | {b.text[:70]}")
            print(f"        > {' > '.join(b.breadcrumb)}")


def main(pdf_path: str, page: int | None = None) -> None:
    parsed = parse_pdf(Path(pdf_path))
    print(f"\n=== {pdf_path} ===")
    print(f"Pages: {parsed.total_pages}, Blocks: {len(parsed.blocks)}")
    print_layout(parsed)
    print_structure(parsed)
    if page:
        print_page(parsed, page)


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/raw/R107r9e.pdf"
    page = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(pdf, page)
