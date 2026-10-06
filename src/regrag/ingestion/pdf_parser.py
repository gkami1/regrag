"""PDF parsing: extraction -> layout analysis -> block classification.

Extraction only records what is on the page (text, position, dominant font, rules).
Furniture detection lives in `layout.py`. The title/subtitle heuristics below are
still R107-tuned and will be replaced by document-relative style analysis.
"""

import logging
import re
from collections import Counter
from pathlib import Path

import pymupdf

from regrag.ingestion.layout import LayoutConfig, annotate_layout
from regrag.ingestion.models import HLine, PageInfo, ParsedBlock, ParsedDocument

logger = logging.getLogger(__name__)

# PyMuPDF span flag: bit 4 (value 16) = bold.
_BOLD_FLAG = 16

# ToC entry: text, at least 4 dots in a row, page number at the end.
_TOC_RE = re.compile(r"\.{4,}\s*\d+\s*$")

# Numbering: "1.", "1.1.", "3.10.12."
_NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.\s+\S")

# Annex heading: "Annex 12", "Annex 1 - Part 1 - Appendix 2"
_ANNEX_RE = re.compile(r"^\s*Annex\s+\d+", re.IGNORECASE)


def _normalize(text: str) -> str:
    """Collapse whitespace and join multi-line blocks into one line."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def _dominant_font(block: dict) -> tuple[float, bool]:
    """Font size covering the most characters, and whether most characters are bold.

    Weighting by characters (not spans) keeps a 1-char superscript footnote marker
    from dragging a 10pt paragraph's size down.
    """
    sizes: Counter[float] = Counter()
    bold_chars = total_chars = 0
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            n = len(span.get("text", "").strip())
            if not n:
                continue
            sizes[round(span.get("size", 0.0), 1)] += n
            total_chars += n
            if span.get("flags", 0) & _BOLD_FLAG:
                bold_chars += n
    if not total_chars:
        return 0.0, False
    return sizes.most_common(1)[0][0], bold_chars / total_chars > 0.5


def _extract_hlines(page: pymupdf.Page) -> list[HLine]:
    """Thin horizontal drawings: rules, separators, table borders."""
    lines = []
    for drawing in page.get_drawings():
        r = drawing["rect"]
        if r.height < 1.5 and r.width > 1:
            lines.append(HLine(y=r.y0, x0=r.x0, x1=r.x1))
    return lines


def _extract_blocks(page: pymupdf.Page, page_number: int) -> list[ParsedBlock]:
    blocks = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:  # 0 = text, 1 = image
            continue
        raw = "\n".join(
            "".join(span.get("text", "") for span in line.get("spans", []))
            for line in block.get("lines", [])
        )
        text = _normalize(raw)
        if not text:
            continue
        size, bold = _dominant_font(block)
        blocks.append(
            ParsedBlock(
                text=text,
                page_number=page_number,
                bbox=tuple(round(v, 1) for v in block["bbox"]),
                font_size=size,
                is_bold=bold,
            )
        )
    return blocks


def _extract_section_number(text: str) -> str | None:
    m = _NUMBERED_RE.match(text)
    return m.group(1) if m else None


def _classify(text: str, size: float, is_bold: bool) -> tuple[str, str | None]:
    section_number = _extract_section_number(text)

    # 1. Large bold: top-level heading.
    if size >= 13 and is_bold:
        return "title", section_number
    # 2. Medium bold: subheading.
    if size >= 12 and is_bold:
        return "subtitle", section_number
    # 3. Short bold numbered: section heading ("1. Scope").
    if section_number and is_bold and len(text) < 120:
        return "title", section_number
    # 4. Unnumbered annex heading.
    if _ANNEX_RE.match(text) and len(text) < 150:
        return "title", None
    # 5. Short numbered line without a final period: looks like a heading
    #    ("2.2. Definition of type(s)", but not "1.2.3. Off-road vehicles.").
    if section_number and len(text) < 60 and not text.rstrip().endswith("."):
        return "subtitle", section_number
    # 6. Body text (keep the section number!).
    return "text", section_number


def parse_pdf(
    path: Path,
    min_block_chars: int = 5,
    layout_config: LayoutConfig | None = None,
) -> ParsedDocument:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    with pymupdf.open(path) as doc:
        logger.info("Parsing %s (%d pages)", path.name, len(doc))
        pages: list[PageInfo] = []
        blocks: list[ParsedBlock] = []
        for page_number, page in enumerate(doc, start=1):
            pages.append(
                PageInfo(
                    number=page_number,
                    width=page.rect.width,
                    height=page.rect.height,
                    hlines=_extract_hlines(page),
                )
            )
            blocks.extend(_extract_blocks(page, page_number))

    # Layout runs on *all* blocks: short ones like page numbers are exactly the
    # repeated evidence it needs, so the length filter must come after it.
    report = annotate_layout(blocks, pages, layout_config)

    for b in blocks:
        if b.role != "body":
            continue
        if _TOC_RE.search(b.text):
            b.role = "toc"
        elif len(b.text) < min_block_chars:
            b.role = "noise"
        else:
            b.block_type, b.section_number = _classify(b.text, b.font_size, b.is_bold)

    report.role_counts = Counter(b.role for b in blocks)
    parsed = ParsedDocument(source_path=str(path), pages=pages, blocks=blocks, layout=report)
    logger.info("Parsed %d blocks: %s", len(blocks), dict(report.role_counts))
    return parsed
