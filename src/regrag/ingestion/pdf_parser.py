"""PDF parsing pipeline: extraction -> layout -> styles -> structure.

Extraction (this module) only records what is on the page: text, position,
dominant font, horizontal rules. Every decision is made by a later stage:

- `layout.py`:    page furniture (headers, footers, footnotes) -> block.role
- `styles.py`:    heading font styles relative to body text   -> block.heading_level
- `structure.py`: scopes, numbering tree, breadcrumbs          -> block.scope/breadcrumb
"""

import logging
import re
from collections import Counter
from pathlib import Path

import pymupdf
from pydantic import ConfigDict, Field

from regrag.ingestion.layout import LayoutConfig, annotate_layout
from regrag.ingestion.models import HLine, Model, PageInfo, ParsedBlock, ParsedDocument
from regrag.ingestion.structure import UNECE_PROFILE, DocProfile, annotate_structure
from regrag.ingestion.styles import StyleConfig, annotate_styles

logger = logging.getLogger(__name__)

# Version of the parsing *logic*. Bump it whenever a change may alter the output
# for the same PDF and config (a new rule, a fixed bug); cached results in
# data/processed/ produced by an older version are then re-parsed.
PARSER_VERSION = "0.3.0"


class ParserConfig(Model):
    """Every setting that affects parser output, in one place.

    The whole object is recorded in the processed-document manifest and is part
    of the cache key, so no setting can change without invalidating the cache.
    """

    model_config = ConfigDict(frozen=True)

    min_block_chars: int = 5  # shorter body blocks become role="noise"
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    styles: StyleConfig = Field(default_factory=StyleConfig)
    profile: DocProfile = UNECE_PROFILE


def default_doc_id(path: Path) -> str:
    """Derive a doc id from a file name: "R107r9e.pdf" -> "r107r9e"."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", path.stem.lower()).strip("-.")
    return slug or "document"


# PyMuPDF span flag: bit 4 (value 16) = bold.
_BOLD_FLAG = 16

# ToC entry: text, at least 4 dots in a row, page number at the end.
_TOC_RE = re.compile(r"\.{4,}\s*\d+\s*$")


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


def parse_pdf(
    path: Path,
    config: ParserConfig | None = None,
    doc_id: str | None = None,
) -> ParsedDocument:
    cfg = config or ParserConfig()
    doc_id = doc_id or default_doc_id(path)
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
    report = annotate_layout(blocks, pages, cfg.layout)

    for b in blocks:
        if b.role != "body":
            continue
        if _TOC_RE.search(b.text):
            b.role = "toc"
        elif len(b.text) < cfg.min_block_chars:
            b.role = "noise"
    report.role_counts = dict(Counter(b.role for b in blocks))

    body = [b for b in blocks if b.role == "body"]
    styles = annotate_styles(body, cfg.styles)
    structure = annotate_structure(body, cfg.profile)

    for i, b in enumerate(blocks):
        b.id = f"{doc_id}:{i:05d}"

    parsed = ParsedDocument(
        doc_id=doc_id,
        source_path=path.as_posix(),
        pages=pages,
        blocks=blocks,
        layout=report,
        styles=styles,
        structure=structure,
    )
    logger.info("Parsed %d blocks: %s", len(blocks), dict(report.role_counts))
    return parsed
