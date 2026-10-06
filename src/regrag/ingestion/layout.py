"""Layout analysis: tag page furniture (running headers, footers, footnotes).

Document-agnostic by design: every decision comes from statistics of the document
itself, never from strings specific to one publisher.

1. Repetition. A block near the top/bottom edge whose digit-normalised text recurs
   at the same distance from that edge on many pages is a running header/footer.
   Normalising digits makes "Page 5" and "Page 47" the same pattern.
2. Margin zones. The repeated blocks tell us how deep the header/footer zones are;
   any other block lying entirely inside a zone is furniture too (a one-off stamp,
   a header line that changes per annex, ...).
3. Footnotes. They don't repeat, so they need their own evidence: they sit below an
   *isolated* short horizontal rule (the classic footnote separator) AND are set in
   a smaller font than body text. Table borders are also short rules, but come in
   groups at the same height, so they are not mistaken for separators.
"""

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from regrag.ingestion.models import HLine, LayoutReport, PageInfo, ParsedBlock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayoutConfig:
    # Only blocks within this fraction of the page height from an edge can be furniture.
    edge_ratio: float = 0.15
    # Vertical tolerance (pt) when matching positions across pages.
    band: float = 5.0
    # A pattern is "running" if it occurs on at least this share of pages...
    min_repeat_ratio: float = 0.3
    # ...and on at least this many pages (protects short documents).
    min_repeat_pages: int = 3
    # Footnote separator width, as a fraction of the page width.
    separator_min_width_ratio: float = 0.05
    separator_max_width_ratio: float = 0.25
    # Footnote text must be at least this many pt smaller than body text.
    footnote_size_delta: float = 0.5


_DIGITS_RE = re.compile(r"\d+")
_SPACE_RE = re.compile(r"\s+")


def normalize_pattern(text: str) -> str:
    """Collapse the parts of a line that change from page to page."""
    text = _DIGITS_RE.sub("#", text.lower())
    return _SPACE_RE.sub(" ", text).strip()[:80]


def body_font_size(blocks: list[ParsedBlock]) -> float:
    """Most common font size, weighted by characters: the document's baseline."""
    weights: Counter[float] = Counter()
    for b in blocks:
        weights[round(b.font_size, 1)] += len(b.text)
    return weights.most_common(1)[0][0] if weights else 0.0


def _edge_key(block: ParsedBlock, page: PageInfo, cfg: LayoutConfig) -> tuple[str, int] | None:
    """Which edge a block hugs and its distance from it, quantised to `band`.

    Distance is measured from the edge (not as an absolute y) so that pages of
    different heights, e.g. landscape table pages, still line up.
    """
    limit = cfg.edge_ratio * page.height
    if block.y0 < limit:
        return "top", round(block.y0 / cfg.band)
    if page.height - block.y1 < limit:
        return "bottom", round((page.height - block.y1) / cfg.band)
    return None


def _tag_running_furniture(
    blocks: list[ParsedBlock],
    pages: dict[int, PageInfo],
    cfg: LayoutConfig,
) -> tuple[float, float, list[str]]:
    """Steps 1 and 2. Returns (header_depth, footer_depth, repeated patterns)."""
    occurrences: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    keyed: list[tuple[ParsedBlock, tuple[str, int, str]]] = []
    for b in blocks:
        edge = _edge_key(b, pages[b.page_number], cfg)
        if edge is None:
            continue
        key = (*edge, normalize_pattern(b.text))
        occurrences[key].add(b.page_number)
        keyed.append((b, key))

    threshold = max(cfg.min_repeat_pages, math.ceil(cfg.min_repeat_ratio * len(pages)))
    repeated = {k for k, pgs in occurrences.items() if len(pgs) >= threshold}

    header_depth = footer_depth = 0.0
    for b, key in keyed:
        if key not in repeated:
            continue
        if key[0] == "top":
            b.role = "header"
            header_depth = max(header_depth, b.y1)
        else:
            b.role = "footer"
            footer_depth = max(footer_depth, pages[b.page_number].height - b.y0)

    # Zone rule: anything fully inside a learned margin is furniture as well.
    for b in blocks:
        if b.role != "body":
            continue
        page = pages[b.page_number]
        if header_depth and b.y1 <= header_depth + cfg.band:
            b.role = "header"
        elif footer_depth and page.height - b.y0 <= footer_depth + cfg.band:
            b.role = "footer"

    patterns = sorted(f"{k[0]}: {k[2]!r}" for k in repeated)
    return header_depth, footer_depth, patterns


def _find_footnote_separator(page: PageInfo, cfg: LayoutConfig) -> float | None:
    """y of the lowest isolated short rule in the lower half of the page, if any."""
    candidates: list[HLine] = []
    for line in page.hlines:
        if line.y < page.height / 2:
            continue
        ratio = line.width / page.width
        if not cfg.separator_min_width_ratio <= ratio <= cfg.separator_max_width_ratio:
            continue
        # Table rows draw several segments at the same height; a separator stands alone.
        neighbours = sum(1 for other in page.hlines if abs(other.y - line.y) < cfg.band)
        if neighbours == 1:  # only itself
            candidates.append(line)
    return max((c.y for c in candidates), default=None)


def _tag_footnotes(
    blocks: list[ParsedBlock],
    pages: dict[int, PageInfo],
    body_size: float,
    cfg: LayoutConfig,
) -> list[int]:
    """Step 3. Returns the pages where footnotes were found."""
    separators = {n: _find_footnote_separator(p, cfg) for n, p in pages.items()}
    found: set[int] = set()
    for b in blocks:
        sep_y = separators[b.page_number]
        if (
            b.role == "body"
            and sep_y is not None
            and b.y0 >= sep_y - 1
            and b.font_size <= body_size - cfg.footnote_size_delta
        ):
            b.role = "footnote"
            found.add(b.page_number)
    return sorted(found)


def annotate_layout(
    blocks: list[ParsedBlock],
    pages: list[PageInfo],
    config: LayoutConfig | None = None,
) -> LayoutReport:
    """Set `role` on furniture blocks in place and report what was learned."""
    cfg = config or LayoutConfig()
    by_number = {p.number: p for p in pages}

    header_depth, footer_depth, patterns = _tag_running_furniture(blocks, by_number, cfg)
    # Baseline from the remaining text only, so furniture can't skew it.
    body_size = body_font_size([b for b in blocks if b.role == "body"])
    footnote_pages = _tag_footnotes(blocks, by_number, body_size, cfg)

    report = LayoutReport(
        body_font_size=body_size,
        header_depth=round(header_depth, 1),
        footer_depth=round(footer_depth, 1),
        repeated_patterns=patterns,
        footnote_pages=footnote_pages,
        role_counts=Counter(b.role for b in blocks),
    )
    logger.info(
        "Layout: body=%.1fpt header_depth=%.0fpt footer_depth=%.0fpt roles=%s",
        report.body_font_size,
        report.header_depth,
        report.footer_depth,
        dict(report.role_counts),
    )
    return report
