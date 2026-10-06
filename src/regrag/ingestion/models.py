"""Data model shared by the ingestion stages.

Kept separate from the parser so that every stage (extraction, layout analysis,
structure) can import it without importing each other.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

# (x0, y0, x1, y1) in PDF points; origin is the top-left corner, y grows downward.
BBox = tuple[float, float, float, float]

# What a block *is* on the page. Furniture is tagged, never deleted: downstream
# stages decide what to use, and the tags stay inspectable for debugging.
Role = Literal["body", "header", "footer", "footnote", "toc", "noise"]

BlockType = Literal["heading", "text"]


@dataclass(frozen=True)
class HLine:
    """A horizontal rule drawn on the page (table border, separator, underline)."""

    y: float
    x0: float
    x1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass
class PageInfo:
    number: int  # 1-based
    width: float
    height: float
    hlines: list[HLine] = field(default_factory=list)


@dataclass
class ParsedBlock:
    text: str
    page_number: int
    bbox: BBox
    font_size: float = 0.0  # dominant size, weighted by characters
    is_bold: bool = False  # majority of characters are bold
    role: Role = "body"
    # Set by style analysis: rank of this block's font style among heading styles
    # (1 = most prominent), None for body-styled text.
    heading_level: int | None = None
    # Set by structure analysis.
    block_type: BlockType = "text"
    section_number: str | None = None  # "3.10.12" or None
    scope: str | None = None  # "Regulation", "Annex 3", "Annex 1 - Part 1 - Appendix 2"
    breadcrumb: list[str] = field(default_factory=list)  # scope title, then section path

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]


@dataclass
class LayoutReport:
    """What layout analysis learned about a document. Useful for logging and debugging."""

    body_font_size: float
    header_depth: float  # pt from the top edge covered by running headers (0 = none found)
    footer_depth: float  # pt from the bottom edge covered by running footers
    repeated_patterns: list[str] = field(default_factory=list)
    footnote_pages: list[int] = field(default_factory=list)
    role_counts: Counter[str] = field(default_factory=Counter)


@dataclass
class StyleReport:
    body_style: str  # e.g. "10.0 regular"
    heading_levels: dict[str, int] = field(default_factory=dict)  # style -> level


@dataclass
class ScopeInfo:
    id: str  # "Annex 3"
    title: str  # "Requirements to be met by all vehicles"
    first_page: int


@dataclass
class StructureReport:
    scopes: list[ScopeInfo] = field(default_factory=list)
    numbered_blocks: int = 0
    heading_blocks: int = 0
    # Numbering that goes backwards without a scope change, e.g. 7.6 after 7.9.
    # Usually a parsing problem (a table cell or list item that looks numbered).
    backward_jumps: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    source_path: str
    pages: list[PageInfo]
    blocks: list[ParsedBlock] = field(default_factory=list)
    layout: LayoutReport | None = None
    styles: StyleReport | None = None
    structure: StructureReport | None = None

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    def body_blocks(self) -> list[ParsedBlock]:
        return [b for b in self.blocks if b.role == "body"]

    def to_text(self) -> str:
        return "\n\n".join(b.text for b in self.body_blocks())
