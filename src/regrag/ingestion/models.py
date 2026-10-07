"""Data model shared by the ingestion stages.

Kept separate from the parser so that every stage (extraction, layout analysis,
structure) can import it without importing each other.

These are Pydantic models because they cross a boundary: they are written to
`data/processed/` and read back later, possibly by another process or a newer
version of the code. Data read from disk is external input, so it is validated:
a corrupted or hand-edited file fails at load time, naming the bad field,
instead of failing three stages later.

Validation runs when a model is *created* (or loaded). Assigning to a field
afterwards (`block.role = "footer"`) is not re-validated, which keeps the
parser stages that annotate blocks in place cheap.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# (x0, y0, x1, y1) in PDF points; origin is the top-left corner, y grows downward.
BBox = tuple[float, float, float, float]

# What a block *is* on the page. Furniture is tagged, never deleted: downstream
# stages decide what to use, and the tags stay inspectable for debugging.
Role = Literal["body", "header", "footer", "footnote", "toc", "noise"]

BlockType = Literal["heading", "text"]

# Document id: becomes a directory name under data/processed/ and a prefix of
# every block id, so it is restricted to a safe slug ("unece-r107-rev9").
# This also rules out path tricks like "../../etc".
DocId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,99}$")]


class Model(BaseModel):
    """Base for all ingestion models: unknown fields are an error, not silently dropped.

    That turns a typo in a file ("fotnote_pages") or a field from an incompatible
    schema into a clear validation error.
    """

    model_config = ConfigDict(extra="forbid")


class HLine(Model):
    """A horizontal rule drawn on the page (table border, separator, underline)."""

    model_config = ConfigDict(frozen=True)

    y: float
    x0: float
    x1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0


class PageInfo(Model):
    number: int = Field(ge=1)  # 1-based
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    hlines: list[HLine] = Field(default_factory=list)


class ParsedBlock(Model):
    # "<doc_id>:<index>", e.g. "unece-r107-rev9:00412". Chunks, citations and
    # evaluation sets refer to blocks by id. Stable for a given source PDF and
    # parser version; a parser change that adds or removes blocks renumbers them.
    id: str | None = None
    text: str
    page_number: int = Field(ge=1)
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
    breadcrumb: list[str] = Field(default_factory=list)  # scope title, then section path

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]


class LayoutReport(Model):
    """What layout analysis learned about a document. Useful for logging and debugging."""

    body_font_size: float
    header_depth: float  # pt from the top edge covered by running headers (0 = none found)
    footer_depth: float  # pt from the bottom edge covered by running footers
    repeated_patterns: list[str] = Field(default_factory=list)
    footnote_pages: list[int] = Field(default_factory=list)
    role_counts: dict[str, int] = Field(default_factory=dict)


class StyleReport(Model):
    body_style: str  # e.g. "10.0 regular"
    heading_levels: dict[str, int] = Field(default_factory=dict)  # style -> level


class ScopeInfo(Model):
    id: str  # "Annex 3"
    title: str  # "Requirements to be met by all vehicles"
    first_page: int


class StructureReport(Model):
    scopes: list[ScopeInfo] = Field(default_factory=list)
    numbered_blocks: int = 0
    heading_blocks: int = 0
    # Numbering that goes backwards without a scope change, e.g. 7.6 after 7.9.
    # Usually a parsing problem (a table cell or list item that looks numbered).
    backward_jumps: list[str] = Field(default_factory=list)


class ParsedDocument(Model):
    doc_id: DocId
    source_path: str
    pages: list[PageInfo]
    blocks: list[ParsedBlock] = Field(default_factory=list)
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
