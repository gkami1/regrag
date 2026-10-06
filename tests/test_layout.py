"""Unit tests for layout analysis on synthetic pages (no PDF needed).

Each test builds a tiny fake document: A4 pages (595 x 842 pt) with blocks
placed where a real PDF would put them.
"""

from regrag.ingestion.layout import annotate_layout, normalize_pattern
from regrag.ingestion.models import HLine, PageInfo, ParsedBlock

W, H = 595.0, 842.0
TOPICS = [
    "exits",
    "seats",
    "doors",
    "steps",
    "ramps",
    "lifts",
    "lighting",
    "gangways",
    "handrails",
    "windows",
    "hatches",
    "batteries",
]


def block(text: str, page: int, y0: float, y1: float, size: float = 10.0) -> ParsedBlock:
    return ParsedBlock(text=text, page_number=page, bbox=(70.0, y0, 520.0, y1), font_size=size)


def make_pages(n: int, hlines: dict[int, list[HLine]] | None = None) -> list[PageInfo]:
    hlines = hlines or {}
    return [
        PageInfo(number=i, width=W, height=H, hlines=hlines.get(i, [])) for i in range(1, n + 1)
    ]


def standard_doc(n_pages: int = 10) -> list[ParsedBlock]:
    """Every page: running header, a body paragraph, page number at the bottom."""
    blocks = []
    for p in range(1, n_pages + 1):
        blocks.append(block("E/ECE/324/Rev.3/Add.106/Rev.9", p, 34, 55))
        # Must differ in words, not only digits: digits are normalised away, so
        # "1.1. Text" and "2.1. Text" would look like a running header.
        blocks.append(block(f"Body paragraph about {TOPICS[p % len(TOPICS)]}.", p, 80, 120))
        blocks.append(block(str(p), p, 804, 814))
    return blocks


def roles(blocks: list[ParsedBlock]) -> list[str]:
    return [b.role for b in blocks]


def test_normalize_pattern_merges_page_numbers():
    assert normalize_pattern("Page 5") == normalize_pattern("Page  47")


def test_running_header_and_page_numbers_are_tagged():
    blocks = standard_doc()
    report = annotate_layout(blocks, make_pages(10))

    assert roles(blocks[0::3]) == ["header"] * 10
    assert roles(blocks[1::3]) == ["body"] * 10
    assert roles(blocks[2::3]) == ["footer"] * 10
    assert report.header_depth == 55
    assert report.body_font_size == 10.0


def test_one_off_block_inside_learned_header_zone_is_tagged():
    blocks = standard_doc()
    stamp = block("Annex 3", 4, 40, 52)  # appears once, but sits in the header zone
    blocks.append(stamp)
    annotate_layout(blocks, make_pages(10))
    assert stamp.role == "header"


def test_short_document_has_no_furniture():
    blocks = standard_doc(n_pages=2)
    annotate_layout(blocks, make_pages(2))
    assert set(roles(blocks)) == {"body"}


def test_repeated_text_mid_page_is_not_furniture():
    blocks = standard_doc()
    blocks += [block("(Reserved)", p, 400, 412) for p in range(1, 11)]
    annotate_layout(blocks, make_pages(10))
    assert all(b.role == "body" for b in blocks if b.text == "(Reserved)")


def test_footnote_below_isolated_separator():
    blocks = standard_doc()
    note = block("1 As defined in the Consolidated Resolution.", 5, 752, 785, size=9.0)
    body_below = block("7.5.1. Engine compartment", 5, 760, 772, size=10.0)
    blocks += [note, body_below]
    pages = make_pages(10, {5: [HLine(y=747, x0=70, x1=144)]})

    report = annotate_layout(blocks, pages)

    assert note.role == "footnote"
    assert body_below.role == "body"  # body-size text is never a footnote
    assert report.footnote_pages == [5]


def test_table_rules_are_not_separators():
    y = 600.0
    blocks = standard_doc()
    cell = block("Tc1 250 mm", 5, y + 5, y + 15, size=9.0)
    blocks.append(cell)
    row = [HLine(y=y, x0=70 + i * 100, x1=170 + i * 100) for i in range(3)]  # 3 cells, same y
    annotate_layout(blocks, make_pages(10, {5: row}))
    assert cell.role == "body"
