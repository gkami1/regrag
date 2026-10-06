"""Unit tests for scopes, the numbering tree and breadcrumbs."""

from regrag.ingestion.models import ParsedBlock
from regrag.ingestion.structure import annotate_structure


def block(text: str, level: int | None = None, page: int = 1) -> ParsedBlock:
    b = ParsedBlock(text=text, page_number=page, bbox=(0, 0, 1, 1))
    b.heading_level = level
    return b


def test_numbering_builds_nested_breadcrumbs():
    blocks = [
        block("7. Requirements", level=1),
        block("7.6. Exits"),
        block("7.6.1. Number of exits"),
        block("7.6.1.1. The minimum number of doors in a vehicle shall be two."),
        block("(a) a list item without a number"),
        block("7.7. Interior arrangements"),
    ]
    annotate_structure(blocks)

    assert blocks[3].breadcrumb == [
        "Regulation",
        "7 Requirements",
        "7.6 Exits",
        "7.6.1 Number of exits",
        "7.6.1.1 The minimum number of doors in a vehicle…",
    ]
    assert blocks[4].breadcrumb == blocks[3].breadcrumb  # unnumbered inherits context
    assert blocks[5].breadcrumb == ["Regulation", "7 Requirements", "7.7 Interior arrangements"]


def test_label_vs_content():
    blocks = [
        block("7.6.1. Number of exits"),
        block("7.6.5.1.1. Override all other door controls;"),
        block("7.2.3.2. (Reserved)"),
        block("7.7.8.1.1.1. 200 mm in the case of Class I, II, A or B; or"),
    ]
    annotate_structure(blocks)
    assert [b.block_type for b in blocks] == ["heading", "text", "text", "text"]


def test_annex_opens_scope_and_restarts_numbering():
    blocks = [
        block("1. Scope", level=1),
        block("1.1. This Regulation applies to vehicles."),
        block("Annex 8", level=1, page=9),
        block("Accommodation and accessibility for passengers with", level=1, page=9),
        block("reduced mobility", level=1, page=9),
        block("Figure 1 Wheelchair space", level=3, page=9),  # caption, not title
        block("1. General", page=9),
    ]
    report = annotate_structure(blocks)

    assert [s.id for s in report.scopes] == ["Regulation", "Annex 8"]
    assert report.scopes[1].title == (
        "Accommodation and accessibility for passengers with reduced mobility"
    )
    assert blocks[-1].scope == "Annex 8"
    assert blocks[-1].breadcrumb == [
        "Annex 8: Accommodation and accessibility for passengers with reduced mobility",
        "1 General",
    ]
    assert report.backward_jumps == []  # 1 after 1.1 is fine: the scope changed


def test_scope_needs_heading_style():
    # A table cell reading "Annex 4 ..." in body font must not open a scope.
    blocks = [block("1. Scope", level=1), block("Annex 4 - see figure for details")]
    annotate_structure(blocks)
    assert blocks[1].scope == "Regulation"


def test_hyphenated_title_lines_are_joined():
    blocks = [
        block("Annex 6", level=1),
        block("Guidelines for measuring the closing forces of power-", level=1),
        block("operated doors", level=1),
    ]
    report = annotate_structure(blocks)
    assert report.scopes[0].title.endswith("power-operated doors")


def test_backward_jump_is_reported():
    blocks = [block("7.9. Articulated section"), block("7.6. Exits")]
    report = annotate_structure(blocks)
    assert len(report.backward_jumps) == 1
