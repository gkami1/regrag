"""Unit tests for document-relative heading style detection."""

from regrag.ingestion.models import ParsedBlock
from regrag.ingestion.styles import annotate_styles


def block(text: str, size: float = 10.0, bold: bool = False) -> ParsedBlock:
    return ParsedBlock(text=text, page_number=1, bbox=(0, 0, 1, 1), font_size=size, is_bold=bold)


BODY = [block("Body text of a regulation paragraph. " * 20) for _ in range(30)]


def test_levels_are_ranked_relative_to_body():
    h1 = block("Annex 3 Requirements to be met by all vehicles", 14.0, bold=True)
    h2 = block("Revision 9 of the regulation document title", 12.0, bold=True)
    h3 = block("Figure 1 Access to service doors and gangways", 10.0, bold=True)
    report = annotate_styles([*BODY, h1, h2, h3])

    assert (h1.heading_level, h2.heading_level, h3.heading_level) == (1, 2, 3)
    assert all(b.heading_level is None for b in BODY)
    assert report.body_style == "10.0 regular"


def test_same_sizes_mean_different_things_in_a_larger_document():
    # In a 14pt document, 14pt bold is merely emphasis and 18pt is the top level.
    body = [block("Large print body text paragraph. " * 20, 14.0) for _ in range(30)]
    big = block("Chapter heading in a large print document", 18.0, bold=True)
    emphasis = block("Emphasised heading at body size in bold", 14.0, bold=True)
    annotate_styles([*body, big, emphasis])
    assert (big.heading_level, emphasis.heading_level) == (1, 2)


def test_tiny_fragments_in_odd_sizes_are_not_headings():
    formula = block("(t)dt F T", 12.1)  # equation fragment, too little text to be a style
    annotate_styles([*BODY, formula])
    assert formula.heading_level is None


def test_common_style_is_not_a_heading_style():
    # A second style covering lots of text is another body style, not headings.
    bold_body = [block("Bold paragraph used everywhere. " * 20, 10.0, bold=True) for _ in range(10)]
    annotate_styles([*BODY, *bold_body])
    assert all(b.heading_level is None for b in bold_body)
